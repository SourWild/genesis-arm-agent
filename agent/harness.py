"""LLM tool-calling main loop for the Franka pick-place agent (Kimi K2 via OpenAI-compatible API)."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from openai import APIError, OpenAI

from agent import tools
from agent.prompts import SYSTEM_PROMPT
from sim import controller as sim_controller
from sim.scene import build_scene
from utils.voice import speak

load_dotenv()

MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
MOONSHOT_MODEL = os.environ.get("MOONSHOT_MODEL", "kimi-k2.6")

MAX_STEPS = 20
MAX_API_RETRIES = 3
MAX_MALFORMED_TOOL_CALL_RETRIES = 3
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

COLOR_ZH = {"red": "红", "green": "绿", "blue": "蓝"}
# How close the EE needs to be (xy, meters) to a block/target to credit a grasp/placement
# to it, when announcing which one just happened.
GRASP_ANNOUNCE_RADIUS_M = 0.05
PLACE_ANNOUNCE_RADIUS_M = 0.08
VOICE_FAILURE_MSG_MAX_CHARS = 40


def _nearest_color(entities: dict[str, Any], ee_xy: list[float]) -> tuple[Optional[str], float]:
    """Nearest entity (by xy distance) among a {color: RigidEntity} dict, e.g. scene.blocks."""
    best_color, best_dist = None, float("inf")
    for color, entity in entities.items():
        xy = entity.get_pos().tolist()[:2]
        dist = float(np.linalg.norm(np.array(xy) - np.array(ee_xy)))
        if dist < best_dist:
            best_color, best_dist = color, dist
    return best_color, best_dist


class ToolExecutor:
    """Binds a built scene to the 5 tools so the LLM only ever sees TOOLS_SCHEMA's declared args."""

    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        method = getattr(self, tool_name, None)
        if method is None:
            return {"success": False, "message": f"未知 tool: {tool_name!r}", "data": None}
        return method(**args)

    def get_scene_state(self) -> dict[str, Any]:
        return tools.get_scene_state(self.scene)

    def get_camera_image(self, view: str) -> dict[str, Any]:
        return tools.get_camera_image(self.scene, view)

    def move_ee(self, x: float, y: float, z: float, gripper_state: str) -> dict[str, Any]:
        return tools.move_ee(self.scene, x, y, z, gripper_state)

    def control_gripper(self, action: str) -> dict[str, Any]:
        return tools.control_gripper(self.scene, action)

    def check_task_success(self, block_color: str, target_color: str) -> dict[str, Any]:
        return tools.check_task_success(self.scene, block_color, target_color)


class TraceWriter:
    """Appends one JSON object per line to logs/{timestamp}.jsonl."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now().isoformat(), **record}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _call_llm_with_retry(client: OpenAI, messages: list[dict], max_retries: int = MAX_API_RETRIES):
    """Call chat.completions.create with exponential-backoff retry on API errors."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MOONSHOT_MODEL,
                messages=messages,
                tools=tools.TOOLS_SCHEMA,
                tool_choice="auto",
            )
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"LLM API 调用失败，重试 {max_retries} 次后仍然失败: {last_error}") from last_error


def run_episode(
    instruction: str,
    max_steps: int = MAX_STEPS,
    seed: Optional[int] = None,
    show_viewer: bool = False,
    block_color: Optional[str] = None,
    target_color: Optional[str] = None,
) -> dict[str, Any]:
    """Run one instruction through the LLM tool-calling loop against a fresh scene.

    Returns {"success": bool, "message": str, "steps_used": int, "trace_path": str}.
    `success` is the LLM's own self-report (it decided it was done). If `block_color` and
    `target_color` are given, an objective `ground_truth` dict (from tools.check_task_success)
    is also included — evaluate.py uses this instead of trusting the LLM's self-report.
    """
    if not MOONSHOT_API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY 未设置，请在 .env 里配置")

    client = OpenAI(api_key=MOONSHOT_API_KEY, base_url=MOONSHOT_BASE_URL)

    scene = build_scene(show_viewer=show_viewer, seed=seed)
    executor = ToolExecutor(scene)

    trace_path = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    trace = TraceWriter(trace_path)
    trace.write({"type": "episode_start", "instruction": instruction, "model": MOONSHOT_MODEL})
    speak("收到指令，开始执行")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    malformed_retries = 0
    holding_block: Optional[str] = None  # color of the block currently believed to be grasped
    result = {"success": False, "message": "未完成：达到最大步数限制", "steps_used": max_steps}

    def finalize(result: dict[str, Any]) -> dict[str, Any]:
        if block_color is not None and target_color is not None:
            verdict = tools.check_task_success(scene, block_color, target_color)
            result["ground_truth"] = verdict["data"]
        result["trace_path"] = str(trace_path)
        trace.write({"type": "episode_end", **result})
        if result["success"]:
            speak("任务完成")
        else:
            speak(f"任务失败，原因：{result['message'][:VOICE_FAILURE_MSG_MAX_CHARS]}")
        return result

    try:
        for step in range(1, max_steps + 1):
            trace.write({"type": "llm_request", "step": step, "n_messages": len(messages)})
            response = _call_llm_with_retry(client, messages)
            choice = response.choices[0].message

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": choice.content}
            if choice.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.tool_calls
                ]
            messages.append(assistant_msg)
            trace.write(
                {
                    "type": "llm_response",
                    "step": step,
                    "content": choice.content,
                    "tool_calls": assistant_msg.get("tool_calls"),
                }
            )

            if not choice.tool_calls:
                result = {"success": True, "message": choice.content or "(无文本内容)", "steps_used": step}
                break

            for tc in choice.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    malformed_retries += 1
                    tool_result = {
                        "success": False,
                        "message": f"tool 调用参数不是合法 JSON: {e}",
                        "data": None,
                    }
                    if malformed_retries > MAX_MALFORMED_TOOL_CALL_RETRIES:
                        result = {
                            "success": False,
                            "message": f"tool 调用格式连续错误超过 {MAX_MALFORMED_TOOL_CALL_RETRIES} 次，终止",
                            "steps_used": step,
                        }
                        return finalize(result)
                else:
                    trace.write({"type": "tool_call", "step": step, "name": name, "args": args})
                    tool_result = executor.dispatch(name, args)
                    trace.write({"type": "tool_result", "step": step, "name": name, "result": tool_result})

                    if name == "control_gripper" and tool_result["success"]:
                        ee_xy = sim_controller.get_ee_pos(scene)[:2]
                        if args.get("action") == "close":
                            color, dist = _nearest_color(scene.blocks, ee_xy)
                            if color is not None and dist < GRASP_ANNOUNCE_RADIUS_M:
                                holding_block = color
                                speak(f"已抓取{COLOR_ZH[color]}方块")
                        elif args.get("action") == "open" and holding_block is not None:
                            color, dist = _nearest_color(scene.targets, ee_xy)
                            if color is not None and dist < PLACE_ANNOUNCE_RADIUS_M:
                                speak(f"已放置到{COLOR_ZH[color]}区域")
                            holding_block = None

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
        else:
            trace.write({"type": "timeout", "max_steps": max_steps})

        return finalize(result)
    finally:
        trace.close()
