"""The 4 pick-place tools exposed to the LLM planner, plus their function-calling schema.

Every tool takes the built `scene` (see sim.scene.build_scene) as its first argument and
returns the project-wide unified format: {"success": bool, "message": str, "data": dict | None}.
The harness (Phase 4) binds `scene` via a dispatch table before handing tool calls to the LLM —
Kimi K2 / Claude only ever see the remaining arguments, described by TOOLS_SCHEMA below.
"""

import base64
from typing import Any, Optional

import imageio.v3 as iio
import numpy as np

import genesis as gs
from sim import controller


def _ok(message: str, data: Optional[dict] = None) -> dict[str, Any]:
    return {"success": True, "message": message, "data": data}


def _err(message: str) -> dict[str, Any]:
    return {"success": False, "message": message, "data": None}


def get_scene_state(scene: gs.Scene) -> dict[str, Any]:
    """Tool 1: snapshot of all blocks, targets, EE position, and gripper state."""
    try:
        blocks = [
            {"name": color, "xyz": entity.get_pos().tolist(), "color": color}
            for color, entity in scene.blocks.items()
        ]
        targets = [
            {"name": color, "xyz": entity.get_pos().tolist(), "color": color}
            for color, entity in scene.targets.items()
        ]
        data = {
            "blocks": blocks,
            "targets": targets,
            "ee_pos": controller.get_ee_pos(scene),
            "gripper_state": controller.get_gripper_state(scene),
        }
        return _ok("场景状态获取成功", data)
    except Exception as e:
        return _err(f"get_scene_state 失败: {e}")


def get_camera_image(scene: gs.Scene, view: str) -> dict[str, Any]:
    """Tool 2: render the 'top' or 'side' camera and return a base64-encoded PNG."""
    try:
        if view not in ("top", "side"):
            return _err(f"view 必须是 'top' 或 'side'，收到: {view!r}")
        camera = scene.cam_top if view == "top" else scene.cam_side
        rgb, _, _, _ = camera.render()
        png_bytes = iio.imwrite("<bytes>", rgb, extension=".png")
        image_b64 = base64.b64encode(png_bytes).decode("ascii")
        data = {"view": view, "format": "png", "image_base64": image_b64}
        return _ok(f"{view} 相机截图成功", data)
    except Exception as e:
        return _err(f"get_camera_image 失败: {e}")


def move_ee(scene: gs.Scene, x: float, y: float, z: float, gripper_state: str) -> dict[str, Any]:
    """Tool 3: smoothly move the end-effector to (x, y, z) while driving the gripper."""
    try:
        if gripper_state not in ("open", "close"):
            return _err(f"gripper_state 必须是 'open' 或 'close'，收到: {gripper_state!r}")
        target = (x, y, z)
        reached = controller.move_ee_smooth(scene, target, gripper_state=gripper_state)
        error_xyz = [reached[i] - target[i] for i in range(3)]
        error_norm = float(np.linalg.norm(error_xyz))
        data = {
            "target_xyz": list(target),
            "reached_xyz": reached,
            "error_xyz": error_xyz,
            "error_norm_m": error_norm,
            "gripper_state": controller.get_gripper_state(scene),
        }
        return _ok(f"移动完成，末端误差 {error_norm * 1000:.1f}mm", data)
    except Exception as e:
        return _err(f"move_ee 失败: {e}")


def control_gripper(scene: gs.Scene, action: str) -> dict[str, Any]:
    """Tool 4: open or close the gripper without moving the end-effector."""
    try:
        if action not in ("open", "close"):
            return _err(f"action 必须是 'open' 或 'close'，收到: {action!r}")
        finger_positions = controller.control_gripper(scene, action)
        data = {
            "action": action,
            "finger_positions": finger_positions,
            "gripper_state": controller.get_gripper_state(scene),
        }
        return _ok(f"gripper {action} 完成", data)
    except Exception as e:
        return _err(f"control_gripper 失败: {e}")


# OpenAI-compatible function-calling schema (Kimi K2 uses the same format).
TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_scene_state",
            "description": (
                "获取当前场景状态：所有方块和目标区域的名字/位置/颜色、"
                "机械臂末端(EE)当前坐标、夹爪开合状态。不需要参数。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_image",
            "description": "获取相机截图，用于视觉确认场景当前布局。'top' 是俯视图，'side' 是侧视图。",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": ["top", "side"],
                        "description": "相机视角",
                    }
                },
                "required": ["view"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_ee",
            "description": (
                "平滑移动机械臂末端(EE)到指定的世界坐标位置（单位：米，桌面 z=0，"
                "工作区 x∈[0.3,0.7]，y∈[-0.3,0.3]），移动过程中同时设置夹爪开合状态。"
                "返回实际到达位置与目标的误差。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "目标 x 坐标（米）"},
                    "y": {"type": "number", "description": "目标 y 坐标（米）"},
                    "z": {"type": "number", "description": "目标 z 坐标（米）"},
                    "gripper_state": {
                        "type": "string",
                        "enum": ["open", "close"],
                        "description": "移动过程中保持的夹爪状态",
                    },
                },
                "required": ["x", "y", "z", "gripper_state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_gripper",
            "description": "单独控制夹爪开合，不移动机械臂末端。通常用于抓取/释放方块前后的精细控制。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "close"],
                        "description": "夹爪动作",
                    }
                },
                "required": ["action"],
            },
        },
    },
]
