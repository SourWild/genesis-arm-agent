"""The 5 pick-place tools exposed to the LLM planner, plus their function-calling schema.

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
from sim.scene import TARGET_RADIUS

# "宽松成功判定": a block counts as placed once its center is within the target zone's
# radius, ignoring height — matches Phase 5's spec (< 4cm, which happens to equal TARGET_RADIUS).
SUCCESS_XY_THRESHOLD = TARGET_RADIUS


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
    """Tool 3: smoothly move the end-effector to (x, y, z) while driving the gripper.

    success=False (with the reached position still reported in data) in two distinct cases:
    - IK 不可达: the target pose itself has no solution under the fixed downward grasp
      orientation — usually outside the reachable workspace.
    - 位置偏差过大: IK found a solution but the arm didn't get there physically (e.g.
      blocked by a collision with the table or another block).
    """
    try:
        if gripper_state not in ("open", "close"):
            return _err(f"gripper_state 必须是 'open' 或 'close'，收到: {gripper_state!r}")
        target = (x, y, z)
        result = controller.move_ee_smooth(scene, target, gripper_state=gripper_state)
        reached = result["reached_pos"]
        error_xyz = [reached[i] - target[i] for i in range(3)]
        error_norm = float(np.linalg.norm(error_xyz))
        data = {
            "target_xyz": list(target),
            "reached_xyz": reached,
            "error_xyz": error_xyz,
            "error_norm_m": error_norm,
            "gripper_state": controller.get_gripper_state(scene),
            "ik_reachable": result["ik_reachable"],
        }
        if not result["ik_reachable"]:
            return {
                "success": False,
                "message": (
                    f"移动失败：目标位置 IK 不可达（残差 {result['ik_error_m'] * 1000:.1f}mm），"
                    "可能超出机械臂工作范围，建议换一个更靠近工作区中心的坐标重试"
                ),
                "data": data,
            }
        if error_norm > controller.MOVE_ERROR_WARN_M:
            return {
                "success": False,
                "message": (
                    f"移动完成但位置偏差过大（{error_norm * 1000:.1f}mm，超过 "
                    f"{controller.MOVE_ERROR_WARN_M * 1000:.0f}mm 容差），可能被方块或桌面卡住，"
                    "建议 get_scene_state 确认实际位置后重试"
                ),
                "data": data,
            }
        return _ok(f"移动完成，末端误差 {error_norm * 1000:.1f}mm", data)
    except Exception as e:
        return _err(f"move_ee 失败: {e}")


def control_gripper(scene: gs.Scene, action: str) -> dict[str, Any]:
    """Tool 4: open or close the gripper without moving the end-effector.

    success=False when closing lands almost fully shut (< 1cm between fingers) — that
    usually means the gripper closed on empty air instead of a block (抓空).
    """
    try:
        if action not in ("open", "close"):
            return _err(f"action 必须是 'open' 或 'close'，收到: {action!r}")
        finger_positions = controller.control_gripper(scene, action)
        data = {
            "action": action,
            "finger_positions": finger_positions,
            "gripper_state": controller.get_gripper_state(scene),
        }
        if action == "close" and (sum(finger_positions) / 2) < controller.EMPTY_GRASP_THRESHOLD:
            data["possible_empty_grasp"] = True
            return {
                "success": False,
                "message": (
                    f"gripper 已闭合但指间距几乎为 0（{sum(finger_positions) / 2 * 1000:.1f}mm），"
                    "大概率没有夹到方块（抓空），建议重新 get_scene_state 确认方块位置后重试抓取"
                ),
                "data": data,
            }
        return _ok(f"gripper {action} 完成", data)
    except Exception as e:
        return _err(f"control_gripper 失败: {e}")


def check_task_success(scene: gs.Scene, block_color: str, target_color: str) -> dict[str, Any]:
    """Tool 5: objectively check whether block_color's block is within the success threshold
    of target_color's target zone (xy distance < 4cm, height ignored — "宽松成功判定").

    Use this instead of eyeballing get_scene_state coordinates before declaring a task done.
    """
    try:
        if block_color not in scene.blocks:
            return _err(f"未知方块颜色: {block_color!r}，可选: {list(scene.blocks)}")
        if target_color not in scene.targets:
            return _err(f"未知目标区域颜色: {target_color!r}，可选: {list(scene.targets)}")
        block_xy = np.array(scene.blocks[block_color].get_pos().tolist()[:2])
        target_xy = np.array(scene.targets[target_color].get_pos().tolist()[:2])
        distance = float(np.linalg.norm(block_xy - target_xy))
        task_success = distance < SUCCESS_XY_THRESHOLD
        data = {
            "block_color": block_color,
            "target_color": target_color,
            "xy_distance_m": distance,
            "threshold_m": SUCCESS_XY_THRESHOLD,
            "task_success": task_success,
        }
        verdict = "达标" if task_success else "未达标"
        msg = (
            f"{block_color}方块距{target_color}目标区域中心 {distance * 1000:.1f}mm，"
            f"{verdict}（阈值 {SUCCESS_XY_THRESHOLD * 1000:.0f}mm）"
        )
        return _ok(msg, data)
    except Exception as e:
        return _err(f"check_task_success 失败: {e}")


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
    {
        "type": "function",
        "function": {
            "name": "check_task_success",
            "description": (
                "客观检查某个方块是否已经放进某个目标区域：判据是方块中心与目标区域中心的水平(xy)"
                "距离小于 4cm（忽略高度）。完成放置动作后，用这个工具确认，而不是自己心算坐标距离。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_color": {
                        "type": "string",
                        "enum": ["red", "green", "blue"],
                        "description": "要检查的方块颜色",
                    },
                    "target_color": {
                        "type": "string",
                        "enum": ["red", "green", "blue"],
                        "description": "要检查的目标区域颜色",
                    },
                },
                "required": ["block_color", "target_color"],
            },
        },
    },
]
