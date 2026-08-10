"""Franka control wrapper: smooth end-effector motion and gripper control on a built scene."""

import math
from typing import Any, Optional

import numpy as np

import genesis as gs

GRIPPER_OPEN = 0.04  # meters, per-finger travel at fully open
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN_THRESHOLD = 0.02  # meters, avg finger pos above this counts as "open"
EMPTY_GRASP_THRESHOLD = 0.01  # meters, avg finger pos below this after 'close' means likely nothing grasped

WAYPOINT_SPACING = 0.05  # meters, one IK waypoint per ~5cm of straight-line travel
DEFAULT_SETTLE_STEPS = 15
DEFAULT_FINAL_SETTLE_STEPS = 90  # extra steps held at the last waypoint so the PD controller converges
# Finger actuators are weak/force-limited: a standalone open<->close command covers the
# full ~4cm travel from rest, which needs far more steps than a single arm waypoint does.
DEFAULT_GRIPPER_SETTLE_STEPS = 90

# IK's own residual error at the final waypoint above this = target pose is not reachable
# under the fixed downward orientation (outside workspace or conflicts with the constraint).
IK_UNREACHABLE_ERROR_M = 0.01
# Physical (post-settle) position error above this, with IK itself reachable, means the
# arm got physically stuck (collision/contact) before reaching the commanded pose.
MOVE_ERROR_WARN_M = 0.02

APPROACH_HEIGHT = 0.05  # meters above the grasp/place height for the approach waypoint
LIFT_HEIGHT = 0.10  # meters to lift clear after grasping or releasing


def get_ee_pos(scene: gs.Scene) -> list[float]:
    """Current end-effector position in world frame, meters.

    This is the grasp point (finger midpoint), not the raw wrist/hand link — see
    `scene.grasp_point_offset` in sim/scene.py for why. move_ee's xyz refers to this point.
    """
    hand_pos = np.array(scene.hand.get_pos().tolist())
    return (hand_pos - np.array([0.0, 0.0, scene.grasp_point_offset])).tolist()


def get_gripper_state(scene: gs.Scene) -> str:
    """'open' or 'close', based on average finger dof position vs GRIPPER_OPEN_THRESHOLD."""
    finger_pos = scene.franka.get_dofs_position(dofs_idx_local=scene.finger_dofs)
    avg = float(finger_pos.mean())
    return "open" if avg > GRIPPER_OPEN_THRESHOLD else "close"


def _set_gripper_target(scene: gs.Scene, gripper_state: str) -> None:
    if gripper_state not in ("open", "close"):
        raise ValueError(f"gripper_state must be 'open' or 'close', got {gripper_state!r}")
    target = GRIPPER_OPEN if gripper_state == "open" else GRIPPER_CLOSED
    scene.franka.control_dofs_position([target, target], dofs_idx_local=scene.finger_dofs)


def control_gripper(
    scene: gs.Scene, action: str, settle_steps: int = DEFAULT_GRIPPER_SETTLE_STEPS
) -> list[float]:
    """Open or close the gripper and step the sim until settled. Returns final per-finger positions."""
    _set_gripper_target(scene, action)
    for _ in range(settle_steps):
        scene.step()
    return scene.franka.get_dofs_position(dofs_idx_local=scene.finger_dofs).tolist()


def move_ee_smooth(
    scene: gs.Scene,
    target_pos: tuple[float, float, float],
    gripper_state: Optional[str] = None,
    waypoint_spacing: float = WAYPOINT_SPACING,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> dict[str, Any]:
    """Move the end-effector to target_pos along a straight-line path, one IK waypoint per
    ~waypoint_spacing meters of travel (so long moves stay smooth without over-stepping short ones).

    If gripper_state is given, the gripper target is set once at the start and held for the
    whole motion (it converges concurrently with the arm). Orientation is held fixed at
    `scene.grasp_quat` (gripper straight down) for every waypoint — otherwise IK leaves
    rotation unconstrained and the gripper approaches objects at arbitrary angles, knocking
    them around instead of grasping them.

    Returns {"reached_pos": [x,y,z], "ik_reachable": bool, "ik_error_m": float}.
    `ik_reachable=False` means the IK solver itself couldn't satisfy the final target pose
    (target likely outside the reachable workspace under the fixed grasp orientation) —
    distinct from a physical/contact-limited shortfall, which shows up as reached_pos being
    far from target_pos despite ik_reachable=True.
    """
    if gripper_state is not None:
        _set_gripper_target(scene, gripper_state)

    start_pos = np.array(get_ee_pos(scene))  # grasp-point space
    end_pos = np.array(target_pos, dtype=float)
    hand_offset = np.array([0.0, 0.0, scene.grasp_point_offset])

    distance = float(np.linalg.norm(end_pos - start_pos))
    n_waypoints = max(1, math.ceil(distance / waypoint_spacing))

    ik_error_m = 0.0
    for i in range(1, n_waypoints + 1):
        alpha = i / n_waypoints
        waypoint = start_pos + alpha * (end_pos - start_pos)
        hand_target = waypoint + hand_offset  # grasp point -> wrist/hand link target for IK
        is_final = i == n_waypoints
        qpos, ik_err = scene.franka.inverse_kinematics(
            link=scene.hand, pos=hand_target.tolist(), quat=scene.grasp_quat, return_error=True
        )
        if is_final:
            ik_error_m = float(np.linalg.norm(ik_err[:3].tolist()))
        scene.franka.control_dofs_position(qpos[scene.arm_dofs], dofs_idx_local=scene.arm_dofs)
        for _ in range(final_settle_steps if is_final else settle_steps):
            scene.step()

    return {
        "reached_pos": get_ee_pos(scene),
        "ik_reachable": ik_error_m <= IK_UNREACHABLE_ERROR_M,
        "ik_error_m": ik_error_m,
    }


def pick(
    scene: gs.Scene,
    target_xyz: tuple[float, float, float],
    approach_height: float = APPROACH_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
) -> dict[str, Any]:
    """Fixed grasp macro: approach 5cm above target (open) -> descend (open) -> close -> lift 10cm.

    `target_xyz` is the grasp point (e.g. a block's center). Returns per-phase results plus
    the final reached position, so callers can inspect exactly where it got stuck if any
    phase came up short (unreachable IK, large position error, or an empty-looking grasp).
    """
    x, y, z = target_xyz
    phases: list[dict[str, Any]] = []

    approach = move_ee_smooth(scene, (x, y, z + approach_height), gripper_state="open")
    phases.append({"phase": "approach", **approach})

    descend = move_ee_smooth(scene, (x, y, z), gripper_state="open")
    phases.append({"phase": "descend", **descend})

    finger_pos = control_gripper(scene, "close")
    empty_grasp = (sum(finger_pos) / 2) < EMPTY_GRASP_THRESHOLD
    phases.append({"phase": "grasp", "finger_positions": finger_pos, "empty_grasp": empty_grasp})

    lift = move_ee_smooth(scene, (x, y, z + lift_height), gripper_state="close")
    phases.append({"phase": "lift", **lift})

    return {"phases": phases, "final_pos": lift["reached_pos"], "empty_grasp": empty_grasp}


def place(
    scene: gs.Scene,
    target_xyz: tuple[float, float, float],
    approach_height: float = APPROACH_HEIGHT,
    lift_height: float = LIFT_HEIGHT,
) -> dict[str, Any]:
    """Fixed place macro: approach 5cm above target (close) -> descend (close) -> open -> lift 10cm clear.

    `target_xyz` is the release point (e.g. a target zone's center). Returns per-phase
    results plus the final reached position.
    """
    x, y, z = target_xyz
    phases: list[dict[str, Any]] = []

    approach = move_ee_smooth(scene, (x, y, z + approach_height), gripper_state="close")
    phases.append({"phase": "approach", **approach})

    descend = move_ee_smooth(scene, (x, y, z), gripper_state="close")
    phases.append({"phase": "descend", **descend})

    finger_pos = control_gripper(scene, "open")
    phases.append({"phase": "release", "finger_positions": finger_pos})

    lift = move_ee_smooth(scene, (x, y, z + lift_height), gripper_state="open")
    phases.append({"phase": "lift", **lift})

    return {"phases": phases, "final_pos": lift["reached_pos"]}
