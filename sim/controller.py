"""Franka control wrapper: smooth end-effector motion and gripper control on a built scene."""

from typing import Optional

import numpy as np

import genesis as gs

GRIPPER_OPEN = 0.04  # meters, per-finger travel at fully open
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN_THRESHOLD = 0.02  # meters, avg finger pos above this counts as "open"

DEFAULT_N_WAYPOINTS = 10
DEFAULT_SETTLE_STEPS = 15
DEFAULT_FINAL_SETTLE_STEPS = 90  # extra steps held at the last waypoint so the PD controller converges


def get_ee_pos(scene: gs.Scene) -> list[float]:
    """Current end-effector (hand link) position in world frame, meters."""
    return scene.hand.get_pos().tolist()


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
    scene: gs.Scene, action: str, settle_steps: int = DEFAULT_SETTLE_STEPS
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
    n_waypoints: int = DEFAULT_N_WAYPOINTS,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> list[float]:
    """Move the end-effector to target_pos along a straight-line path, IK-solved per waypoint.

    If gripper_state is given, the gripper target is set once at the start and held
    for the whole motion (it converges concurrently with the arm). Intermediate waypoints
    only get `settle_steps` each to keep the motion smooth; the final waypoint gets extra
    `final_settle_steps` so the PD controller actually converges before reporting error.
    Returns the final reached end-effector position (world frame, meters).
    """
    if gripper_state is not None:
        _set_gripper_target(scene, gripper_state)

    start_pos = np.array(get_ee_pos(scene))
    end_pos = np.array(target_pos, dtype=float)

    for i in range(1, n_waypoints + 1):
        alpha = i / n_waypoints
        waypoint = start_pos + alpha * (end_pos - start_pos)
        qpos = scene.franka.inverse_kinematics(link=scene.hand, pos=waypoint.tolist())
        scene.franka.control_dofs_position(qpos[scene.arm_dofs], dofs_idx_local=scene.arm_dofs)
        steps = settle_steps if i < n_waypoints else final_settle_steps
        for _ in range(steps):
            scene.step()

    return get_ee_pos(scene)
