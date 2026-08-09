"""Phase 1 smoke test: move Franka end-effector to a target pose and cycle the gripper.

Verifies the Genesis + Metal backend pipeline works end-to-end on this Mac:
scene build, IK solve, position control, and the viewer window.
"""

import time

import numpy as np

import genesis as gs

TARGET_POS = (0.5, 0.0, 0.3)  # meters, world frame (table z=0)
GRIPPER_OPEN = 0.04  # meters, per-finger travel
GRIPPER_CLOSED = 0.0
N_GRIPPER_CYCLES = 3
SETTLE_STEPS = 150  # sim steps to let a control target converge
# Franka MJCF actuators aren't PD-reducible by default; explicit gains are
# required for control_dofs_position to converge (see Genesis quickstart).
ARM_KP = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
ARM_KV = [450, 450, 350, 350, 200, 200, 200]
FINGER_KP = [100, 100]
FINGER_KV = [10, 10]


def main() -> None:
    gs.init(backend=gs.metal, precision="32")

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    arm_dofs = [franka.get_joint(f"joint{i}").dof_idx_local for i in range(1, 8)]
    finger_dofs = [
        franka.get_joint("finger_joint1").dof_idx_local,
        franka.get_joint("finger_joint2").dof_idx_local,
    ]
    franka.set_dofs_kp(np.array(ARM_KP + FINGER_KP), dofs_idx_local=arm_dofs + finger_dofs)
    franka.set_dofs_kv(np.array(ARM_KV + FINGER_KV), dofs_idx_local=arm_dofs + finger_dofs)

    hand = franka.get_link("hand")
    qpos = franka.inverse_kinematics(link=hand, pos=TARGET_POS)
    franka.control_dofs_position(qpos[arm_dofs], dofs_idx_local=arm_dofs)
    for _ in range(SETTLE_STEPS):
        scene.step()

    ee_pos = hand.get_pos()
    print(f"[sanity_check] target pos: {TARGET_POS}, reached pos: {ee_pos.tolist()}")

    for cycle in range(N_GRIPPER_CYCLES):
        franka.control_dofs_position(
            [GRIPPER_OPEN, GRIPPER_OPEN], dofs_idx_local=finger_dofs
        )
        for _ in range(SETTLE_STEPS):
            scene.step()
        franka.control_dofs_position(
            [GRIPPER_CLOSED, GRIPPER_CLOSED], dofs_idx_local=finger_dofs
        )
        for _ in range(SETTLE_STEPS):
            scene.step()
        print(f"[sanity_check] gripper cycle {cycle + 1}/{N_GRIPPER_CYCLES} done")

    print("[sanity_check] SMOKE TEST OK — close the viewer window to exit.")
    time.sleep(8)


if __name__ == "__main__":
    main()
