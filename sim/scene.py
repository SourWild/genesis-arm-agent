"""Genesis pick-place scene: table + 3 colored blocks + 3 colored target zones + cameras."""

import random
from typing import Optional

import genesis as gs

TABLE_Z = 0.0  # meters, tabletop height in world frame
BLOCK_SIZE = 0.04  # meters, cube edge length
BLOCK_HALF = BLOCK_SIZE / 2
BLOCK_FOOTPRINT_RADIUS = BLOCK_HALF * (2**0.5)  # half-diagonal, for overlap checks

TARGET_RADIUS = 0.04  # meters (8cm diameter disk)
TARGET_HEIGHT = 0.004  # meters, thin disk

BLOCK_X_RANGE = (0.4, 0.6)
BLOCK_Y_RANGE = (-0.2, 0.2)
TARGET_X_RANGE = (0.4, 0.6)
TARGET_Y_RANGE = (-0.2, 0.2)

MIN_SEPARATION_MARGIN = 0.015  # meters, extra gap enforced between any two object footprints

COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.75, 0.2, 1.0),
    "blue": (0.15, 0.35, 0.9, 1.0),
}

FRANKA_XML = "xml/franka_emika_panda/panda.xml"
# Franka MJCF actuators aren't PD-reducible by default; explicit gains are
# required for control_dofs_position to converge (see Genesis quickstart).
ARM_KP = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
ARM_KV = [450, 450, 350, 350, 200, 200, 200]
FINGER_KP = [100, 100]
FINGER_KV = [10, 10]


def _sample_non_overlapping(
    existing: list[tuple[float, float, float]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    radius: float,
    rng: random.Random,
    max_attempts: int = 500,
) -> tuple[float, float]:
    """Rejection-sample an (x, y) at least `radius + other_radius + margin` from every existing object."""
    for _ in range(max_attempts):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        if all(
            ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5 >= radius + er + MIN_SEPARATION_MARGIN
            for ex, ey, er in existing
        ):
            return x, y
    raise RuntimeError(
        "Failed to sample a non-overlapping position after "
        f"{max_attempts} attempts; ranges too tight for object count/size."
    )


def _randomize_layout(rng: random.Random) -> dict[str, dict[str, tuple[float, float]]]:
    """Sample non-overlapping (x, y) positions for all 3 blocks and 3 target zones."""
    placed: list[tuple[float, float, float]] = []
    layout: dict[str, dict[str, tuple[float, float]]] = {"blocks": {}, "targets": {}}

    for color in COLORS:
        x, y = _sample_non_overlapping(placed, BLOCK_X_RANGE, BLOCK_Y_RANGE, BLOCK_FOOTPRINT_RADIUS, rng)
        placed.append((x, y, BLOCK_FOOTPRINT_RADIUS))
        layout["blocks"][color] = (x, y)

    for color in COLORS:
        x, y = _sample_non_overlapping(placed, TARGET_X_RANGE, TARGET_Y_RANGE, TARGET_RADIUS, rng)
        placed.append((x, y, TARGET_RADIUS))
        layout["targets"][color] = (x, y)

    return layout


def build_scene(show_viewer: bool = False, seed: Optional[int] = None) -> gs.Scene:
    """Build the pick-place scene: gray table, 3 colored blocks, 3 colored target zones, 2 cameras.

    Calls `gs.init()` internally, so this should be the only entry point used per process.
    The returned Scene has extra attributes attached for later phases to use:
    - `blocks` / `targets`: dict of color name -> RigidEntity
    - `cam_top` / `cam_side`: Camera handles
    - `franka` / `hand`: RigidEntity and end-effector link
    - `arm_dofs` / `finger_dofs`: dof index lists for arm and gripper control
    """
    gs.init(backend=gs.metal, precision="32")

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5, 1.0)),
    )

    rng = random.Random(seed)
    layout = _randomize_layout(rng)

    blocks = {}
    for color, rgba in COLORS.items():
        x, y = layout["blocks"][color]
        blocks[color] = scene.add_entity(
            gs.morphs.Box(pos=(x, y, TABLE_Z + BLOCK_HALF), size=(BLOCK_SIZE,) * 3),
            surface=gs.surfaces.Default(color=rgba),
        )

    targets = {}
    for color, rgba in COLORS.items():
        x, y = layout["targets"][color]
        targets[color] = scene.add_entity(
            gs.morphs.Cylinder(
                pos=(x, y, TABLE_Z + TARGET_HEIGHT / 2),
                radius=TARGET_RADIUS,
                height=TARGET_HEIGHT,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(rgba[0], rgba[1], rgba[2], 0.6)),
        )

    cam_top = scene.add_camera(
        res=(480, 480), pos=(0.5, 0.0, 1.2), lookat=(0.5, 0.0, 0.0), fov=50, GUI=False
    )
    cam_side = scene.add_camera(
        res=(480, 480), pos=(1.3, -0.9, 0.6), lookat=(0.5, 0.0, 0.1), fov=45, GUI=False
    )

    franka = scene.add_entity(gs.morphs.MJCF(file=FRANKA_XML))

    scene.build()

    arm_dofs = [franka.get_joint(f"joint{i}").dof_idx_local for i in range(1, 8)]
    finger_dofs = [
        franka.get_joint("finger_joint1").dof_idx_local,
        franka.get_joint("finger_joint2").dof_idx_local,
    ]
    franka.set_dofs_kp(ARM_KP + FINGER_KP, dofs_idx_local=arm_dofs + finger_dofs)
    franka.set_dofs_kv(ARM_KV + FINGER_KV, dofs_idx_local=arm_dofs + finger_dofs)

    scene.blocks = blocks
    scene.targets = targets
    scene.cam_top = cam_top
    scene.cam_side = cam_side
    scene.franka = franka
    scene.hand = franka.get_link("hand")
    scene.arm_dofs = arm_dofs
    scene.finger_dofs = finger_dofs
    scene._layout_rng = rng

    return scene


def reset_scene(scene: gs.Scene, seed: Optional[int] = None) -> None:
    """Randomize block and target positions in-place on an already-built scene.

    Uses a fresh RNG if `seed` is given, otherwise continues drawing from the
    scene's own RNG (so repeated calls without a seed keep producing new layouts).
    """
    rng = random.Random(seed) if seed is not None else scene._layout_rng
    layout = _randomize_layout(rng)

    for color, entity in scene.blocks.items():
        x, y = layout["blocks"][color]
        entity.set_pos((x, y, TABLE_Z + BLOCK_HALF))

    for color, entity in scene.targets.items():
        x, y = layout["targets"][color]
        entity.set_pos((x, y, TABLE_Z + TARGET_HEIGHT / 2))
