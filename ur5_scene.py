"""
ur5_scene.py — Shared Isaac Sim 6.0 scene helpers for the UR5 project.

Constants and builders that both `task1b_isaac_fk_validation.py` and
`task5_cartesian_motion.py` need: desk geometry, lighting rig, UR5 prim
paths, USD asset resolution, EE-prim search, and a 60 Hz step pacer.

Import this **after** `SimulationApp()` has constructed the kit app —
its body pulls Isaac Sim's `isaacsim.core.experimental.*` extension
namespace, which only loads when the simulator is up.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from isaacsim.core.experimental.objects import Cube, DistantLight, DomeLight
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

# ---------------------------------------------------------------------------
# USD asset source — set to a local .usd to skip the S3 stream
# ---------------------------------------------------------------------------
UR5_USD_LOCAL_PATH: str | None = None

# ---------------------------------------------------------------------------
# Desk geometry (warm wood, four legs)
# ---------------------------------------------------------------------------
TABLE_WIDTH:        float = 1.50
TABLE_DEPTH:        float = 1.20
TABLE_HEIGHT:       float = 0.75
DESK_TOP_THICKNESS: float = 0.05
DESK_LEG_THICKNESS: float = 0.07
DESK_LEG_INSET:     float = 0.05

ROBOT_BASE_Z:    float = TABLE_HEIGHT
ROBOT_PRIM_PATH: str   = "/World/UR5"
ROBOT_NAME:      str   = "ur5"

EE_CANDIDATE_PRIMS: list[str] = [
    f"{ROBOT_PRIM_PATH}/ee_link",
    f"{ROBOT_PRIM_PATH}/tool0",
    f"{ROBOT_PRIM_PATH}/wrist_3_link",
]
BASE_PRIM_PATH: str = f"{ROBOT_PRIM_PATH}/base_link"

# ---------------------------------------------------------------------------
# Step pacer — simulation_app.update() in Isaac Sim 6.0 is not throttled
# to physics rate, so an inner loop can blow past 60 Hz and read as a
# snap. ``make_step_pacer(simulation_app)`` returns a callable that pads
# each ``update()`` to one physics tick.
# ---------------------------------------------------------------------------
def make_step_pacer(simulation_app, dt: float = 1.0 / 60.0):
    """Return a ``step()`` closure that updates the app and sleeps until ``dt``."""
    def step() -> None:
        t0 = time.perf_counter()
        simulation_app.update()
        slack = dt - (time.perf_counter() - t0)
        if slack > 0.0:
            time.sleep(slack)
    return step


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------
def resolve_ur5_usd() -> str:
    """Return a USD path for the UR5 — local file if set, else the Nucleus S3 URL."""
    if UR5_USD_LOCAL_PATH:
        p = Path(UR5_USD_LOCAL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Local USD not found: {UR5_USD_LOCAL_PATH}")
        return str(p.resolve())
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Omniverse Nucleus / S3 not reachable; "
            "set UR5_USD_LOCAL_PATH to a local UR5 USD instead."
        )
    return f"{root}/Isaac/Robots/UniversalRobots/ur5/ur5.usd"


def find_ee_prim(stage) -> str:
    """Return the first valid EE prim path under ``ROBOT_PRIM_PATH``."""
    for path in EE_CANDIDATE_PRIMS:
        if stage.GetPrimAtPath(path).IsValid():
            return path
    raise RuntimeError(
        "Cannot find the EE prim. Expected one of:\n"
        + "\n".join(f"  {p}" for p in EE_CANDIDATE_PRIMS)
    )


# ---------------------------------------------------------------------------
# Scene builders — RTX renders MDL natively, so we only set colour where we
# author the geometry ourselves (the desk).
# ---------------------------------------------------------------------------
def build_lights() -> None:
    """Sky dome + warm key sun + cool fill (no overrides on the UR5)."""
    sky = DomeLight(paths="/World/Lights/Sky")
    sky.set_intensities([800.0])
    sky.set_colors([[0.92, 0.95, 1.00]])

    sun = DistantLight(
        paths="/World/Lights/Sun",
        angles=[2.0],
        orientations=np.array([[0.6532815, -0.2705981, 0.6532815, -0.2705981]]),
    )
    sun.set_intensities([3000.0])
    sun.set_colors([[1.00, 0.96, 0.88]])

    fill = DistantLight(
        paths="/World/Lights/Fill",
        angles=[5.0],
        orientations=np.array([[0.5, -0.5, -0.5, 0.5]]),
    )
    fill.set_intensities([600.0])
    fill.set_colors([[0.85, 0.90, 1.00]])
    print("[ur5_scene] lights placed (dome + sun + fill)")


def build_desk(stage) -> None:
    """Single Cube for the top + four Cube legs, all in warm wood color.

    ``Cube`` from ``isaacsim.core.experimental.objects`` wraps ``UsdGeom.Cube``
    whose ``size`` attribute defaults to **2** (USD spec). Pass ``sizes=[1.0]``
    so ``scales`` reads as actual metres.
    """
    Cube(
        paths="/World/Desk/Top",
        sizes=[1.0],
        positions=np.array([[0.0, 0.0, TABLE_HEIGHT - DESK_TOP_THICKNESS / 2.0]]),
        scales=np.array([[TABLE_WIDTH, TABLE_DEPTH, DESK_TOP_THICKNESS]]),
    )
    half_x = TABLE_WIDTH  / 2.0 - DESK_LEG_INSET - DESK_LEG_THICKNESS / 2.0
    half_y = TABLE_DEPTH  / 2.0 - DESK_LEG_INSET - DESK_LEG_THICKNESS / 2.0
    leg_h  = TABLE_HEIGHT - DESK_TOP_THICKNESS
    leg_z  = leg_h / 2.0
    for i, (sx, sy) in enumerate([(+1, +1), (+1, -1), (-1, +1), (-1, -1)]):
        Cube(
            paths=f"/World/Desk/Leg_{i}",
            sizes=[1.0],
            positions=np.array([[sx * half_x, sy * half_y, leg_z]]),
            scales=np.array([[DESK_LEG_THICKNESS, DESK_LEG_THICKNESS, leg_h]]),
        )

    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Desk/"):
            if UsdGeom.Gprim(prim):
                disp = UsdGeom.Gprim(prim).CreateDisplayColorPrimvar(
                    UsdGeom.Tokens.constant
                )
                disp.Set([Gf.Vec3f(0.55, 0.35, 0.18)])
    print("[ur5_scene] desk built (1 top + 4 legs)")
