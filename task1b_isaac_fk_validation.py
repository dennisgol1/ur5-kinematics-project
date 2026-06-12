#!/usr/bin/env python3
"""
task1b_isaac_fk_validation.py — Task 1b: FK Validation inside Isaac Sim
=========================================================================
Loads the UR5 robot into Isaac Sim, steps through the same 5 joint
configurations defined in task1_fk_validation.py, reads the end-effector
world pose directly from the simulator, and prints a side-by-side
comparison table against the DH-math FK results.

Usage
-----
# Headed (opens the viewport — recommended for seeing the robot move):
    /path/to/isaac-sim/python.sh task1b_isaac_fk_validation.py

# Headless (CI / SSH):
    /path/to/isaac-sim/python.sh task1b_isaac_fk_validation.py --headless

# Increase dwell time per config (default 2 s):
    /path/to/isaac-sim/python.sh task1b_isaac_fk_validation.py --dwell 4

Typical Isaac Sim python.sh locations
--------------------------------------
  Isaac Sim 5.x  : ~/.local/share/ov/pkg/isaac-sim-5.1.0/python.sh
  Isaac Sim 4.x  : ~/.local/share/ov/pkg/isaac-sim-4.x.x/python.sh

Notes
-----
• All Isaac Sim imports MUST come after SimulationApp() is constructed.
• The UR5 USD asset is pulled from the Omniverse Nucleus server.
  If Nucleus is not reachable, set UR5_USD_LOCAL_PATH below to a local copy.
• Joint ordering in Isaac Sim matches the DH convention: θ₁…θ₆
  = shoulder_pan → shoulder_lift → elbow → wrist_1 → wrist_2 → wrist_3.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CLI args — must be parsed BEFORE SimulationApp consumes sys.argv
# ---------------------------------------------------------------------------
import argparse
import sys

_parser = argparse.ArgumentParser(
    description="UR5 FK validation: DH math vs Isaac Sim",
    add_help=False,   # let Kit handle unknown flags like --/log/level
)
_parser.add_argument("--headless", action="store_true",
                     help="Run without the GUI viewport")
_parser.add_argument("--dwell", type=float, default=2.0,
                     help="Seconds to hold each configuration (default 2.0)")
_parser.add_argument("-h", "--help", action="store_true")
cli, _unknown = _parser.parse_known_args()

if cli.help:
    _parser.print_help()
    sys.exit(0)

# ---------------------------------------------------------------------------
# Launch SimulationApp — this MUST be the first Isaac Sim call
# ---------------------------------------------------------------------------
try:
    from isaacsim import SimulationApp           # Isaac Sim 5.x
except ImportError:
    from omni.isaac.kit import SimulationApp     # Isaac Sim 4.x fallback

simulation_app = SimulationApp(
    {
        "headless": cli.headless,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 3,
    }
)

# ---------------------------------------------------------------------------
# Remaining imports — all AFTER SimulationApp()
# ---------------------------------------------------------------------------
import math
import time
from pathlib import Path

import numpy as np

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_euler_angles

# Nucleus asset root (None when unreachable → use local fallback)
try:
    from omni.isaac.nucleus import get_assets_root_path
except ImportError:
    from omni.isaac.core.utils.nucleus import get_assets_root_path  # 4.x

# Optional viewport camera helper
try:
    from omni.isaac.core.utils.viewports import set_camera_view
    _HAS_VIEWPORT_HELPER = True
except ImportError:
    _HAS_VIEWPORT_HELPER = False

# ---------------------------------------------------------------------------
# Re-use FK math + test configs from Task 1
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from task1_fk_validation import (
    forward_kinematics,
    rotation_to_euler_rpy,
    TEST_CONFIGURATIONS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Uncomment and set this to a local .usd path if Nucleus is not running:
# UR5_USD_LOCAL_PATH = "/path/to/ur5.usd"
UR5_USD_LOCAL_PATH: str | None = None

ROBOT_PRIM_PATH = "/World/UR5"
ROBOT_NAME      = "ur5"

# Ordered prim paths to probe for the end-effector (tried in order).
# "ee_link" is the UR5 flange centre; "tool0" is the tool-side copy.
EE_CANDIDATE_PRIMS = [
    f"{ROBOT_PRIM_PATH}/ee_link",
    f"{ROBOT_PRIM_PATH}/tool0",
    f"{ROBOT_PRIM_PATH}/wrist_3_link",
]

# Physics settle steps after each joint-position command.
# More steps = more accurate pose read; fewer = faster iteration.
SETTLE_STEPS = 30


# ---------------------------------------------------------------------------
# Helper: resolve UR5 USD path
# ---------------------------------------------------------------------------

def _resolve_usd_path() -> str:
    if UR5_USD_LOCAL_PATH:
        p = Path(UR5_USD_LOCAL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Local USD not found: {UR5_USD_LOCAL_PATH}")
        return str(p.resolve())

    root = get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Omniverse Nucleus server is not reachable.\n"
            "Either start the local Nucleus service or set "
            "UR5_USD_LOCAL_PATH in this script to a local .usd file."
        )
    return f"{root}/Isaac/Robots/UniversalRobots/ur5/ur5.usd"


# ---------------------------------------------------------------------------
# Helper: find the EE prim (tries candidates in order)
# ---------------------------------------------------------------------------

def _find_ee_prim(stage) -> str:
    for path in EE_CANDIDATE_PRIMS:
        if stage.GetPrimAtPath(path).IsValid():
            return path
    raise RuntimeError(
        "Could not locate an end-effector prim on the loaded UR5 USD.\n"
        "Expected one of:\n" + "\n".join(f"  {p}" for p in EE_CANDIDATE_PRIMS) +
        "\nInspect the stage hierarchy in the Isaac Sim UI and update "
        "EE_CANDIDATE_PRIMS in this script."
    )


# ---------------------------------------------------------------------------
# Helper: convert Isaac Sim quaternion [w,x,y,z] → RPY degrees
# ---------------------------------------------------------------------------

def _quat_to_rpy_deg(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """
    Isaac Sim's quat_to_euler_angles returns roll/pitch/yaw in radians
    using the extrinsic XYZ (RPY) convention — same as our DH script.
    """
    rpy_rad: np.ndarray = quat_to_euler_angles(quat_wxyz)
    return (
        math.degrees(float(rpy_rad[0])),
        math.degrees(float(rpy_rad[1])),
        math.degrees(float(rpy_rad[2])),
    )


# ---------------------------------------------------------------------------
# Helper: pretty comparison table
# ---------------------------------------------------------------------------

_COL = 18   # column width for numeric values

def _print_comparison_table(results: list[dict]) -> None:
    sep  = "=" * 80
    sep2 = "-" * 80

    print("\n" + sep)
    print("  FK COMPARISON: DH MATH vs ISAAC SIM")
    print(sep)

    for r in results:
        print(f"\n  {r['label']}")
        angles_str = "  ".join(f"θ{i+1}={a:>6.1f}°"
                               for i, a in enumerate(r["angles_deg"]))
        print(f"  Joints : {angles_str}")
        print(sep2)

        # Header
        print(f"  {'Axis':<12} {'DH Math':>{_COL}}  {'Isaac Sim':>{_COL}}  {'|Error|':>{_COL}}")
        print(sep2)

        dh_p, sim_p = r["dh_pos_mm"], r["sim_pos_mm"]
        dh_r, sim_r = r["dh_rpy_deg"], r["sim_rpy_deg"]

        for axis, dv, sv in zip(["X (mm)", "Y (mm)", "Z (mm)"],
                                dh_p, sim_p):
            err = abs(dv - sv)
            flag = "  ✓" if err < 1.0 else "  !"
            print(f"  {axis:<12} {dv:>{_COL}.4f}  {sv:>{_COL}.4f}  "
                  f"{err:>{_COL}.4f}{flag}")

        for axis, dv, sv in zip(["Roll (°)", "Pitch (°)", "Yaw (°)"],
                                dh_r, sim_r):
            err = abs(dv - sv)
            # Wrap angular error to [-180, 180]
            err = min(err, abs(err - 360.0), abs(err + 360.0))
            flag = "  ✓" if err < 0.5 else "  !"
            print(f"  {axis:<12} {dv:>{_COL}.4f}  {sv:>{_COL}.4f}  "
                  f"{err:>{_COL}.4f}{flag}")

        print()

    print(sep)
    print("  ✓ = within tolerance (pos < 1 mm, angle < 0.5°)")
    print("  ! = outside tolerance — check joint-zero offsets or base frame")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import omni.usd

    usd_path = _resolve_usd_path()
    print(f"\n[task1b] UR5 USD : {usd_path}")
    print(f"[task1b] Headless: {cli.headless}")
    print(f"[task1b] Dwell   : {cli.dwell} s per config\n")

    # ── World & robot ──────────────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)

    robot: Robot = world.scene.add(
        Robot(
            prim_path=ROBOT_PRIM_PATH,
            name=ROBOT_NAME,
            usd_path=usd_path,
            position=np.array([0.0, 0.0, 0.0]),
        )
    )

    world.reset()

    # ── Camera (headed mode only) ──────────────────────────────────────────
    if not cli.headless and _HAS_VIEWPORT_HELPER:
        set_camera_view(
            eye=np.array([1.8, 1.8, 1.2]),
            target=np.array([0.0, 0.0, 0.5]),
            camera_prim_path="/OmniverseKit_Persp",
        )

    # ── Locate EE prim on the loaded stage ────────────────────────────────
    stage = omni.usd.get_context().get_stage()
    ee_prim_path = _find_ee_prim(stage)
    ee_xform     = XFormPrim(ee_prim_path)
    print(f"[task1b] Using EE prim: {ee_prim_path}\n")

    # ── Iterate over test configurations ──────────────────────────────────
    results: list[dict] = []

    for idx, (label, angles_deg) in enumerate(TEST_CONFIGURATIONS, start=1):
        angles_rad = np.array([math.radians(a) for a in angles_deg], dtype=np.float64)

        print(f"[{idx}/5] {label}")
        print(f"        Joints (deg): {angles_deg}")

        # Command joint positions
        robot.set_joint_positions(angles_rad)

        # Step physics to settle the articulation
        for _ in range(SETTLE_STEPS):
            world.step(render=not cli.headless)

        # Read EE world pose from Isaac Sim
        pos_sim, quat_wxyz = ee_xform.get_world_pose()

        # pos_sim is in metres (stage units = 1.0 m)
        sim_pos_mm = (
            float(pos_sim[0]) * 1000.0,
            float(pos_sim[1]) * 1000.0,
            float(pos_sim[2]) * 1000.0,
        )
        sim_rpy_deg = _quat_to_rpy_deg(quat_wxyz)

        # Compute DH FK
        T      = forward_kinematics(angles_rad.tolist())
        dh_pos = (T[0, 3] * 1000.0, T[1, 3] * 1000.0, T[2, 3] * 1000.0)
        dh_rpy = tuple(math.degrees(a) for a in rotation_to_euler_rpy(T[:3, :3]))

        results.append(
            {
                "label":      label,
                "angles_deg": angles_deg,
                "dh_pos_mm":  dh_pos,
                "sim_pos_mm": sim_pos_mm,
                "dh_rpy_deg": dh_rpy,
                "sim_rpy_deg": sim_rpy_deg,
            }
        )

        # Print quick per-config summary while simulation runs
        print(f"        DH  pos (mm): X={dh_pos[0]:8.3f}  Y={dh_pos[1]:8.3f}  Z={dh_pos[2]:8.3f}")
        print(f"        Sim pos (mm): X={sim_pos_mm[0]:8.3f}  Y={sim_pos_mm[1]:8.3f}  Z={sim_pos_mm[2]:8.3f}")
        print()

        # Dwell so the user can inspect the pose in the viewport
        if not cli.headless:
            dwell_steps = int(cli.dwell * 60)
            for _ in range(dwell_steps):
                world.step(render=True)
                simulation_app.update()
        else:
            time.sleep(cli.dwell)

    # ── Final comparison table ─────────────────────────────────────────────
    _print_comparison_table(results)

    # ── Keep the viewport open until the user closes it ───────────────────
    if not cli.headless:
        print("[task1b] All configs done — close the Isaac Sim window to exit.")
        while simulation_app.is_running():
            world.step(render=True)
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
