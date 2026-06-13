#!/usr/bin/env python3
"""
task1b_isaac_fk_validation.py — Task 1b: FK validation in Isaac Sim 6.0
=========================================================================
Scene layout
------------
  * A grid ground plane and three studio lights.
  * A simple wooden desk (1.5 m x 1.2 m x 0.75 m thick top + 4 legs).
  * A UR5 mounted on the desk surface, MDL-rendered by RTX.

What it does
------------
  1. Loads the UR5 USD from the Omniverse Nucleus / S3 asset library.
  2. Steps through the 5 test configurations from ``task1_fk_validation.py``.
  3. Reads the EE world pose, converts it into the robot-base frame.
  4. Compares position / RPY against the DH-math FK result.
  5. Prints a side-by-side table, then keeps the simulation open so the
     last pose stays visible.

Usage
-----
    ~/Simulators/isaacsim-6.0/python.sh task1b_isaac_fk_validation.py

    # Snap-move + short dwell (debugging / fast iteration):
    ~/Simulators/isaacsim-6.0/python.sh task1b_isaac_fk_validation.py \\
        --motion snap --dwell 0.5
"""
from __future__ import annotations

import argparse
import sys

_parser = argparse.ArgumentParser(
    description="UR5 FK validation: DH math vs Isaac Sim 6.0",
    add_help=False,
)
_parser.add_argument(
    "--dwell", type=float, default=3.0,
    help="Seconds to hold each configuration in the viewport (default 3.0)",
)
_parser.add_argument(
    "--motion", choices=("smooth", "snap"), default="smooth",
    help="'smooth' (cosine-eased interpolation, default) or 'snap'",
)
_parser.add_argument(
    "--motion-time", type=float, default=2.0,
    help="Seconds to travel between configs in smooth mode (default 2.0)",
)
cli, _unknown = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# SimulationApp — must be the first Isaac Sim call.
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width":  1920,
    "height": 1080,
})

# ---------------------------------------------------------------------------
# All remaining Isaac Sim / USD imports — AFTER SimulationApp().
# ---------------------------------------------------------------------------
import math
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdGeom

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager  # noqa: F401

from ur5_scene import (
    BASE_PRIM_PATH,
    ROBOT_NAME,
    ROBOT_PRIM_PATH,
    TABLE_HEIGHT,
    build_desk,
    build_lights,
    find_ee_prim,
    make_step_pacer,
    place_robot_on_desk,
    resolve_ur5_usd,
)

# Re-use FK math + test configs from Task 1.
sys.path.insert(0, str(Path(__file__).parent))
from task1_fk_validation import (
    forward_kinematics,
    rotation_to_euler_rpy,
    TEST_CONFIGURATIONS,
)

# ---------------------------------------------------------------------------
# Local constants (scene constants/builders live in ur5_scene.py)
# ---------------------------------------------------------------------------
SETTLE_STEPS = 60   # 1 s at 60 Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _smooth_blend(
    q_start: np.ndarray, q_end: np.ndarray, alpha: float,
) -> np.ndarray:
    """Cosine-eased blend between two joint vectors; alpha in [0,1]."""
    s = 0.5 - 0.5 * math.cos(math.pi * float(np.clip(alpha, 0.0, 1.0)))
    return (1.0 - s) * q_start + s * q_end


def _xform_world_pose(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """World pose of an Xformable prim as (pos[3], quat_wxyz[4]).

    Uses ``UsdGeom.Xformable.ComputeLocalToWorldTransform`` — works on every
    prim type (Xform, Mesh, etc.) without needing an Articulation wrapper.
    """
    prim = stage.GetPrimAtPath(prim_path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default(),
    )
    t = m.ExtractTranslation()
    r = m.ExtractRotationQuat()
    return (
        np.array([t[0], t[1], t[2]], dtype=float),
        np.array([r.GetReal(), *r.GetImaginary()], dtype=float),
    )


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([
            0.25 / s,
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ])


def _world_to_base(
    ee_pos_w:    np.ndarray,
    ee_quat_w:   np.ndarray,
    base_pos_w:  np.ndarray,
    base_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R_base = _quat_wxyz_to_rot(base_quat_w)
    R_ee   = _quat_wxyz_to_rot(ee_quat_w)
    R_base_T = R_base.T
    pos_rel  = R_base_T @ (ee_pos_w - base_pos_w)
    R_rel    = R_base_T @ R_ee
    return pos_rel, _rot_to_quat_wxyz(R_rel)


def _quat_wxyz_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion [w,x,y,z] -> intrinsic Roll-Pitch-Yaw (rad) — XYZ order."""
    return tuple(rotation_to_euler_rpy(_quat_wxyz_to_rot(q)))   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Comparison-table printer (unchanged)
# ---------------------------------------------------------------------------
_C          = 16
_POS_TOL_MM = 1.0
_RPY_TOL_DEG = 0.5
_ROW_META: list[tuple[str, bool]] = [
    ("X (mm)",    False),
    ("Y (mm)",    False),
    ("Z (mm)",    False),
    ("Roll (deg)",  True),
    ("Pitch (deg)", True),
    ("Yaw (deg)",   True),
]


def _print_table(results: list[dict]) -> None:
    sep  = "=" * 78
    sep2 = "-" * 78
    print("\n" + sep)
    print("  FK COMPARISON  -  DH Math  vs  Isaac Sim  (robot-base frame)")
    print(sep)
    for r in results:
        print(f"\n  {r['label']}")
        joints = "  ".join(f"theta{i+1}={a:>6.1f}deg"
                           for i, a in enumerate(r["angles_deg"]))
        print(f"  Joints : {joints}")
        print(sep2)
        print(f"  {'':12} {'DH Math':>{_C}}  {'Isaac Sim':>{_C}}  {'|Error|':>{_C}}")
        print(sep2)
        all_dh = (*r["dh_pos_mm"],  *r["dh_rpy_deg"])
        all_si = (*r["sim_pos_mm"], *r["sim_rpy_deg"])
        for (axis, is_angle), dv, sv in zip(_ROW_META, all_dh, all_si):
            err = abs(dv - sv)
            if is_angle:
                err = min(err, abs(err - 360.0), abs(err + 360.0))
            tol  = _RPY_TOL_DEG if is_angle else _POS_TOL_MM
            mark = "  OK" if err < tol else "  !!"
            print(f"  {axis:<12} {dv:>{_C}.4f}  {sv:>{_C}.4f}  {err:>{_C}.4f}{mark}")
        print()
    print(sep)
    print(f"  OK pos < {_POS_TOL_MM} mm  |  OK angle < {_RPY_TOL_DEG} deg  |  "
          f"!! = outside tolerance")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global Usd
    from pxr import Usd                  # imported here so it's after SimulationApp

    usd_path = resolve_ur5_usd()
    print(f"\n[task1b] UR5 asset: {usd_path}")
    print(f"[task1b] Desk height: {TABLE_HEIGHT} m -- robot base at "
          f"z = {TABLE_HEIGHT} m\n")

    stage = omni.usd.get_context().get_stage()

    # ---- Scene -----------------------------------------------------------
    GroundPlane(paths="/World/Ground")
    build_lights()
    build_desk(stage)

    stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PRIM_PATH)

    # Position UR5 so its base sits flush on the desk top.
    place_robot_on_desk(stage, ROBOT_PRIM_PATH, TABLE_HEIGHT)

    # ---- Simulation start ------------------------------------------------
    omni.timeline.get_timeline_interface().play()
    for _ in range(3):
        simulation_app.update()

    articulation = Articulation(paths=ROBOT_PRIM_PATH)
    print(f"[task1b] UR5 articulation: {articulation.num_joints} joints, "
          f"{articulation.num_links} links")

    # Locate EE prim now that the stage is loaded.
    ee_prim_path = find_ee_prim(stage)
    base_path = (
        BASE_PRIM_PATH
        if stage.GetPrimAtPath(BASE_PRIM_PATH).IsValid()
        else ROBOT_PRIM_PATH
    )
    print(f"[task1b] EE prim:   {ee_prim_path}")
    print(f"[task1b] Base prim: {base_path}\n")

    # ---- Iterate FK test configurations ---------------------------------
    # set_dof_positions accepts numpy directly; it expects shape [num_envs, num_dofs].
    def _cmd(q: np.ndarray) -> None:
        articulation.set_dof_positions(q.reshape(1, -1).astype(np.float32))

    # simulation_app.update() doesn't throttle to physics rate in 6.0 — pace
    # each call to ~60 Hz so the motion is visible.
    _step = make_step_pacer(simulation_app)

    raw = articulation.get_dof_positions()
    q_prev = np.asarray(
        raw.numpy() if hasattr(raw, "numpy") else raw,
        dtype=np.float64,
    ).reshape(-1)

    motion_steps = max(1, int(cli.motion_time * 60.0))
    results: list[dict] = []

    for idx, (label, angles_deg) in enumerate(TEST_CONFIGURATIONS, start=1):
        angles_rad = np.array(
            [math.radians(a) for a in angles_deg], dtype=np.float64,
        )
        print(f"[{idx}/5] {label}")
        print(f"        Joints: {[f'{a}deg' for a in angles_deg]}")

        if cli.motion == "smooth":
            for k in range(1, motion_steps + 1):
                _cmd(_smooth_blend(q_prev, angles_rad, k / motion_steps))
                _step()
        else:
            _cmd(angles_rad)
            for _ in range(SETTLE_STEPS):
                _step()

        q_prev = angles_rad

        # Pose readout via USD transform (no Articulation link-pose API yet
        # in 6.0 — direct USD is simpler and works for every prim).
        ee_pos_w,   ee_quat_w   = _xform_world_pose(stage, ee_prim_path)
        base_pos_w, base_quat_w = _xform_world_pose(stage, base_path)
        ee_pos_b, ee_quat_b = _world_to_base(
            ee_pos_w, ee_quat_w, base_pos_w, base_quat_w,
        )
        sim_pos_mm  = tuple(float(v) * 1000.0 for v in ee_pos_b)
        sim_rpy_deg = tuple(math.degrees(a) for a in _quat_wxyz_to_rpy(ee_quat_b))

        T      = forward_kinematics(angles_rad.tolist())
        dh_pos = tuple(T[i, 3] * 1000.0 for i in range(3))
        dh_rpy = tuple(math.degrees(a) for a in rotation_to_euler_rpy(T[:3, :3]))

        results.append({
            "label":       label,
            "angles_deg":  angles_deg,
            "dh_pos_mm":   dh_pos,
            "sim_pos_mm":  sim_pos_mm,
            "dh_rpy_deg":  dh_rpy,
            "sim_rpy_deg": sim_rpy_deg,
        })

        print(f"        DH  (mm) X={dh_pos[0]:8.2f}  "
              f"Y={dh_pos[1]:8.2f}  Z={dh_pos[2]:8.2f}")
        print(f"        Sim (mm) X={sim_pos_mm[0]:8.2f}  "
              f"Y={sim_pos_mm[1]:8.2f}  Z={sim_pos_mm[2]:8.2f}\n")

        for _ in range(max(1, int(cli.dwell * 60))):
            _step()

    _print_table(results)

    print("[task1b] All configurations shown. Close the window to exit.\n")
    while simulation_app.is_running():
        _step()
    simulation_app.close()


if __name__ == "__main__":
    main()
