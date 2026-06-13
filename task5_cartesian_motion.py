#!/usr/bin/env python3
"""
task5_cartesian_motion.py — UR5 multi-waypoint live motion in Isaac Sim 6.0
===========================================================================

Drives the UR5 through a list of Cartesian waypoints. Each consecutive pair
is connected by a quintic-blended linear segment so the speed starts and
ends at zero per segment (stop-and-go). Joint angles come from the same
damped-pseudo-inverse IK that ``task2_trajectory_planner`` uses, warm-
started across the whole concatenated path.

The matplotlib Qt5 dashboard runs **next to** the viewport and the plots
**build up** as the simulation moves — there's no pre-drawn curve with a
sliding cursor. Each time the sim crosses a sample, the new point gets
appended to every line.

Usage:

    /home/ubuntu/Simulators/isaacsim-6.0/python.sh \\
        /home/ubuntu/ur5-kinematics-project/task5_cartesian_motion.py

    # Random tour of 6 reachable points
    /home/ubuntu/Simulators/isaacsim-6.0/python.sh \\
        /home/ubuntu/ur5-kinematics-project/task5_cartesian_motion.py \\
        --random 6 --seed 42 --motion-time 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI — must be parsed before SimulationApp swallows our flags
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="UR5 multi-waypoint linear Cartesian motion in Isaac Sim 6.0",
    add_help=False,
)
_parser.add_argument("--motion-time", type=float, default=8.0,
                     help="Total trajectory time across all segments (default 8.0)")
_parser.add_argument("--dwell-end",   type=float, default=2.0,
                     help="Seconds to hold the final pose (default 2.0)")
_parser.add_argument("--samples-per-segment", type=int, default=80,
                     help="Trajectory samples per segment (default 80)")
_parser.add_argument("--dashboard", choices=("live", "off"), default="live",
                     help="Live (default) or off")
_parser.add_argument("--random", type=int, default=0,
                     help="If > 0, replace the hard-coded waypoints with N "
                          "random reachable ones (closes the loop). "
                          "Default 0 = use WAYPOINTS constant.")
_parser.add_argument("--seed", type=int, default=None,
                     help="RNG seed for --random (default: nondeterministic)")
cli, _unknown = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# SimulationApp — first Isaac Sim call
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width":  1920,
    "height": 1080,
})

# ---------------------------------------------------------------------------
# Everything else — after SimulationApp()
# ---------------------------------------------------------------------------
import math

import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdGeom  # noqa: F401  (kept for future xform helpers)

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation

sys.path.insert(0, str(Path(__file__).parent))
from ur5_scene import (
    ROBOT_PRIM_PATH,
    TABLE_HEIGHT as _DESK_TOP_Z,
    build_desk,
    build_lights,
    make_step_pacer,
    place_robot_on_desk,
    resolve_ur5_usd,
)
from task2_trajectory_planner import (
    fk_chain,
    fk_position,
    generate_trajectory,
    ik_solve,
)

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Waypoints (Cartesian, in the UR5 base frame).
#
# Five DISTINCT positions: a centre / "home" pose plus four cardinal points
# on a vertical clock-face (YZ plane at X = 0.40, radius 0.20 m) at angles
# 0° (right), 90° (up), 180° (left), 270° (down). A sixth entry closes the
# loop back to centre — so the planner builds five quintic segments and the
# arm sweeps centre → right → up → left → down → centre.
# ---------------------------------------------------------------------------
_CARDINAL_CENTER: np.ndarray = np.array([0.40, 0.00, 0.50])
_CARDINAL_RADIUS: float = 0.20


def _cardinal_waypoints() -> np.ndarray:
    """5 distinct points (centre + 4 cardinals) + loop close."""
    cx, cy, cz = _CARDINAL_CENTER
    r = _CARDINAL_RADIUS
    pts = [_CARDINAL_CENTER]
    for deg in (0.0, 90.0, 180.0, 270.0):
        rad = math.radians(deg)
        # YZ-plane circle: Y = r cos θ, Z = r sin θ, X constant.
        pts.append(np.array([cx, cy + r * math.cos(rad), cz + r * math.sin(rad)]))
    pts.append(_CARDINAL_CENTER)   # close the loop
    return np.asarray(pts)


WAYPOINTS: np.ndarray = _cardinal_waypoints()

# Random sampling box — UR5 has ~0.85 m reach; this stays well inside it.
RANDOM_BOX_MIN = np.array([0.25, -0.35, 0.30])
RANDOM_BOX_MAX = np.array([0.55,  0.35, 0.65])

Q_INIT = np.array([
    0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0,
])

NUM_DOF = 6
DASHBOARD_EVERY = 5   # update plots every Nth sim frame (60 Hz / 5 = 12 Hz)

# ---------------------------------------------------------------------------
# Waypoint generation
# ---------------------------------------------------------------------------
def random_waypoints(n: int, rng: np.random.Generator) -> np.ndarray:
    """N random points uniformly inside the reach box; closes the loop."""
    pts = rng.uniform(RANDOM_BOX_MIN, RANDOM_BOX_MAX, size=(n, 3))
    return np.vstack([pts, pts[0:1]])   # close the loop


def resolve_waypoints() -> np.ndarray:
    if cli.random > 0:
        rng = np.random.default_rng(cli.seed)
        pts = random_waypoints(cli.random, rng)
        print(f"[task5] random waypoints (seed={cli.seed}): "
              f"{pts.shape[0]} points (loop closed)")
        return pts
    return WAYPOINTS


# ---------------------------------------------------------------------------
# Multi-segment trajectory (one quintic blend per consecutive pair)
# ---------------------------------------------------------------------------
def multi_segment_trajectory(
    waypoints: np.ndarray, total_T: float, samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate quintic-blended segments. Returns (t, pos, vel, speed)."""
    n_seg = waypoints.shape[0] - 1
    T_per = total_T / n_seg
    t_list, p_list, v_list, s_list = [], [], [], []
    for i in range(n_seg):
        t_seg, p_seg, v_seg, s_seg = generate_trajectory(
            waypoints[i], waypoints[i + 1], T_per, samples_per_segment,
        )
        t_seg = t_seg + i * T_per
        if i > 0:                    # drop duplicate junction sample
            t_seg, p_seg, v_seg, s_seg = t_seg[1:], p_seg[1:], v_seg[1:], s_seg[1:]
        t_list.append(t_seg)
        p_list.append(p_seg)
        v_list.append(v_seg)
        s_list.append(s_seg)
    return (
        np.concatenate(t_list),
        np.concatenate(p_list),
        np.concatenate(v_list),
        np.concatenate(s_list),
    )


def solve_joint_path(positions: np.ndarray, q_init: np.ndarray) -> np.ndarray:
    """Warm-started IK over the whole multi-waypoint Cartesian path."""
    n = positions.shape[0]
    joints = np.zeros((n, NUM_DOF))
    q = ik_solve(positions[0], q_init, max_iter=600, tol=1e-6)
    joints[0] = q
    for i in range(1, n):
        q = ik_solve(positions[i], q, max_iter=80, tol=5e-5)
        joints[i] = q
    return joints


# ---------------------------------------------------------------------------
# Dashboard — modern look, live-build lines (no pre-drawn curves, no cursors)
# ---------------------------------------------------------------------------
_PALETTE = {
    "x":  "#E63946",
    "y":  "#2A9D8F",
    "z":  "#264653",
    "v":  "#6A4C93",
    "j":  plt.cm.viridis(np.linspace(0.15, 0.85, NUM_DOF)),
    "arm": "#E76F51",
    "trace": "#E76F51",
    "plan":  "#888",
    "wp":    "#264653",
}


def _set_style() -> None:
    plt.rcParams.update({
        "font.family":           "DejaVu Sans",
        "font.size":             11,
        "axes.titlesize":        12.5,
        "axes.titleweight":      "semibold",
        "axes.labelsize":        11,
        "xtick.labelsize":       9.5,
        "ytick.labelsize":       9.5,
        "legend.fontsize":       10,
        "legend.frameon":        False,
        "axes.edgecolor":        "#555",
        "axes.linewidth":        0.9,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.facecolor":        "#FBFBFB",
        "figure.facecolor":      "white",
        "grid.color":            "#D5D5D5",
        "grid.linestyle":        "-",
        "grid.linewidth":        0.6,
        "grid.alpha":            0.9,
        "axes.grid":             True,
        "axes.grid.which":       "major",
        "xtick.direction":       "out",
        "ytick.direction":       "out",
    })


def _add_pad(lo: float, hi: float, frac: float = 0.07) -> tuple[float, float]:
    span = hi - lo
    if span == 0.0:
        span = 1.0
    return lo - span * frac, hi + span * frac


def _build_dashboard(
    t:          np.ndarray,
    positions:  np.ndarray,
    speed:      np.ndarray,
    joints:     np.ndarray,
    waypoints:  np.ndarray,
) -> tuple[plt.Figure, dict]:
    _set_style()

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[3.1, 3.1, 1.1],
        hspace=0.40, wspace=0.22,
        left=0.06, right=0.97, top=0.91, bottom=0.05,
    )

    # ---- Panel 1 — 3D: planned path + live arm + EE trace --------------
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_title("3D Stick Figure  ·  live")
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
              color=_PALETTE["plan"], linewidth=1.0,
              linestyle=(0, (1, 1)), alpha=0.55, label="planned path")
    ax3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                 c=_PALETTE["wp"], s=55, marker="X",
                 edgecolor="white", linewidth=0.8, label="waypoints")
    chain0 = np.array([T[:3, 3] for T in fk_chain(joints[0])])
    live_arm,   = ax3d.plot(
        chain0[:, 0], chain0[:, 1], chain0[:, 2],
        color=_PALETTE["arm"], marker="o", markersize=4.5,
        linewidth=2.4, label="arm now",
    )
    ee_trace,   = ax3d.plot([], [], [], color=_PALETTE["trace"],
                             linewidth=1.5, alpha=0.85, label="EE trace")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="upper left", fontsize=8.5)
    pts = np.vstack([positions, waypoints])
    mins, maxs = pts.min(axis=0) - 0.20, pts.max(axis=0) + 0.20
    span = (maxs - mins).max()
    mids = 0.5 * (maxs + mins)
    ax3d.set_xlim(mids[0] - span / 2, mids[0] + span / 2)
    ax3d.set_ylim(mids[1] - span / 2, mids[1] + span / 2)
    ax3d.set_zlim(mids[2] - span / 2, mids[2] + span / 2)
    ax3d.set_box_aspect((1, 1, 1))

    # ---- Panel 2 — EE position vs time (lines grow as sim runs) --------
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_pos.set_title("End-Effector Position vs Time")
    line_x, = ax_pos.plot([], [], color=_PALETTE["x"], linewidth=2.0, label="X")
    line_y, = ax_pos.plot([], [], color=_PALETTE["y"], linewidth=2.0, label="Y")
    line_z, = ax_pos.plot([], [], color=_PALETTE["z"], linewidth=2.0, label="Z")
    ax_pos.set_xlim(t[0], t[-1])
    ax_pos.set_ylim(*_add_pad(positions.min(), positions.max()))
    ax_pos.set_xlabel("time (s)")
    ax_pos.set_ylabel("position (m)")
    ax_pos.legend(loc="best")

    # ---- Panel 3 — |v| vs time (lines grow) -----------------------------
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_vel.set_title("Linear-Velocity Magnitude  ·  per-segment quintic profile")
    line_v, = ax_vel.plot([], [], color=_PALETTE["v"], linewidth=2.2)
    ax_vel.set_xlim(t[0], t[-1])
    ax_vel.set_ylim(*_add_pad(0.0, speed.max()))
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_ylabel(r"$|v|$ (m/s)")

    # Mark segment boundaries with thin vertical guides — *not* a cursor.
    n_seg = waypoints.shape[0] - 1
    if n_seg > 1:
        T_per = (t[-1] - t[0]) / n_seg
        for k in range(1, n_seg):
            ax_vel.axvline(t[0] + k * T_per, color="#bbb",
                           linewidth=0.7, linestyle=":")

    # ---- Panel 4 — joint angles vs time (lines grow) -------------------
    ax_j = fig.add_subplot(gs[1, 1])
    ax_j.set_title("Joint Angles vs Time")
    joint_lines = []
    for j in range(NUM_DOF):
        ln, = ax_j.plot([], [], color=_PALETTE["j"][j], linewidth=1.7,
                        label=fr"$\theta_{{{j + 1}}}$")
        joint_lines.append(ln)
    ax_j.set_xlim(t[0], t[-1])
    j_deg = np.degrees(joints)
    ax_j.set_ylim(*_add_pad(j_deg.min(), j_deg.max()))
    ax_j.set_xlabel("time (s)")
    ax_j.set_ylabel("angle (deg)")
    ax_j.legend(loc="upper right", ncol=3)

    # ---- Panel 5 — equations + parameters (static) ---------------------
    ax_eq = fig.add_subplot(gs[2, :])
    ax_eq.axis("off")
    speed_peak = float(speed.max())
    eqns = "\n".join([
        r"$\mathbf{Multi\!-\!Segment\ Cartesian\ Path:}\quad "
        r"\mathbf{P}(t) = \mathbf{P}_k + (\mathbf{P}_{k+1} - \mathbf{P}_k)\,s_k(t)"
        r"\qquad t\in[kT/n,\,(k+1)T/n]$",
        r"$\mathbf{Quintic\ Blend\ per\ Segment:}\quad "
        r"s(\tau)=10\tau^3-15\tau^4+6\tau^5\qquad"
        r"\dot{s}(\tau)=\frac{1}{T_{seg}}(30\tau^2-60\tau^3+30\tau^4)$",
        r"$\mathbf{Damped\ IK\ Update:}\quad "
        r"\mathbf{q}_{k+1} = \mathbf{q}_k + \alpha\,J^{T}"
        r"(J\,J^{T} + \lambda^{2}\,I)^{-1}"
        r"[\mathbf{p}_{\mathrm{target}} - \mathrm{FK}(\mathbf{q}_k)]$",
        (rf"$N_\mathrm{{wp}}={waypoints.shape[0]}$"
         rf"$\qquad N_\mathrm{{seg}}={n_seg}$"
         rf"$\qquad T={cli.motion_time:.1f}\,\mathrm{{s}}$"
         rf"$\qquad |v|_\mathrm{{peak}}={speed_peak:.3f}\,\mathrm{{m/s}}$"),
    ])
    ax_eq.text(0.5, 0.5, eqns, ha="center", va="center",
               fontsize=12, linespacing=1.75)

    fig.suptitle(
        "UR5 Task 5  ·  Live Multi-Waypoint Cartesian Motion",
        fontsize=15.5, fontweight="semibold", y=0.965,
    )
    artists = {
        "live_arm":    live_arm,
        "ee_trace":    ee_trace,
        "line_x":      line_x,
        "line_y":      line_y,
        "line_z":      line_z,
        "line_v":      line_v,
        "joint_lines": joint_lines,
        # rolling data buffers
        "buf_t":  [],
        "buf_x":  [],
        "buf_y":  [],
        "buf_z":  [],
        "buf_v":  [],
        "buf_j":  [[] for _ in range(NUM_DOF)],
        "buf_ee": ([], [], []),  # (x, y, z) trace
    }
    return fig, artists


def _push_sample(
    artists: dict,
    t_i:     float,
    pos_i:   np.ndarray,
    speed_i: float,
    joints_i: np.ndarray,
) -> None:
    """Append one sample to every line's data buffer."""
    artists["buf_t"].append(t_i)
    artists["buf_x"].append(float(pos_i[0]))
    artists["buf_y"].append(float(pos_i[1]))
    artists["buf_z"].append(float(pos_i[2]))
    artists["buf_v"].append(float(speed_i))
    for j in range(NUM_DOF):
        artists["buf_j"][j].append(math.degrees(float(joints_i[j])))

    chain = np.array([T[:3, 3] for T in fk_chain(joints_i)])
    ee = chain[-1]
    artists["buf_ee"][0].append(float(ee[0]))
    artists["buf_ee"][1].append(float(ee[1]))
    artists["buf_ee"][2].append(float(ee[2]))

    # Update the live arm pose every push.
    artists["live_arm"].set_data_3d(chain[:, 0], chain[:, 1], chain[:, 2])


def _flush_dashboard(artists: dict) -> None:
    """Push every buffer into its corresponding line and redraw."""
    t = artists["buf_t"]
    artists["line_x"].set_data(t, artists["buf_x"])
    artists["line_y"].set_data(t, artists["buf_y"])
    artists["line_z"].set_data(t, artists["buf_z"])
    artists["line_v"].set_data(t, artists["buf_v"])
    for j in range(NUM_DOF):
        artists["joint_lines"][j].set_data(t, artists["buf_j"][j])
    artists["ee_trace"].set_data_3d(*artists["buf_ee"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _log_cardinal_angles(waypoints: np.ndarray) -> None:
    """Print each waypoint's angle around the cardinal centre in the YZ plane.

    Confirms the waypoints really do sit at 0 deg / 90 deg / 180 deg / 270 deg
    relative to ``_CARDINAL_CENTER``. The numbers shown are *Cartesian-space*
    bearings (atan2 of the Y/Z offset) — they are NOT robot joint angles,
    which are six independent values per pose solved by IK.
    """
    cy, cz = float(_CARDINAL_CENTER[1]), float(_CARDINAL_CENTER[2])
    print("[task5] waypoint cardinal-angle verification "
          "(YZ-plane, measured from centre):")
    for i, p in enumerate(waypoints):
        dy, dz = float(p[1]) - cy, float(p[2]) - cz
        if dy == 0.0 and dz == 0.0:
            label = "centre"
        else:
            ang = math.degrees(math.atan2(dz, dy)) % 360.0
            label = f"{ang:6.1f}°"
        print(f"   #{i}  {label}  at ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) m")


def main() -> None:
    waypoints = resolve_waypoints()
    _log_cardinal_angles(waypoints)
    print(f"[task5] planning quintic path through {waypoints.shape[0]} "
          f"waypoints over {cli.motion_time:.1f} s "
          f"({cli.samples_per_segment} samples/segment)")

    t, positions, _vel, speed = multi_segment_trajectory(
        waypoints, cli.motion_time, cli.samples_per_segment,
    )
    joints = solve_joint_path(positions, Q_INIT)

    residual = np.array([
        float(np.linalg.norm(fk_position(joints[i]) - positions[i]))
        for i in range(len(t))
    ])
    print(f"[task5] IK max residual : {residual.max() * 1000:.3f} mm")
    print(f"[task5] peak EE speed   : {speed.max():.3f} m/s")
    print(f"[task5] total samples   : {len(t)}")

    # ---- Isaac Sim scene -----------------------------------------------
    stage = omni.usd.get_context().get_stage()
    GroundPlane(paths="/World/Ground")
    build_lights()
    build_desk(stage)

    usd_path = resolve_ur5_usd()
    print(f"[task5] UR5 asset       : {usd_path}")
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PRIM_PATH)

    # Place the UR5 so the base sits flush on the desk top (no clipping).
    place_robot_on_desk(stage, ROBOT_PRIM_PATH, _DESK_TOP_Z)

    omni.timeline.get_timeline_interface().play()
    for _ in range(3):
        simulation_app.update()

    articulation = Articulation(paths=ROBOT_PRIM_PATH)
    print(f"[task5] UR5 articulation: {articulation.num_joints} joints")

    step = make_step_pacer(simulation_app)

    # ---- dashboard ------------------------------------------------------
    fig = artists = None
    if cli.dashboard == "live":
        fig, artists = _build_dashboard(t, positions, speed, joints, waypoints)
        plt.show(block=False)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # ---- streaming loop -------------------------------------------------
    print(f"[task5] streaming {len(t)} joint commands @ 60 Hz "
          f"(dashboard refresh every {DASHBOARD_EVERY} frames)")
    for i in range(len(t)):
        articulation.set_dof_positions(
            joints[i].reshape(1, -1).astype(np.float32),
        )
        if cli.dashboard == "live" and artists is not None:
            _push_sample(artists, float(t[i]), positions[i],
                         float(speed[i]), joints[i])
            if i % DASHBOARD_EVERY == 0:
                _flush_dashboard(artists)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
        step()

    # Final flush so the very last sample lands on every line.
    if cli.dashboard == "live" and artists is not None:
        _flush_dashboard(artists)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    for _ in range(max(0, int(cli.dwell_end * 60))):
        step()

    print("[task5] done — close the Isaac Sim window to exit.")
    while simulation_app.is_running():
        if fig is not None:
            fig.canvas.flush_events()
        step()
    simulation_app.close()


if __name__ == "__main__":
    main()
