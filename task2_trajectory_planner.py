#!/usr/bin/env python3
"""
task2_trajectory_planner.py — UR5 Cartesian trajectory + visual dashboard
=========================================================================

Generates a linear Cartesian end-effector trajectory between two task-space
points using a **quintic polynomial blending function** (smooth start, zero
endpoint velocity and acceleration). Joint angles are obtained at each step
by **numerical inverse kinematics** with a damped Jacobian pseudo-inverse,
warm-started from the previous step.

The matplotlib dashboard shows, simultaneously, on a single figure:

  Panel 1 — 3D stick figure of the UR5 at start / middle / end poses
  Panel 2 — EE Cartesian position (X, Y, Z) vs time
  Panel 3 — EE linear-velocity magnitude vs time (quintic bell curve)
  Panel 4 — Joint angles theta_1 .. theta_6 vs time
  Panel 5 — Rendered LaTeX equations for the trajectory and IK update

Run:

    python3 task2_trajectory_planner.py

No Isaac Sim required — only numpy and matplotlib.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

# ---------------------------------------------------------------------------
# UR5 standard DH parameters (Craig convention used throughout this project)
# ---------------------------------------------------------------------------
UR5_A:     np.ndarray = np.array([0.0,  -0.425, -0.39225, 0.0,      0.0,     0.0])
UR5_D:     np.ndarray = np.array([0.089159, 0.0, 0.0,     0.10915,  0.09465, 0.0823])
UR5_ALPHA: np.ndarray = np.array([math.pi / 2, 0.0, 0.0,  math.pi / 2,
                                  -math.pi / 2, 0.0])
NUM_DOF = 6


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def _dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk_chain(q: np.ndarray) -> list[np.ndarray]:
    """Return the list of T_0_i for i = 0..6 (7 transforms, base + 6 links)."""
    chain = [np.eye(4)]
    T = np.eye(4)
    for i in range(NUM_DOF):
        T = T @ _dh_transform(q[i], UR5_D[i], UR5_A[i], UR5_ALPHA[i])
        chain.append(T)
    return chain


def fk_position(q: np.ndarray) -> np.ndarray:
    """EE position in base frame, shape (3,)."""
    return fk_chain(q)[-1][:3, 3]


def jacobian_position(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical 3 x 6 EE-position Jacobian via central differences."""
    J = np.zeros((3, NUM_DOF))
    for j in range(NUM_DOF):
        q_plus = q.copy(); q_plus[j]  += eps
        q_minus = q.copy(); q_minus[j] -= eps
        J[:, j] = (fk_position(q_plus) - fk_position(q_minus)) / (2.0 * eps)
    return J


def ik_solve(
    target: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iter: int = 300,
    tol:      float = 5e-5,
    damping:  float = 1e-3,
    step_gain: float = 0.6,
) -> np.ndarray:
    """Damped-least-squares IK: q_{k+1} = q_k + alpha * J^T (J J^T + lambda^2 I)^{-1} e."""
    q = q_init.copy()
    I3 = np.eye(3)
    for _ in range(max_iter):
        err = target - fk_position(q)
        if np.linalg.norm(err) < tol:
            break
        J = jacobian_position(q)
        dq = step_gain * J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * I3, err)
        q = q + dq
    return q


# ---------------------------------------------------------------------------
# Quintic blending function
# ---------------------------------------------------------------------------
def quintic_blend(t: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised quintic polynomial s(t) and s_dot(t) over [0, T].

    s(0) = 0, s(T) = 1, s'(0) = s'(T) = 0, s''(0) = s''(T) = 0.
    """
    tau = np.clip(t / T, 0.0, 1.0)
    s     = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5
    s_dot = (30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4) / T
    return s, s_dot


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------
def generate_trajectory(
    p_start: np.ndarray,
    p_end:   np.ndarray,
    duration: float,
    samples:  int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, position, velocity, vel_magnitude) for a linear path with
    a quintic-blended speed profile."""
    t = np.linspace(0.0, duration, samples)
    s, s_dot = quintic_blend(t, duration)
    delta = p_end - p_start
    positions  = p_start + np.outer(s,     delta)
    velocities = np.outer(s_dot, delta)
    speed = np.linalg.norm(velocities, axis=1)
    return t, positions, velocities, speed


def solve_joint_trajectory(
    positions: np.ndarray, q_init: np.ndarray,
) -> np.ndarray:
    """Joint trajectory by warm-started IK along the EE path."""
    n = positions.shape[0]
    joints = np.zeros((n, NUM_DOF))
    # Solve the first point thoroughly so the warm-start is good.
    q = ik_solve(positions[0], q_init, max_iter=600, tol=1e-6)
    joints[0] = q
    for i in range(1, n):
        q = ik_solve(positions[i], q, max_iter=80, tol=5e-5)
        joints[i] = q
    return joints


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def _draw_stick_figure(ax, q: np.ndarray, color: str, label: str, alpha: float) -> None:
    chain = fk_chain(q)
    xs = [T[0, 3] for T in chain]
    ys = [T[1, 3] for T in chain]
    zs = [T[2, 3] for T in chain]
    ax.plot(xs, ys, zs, color=color, marker='o', markersize=5,
            label=label, alpha=alpha, linewidth=2.5)
    # Mark the EE distinctly.
    ax.scatter(xs[-1], ys[-1], zs[-1], color=color, s=80,
               edgecolor='black', linewidth=0.8, alpha=alpha)


def _equal_axes_3d(ax, points: np.ndarray) -> None:
    """Make 3-D axes share equal scale so the arm doesn't look squashed."""
    mins = points.min(axis=0) - 0.05
    maxs = points.max(axis=0) + 0.05
    span = (maxs - mins).max()
    mids = 0.5 * (maxs + mins)
    ax.set_xlim(mids[0] - span / 2, mids[0] + span / 2)
    ax.set_ylim(mids[1] - span / 2, mids[1] + span / 2)
    ax.set_zlim(mids[2] - span / 2, mids[2] + span / 2)


def build_dashboard(
    t:          np.ndarray,
    positions:  np.ndarray,
    velocities: np.ndarray,
    speed:      np.ndarray,
    joints:     np.ndarray,
    p_start:    np.ndarray,
    p_end:      np.ndarray,
    duration:   float,
) -> plt.Figure:
    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(
        3, 2, figure=fig,
        height_ratios=[3.0, 3.0, 1.1],
        hspace=0.35, wspace=0.22,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )

    # ---- Panel 1: 3D stick figure -----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_title("3D Stick Figure — Start / Middle / End")
    n = positions.shape[0]
    _draw_stick_figure(ax1, joints[0],     "#1f77b4", "Start",  0.55)
    _draw_stick_figure(ax1, joints[n // 2], "#ff7f0e", "Middle", 0.80)
    _draw_stick_figure(ax1, joints[-1],    "#2ca02c", "End",    1.00)
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
             linestyle=':', color='dimgray', alpha=0.7, label='EE path')
    ax1.scatter(*p_start, c='#1f77b4', s=60, marker='X')
    ax1.scatter(*p_end,   c='#2ca02c', s=60, marker='X')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc='upper left', fontsize=8)

    # Build a point cloud of every joint origin in every keyframe + the path
    pts = np.vstack([
        np.array([T[:3, 3] for T in fk_chain(joints[k])])
        for k in (0, n // 2, n - 1)
    ] + [positions, np.array([p_start, p_end])])
    _equal_axes_3d(ax1, pts)

    # ---- Panel 2: position vs time ----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("EE Cartesian Position vs Time")
    ax2.plot(t, positions[:, 0], label="X", color='#d62728', linewidth=2)
    ax2.plot(t, positions[:, 1], label="Y", color='#2ca02c', linewidth=2)
    ax2.plot(t, positions[:, 2], label="Z", color='#1f77b4', linewidth=2)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    # ---- Panel 3: |v| vs time ---------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("EE Linear-Velocity Magnitude  (quintic bell curve)")
    ax3.plot(t, speed, color='#9467bd', linewidth=2.4)
    ax3.fill_between(t, 0.0, speed, color='#9467bd', alpha=0.18)
    peak = float(speed.max())
    t_peak = float(t[int(np.argmax(speed))])
    ax3.axhline(peak, ls='--', color='#9467bd', alpha=0.6,
                label=f"peak |v| = {peak:.3f} m/s  @  t = {t_peak:.2f} s")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("|v| (m/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # ---- Panel 4: joint angles vs time ------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Joint Angles vs Time")
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, NUM_DOF))
    for j in range(NUM_DOF):
        ax4.plot(t, np.degrees(joints[:, j]),
                 label=fr"$\theta_{{{j + 1}}}$",
                 color=palette[j], linewidth=1.8)
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel(r"angle (deg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', ncol=3)

    # ---- Panel 5: equations ----------------------------------------------
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    distance = float(np.linalg.norm(p_end - p_start))
    params_line = (
        rf"$\mathbf{{P_0}}=({p_start[0]:.2f},\,{p_start[1]:.2f},\,{p_start[2]:.2f})\,\mathrm{{m}}$"
        rf"$\quad\mathbf{{P_f}}=({p_end[0]:.2f},\,{p_end[1]:.2f},\,{p_end[2]:.2f})\,\mathrm{{m}}$"
        rf"$\quad T={duration:.1f}\,\mathrm{{s}}$"
        rf"$\quad\|\Delta\mathbf{{P}}\|={distance:.3f}\,\mathrm{{m}}$"
        rf"$\quad v_{{\max}}=\tfrac{{15}}{{8}}\,\|\Delta\mathbf{{P}}\|/T={1.875 * distance / duration:.3f}\,\mathrm{{m/s}}$"
    )
    equations = "\n".join([
        r"$\mathbf{Cartesian\ Path}:\quad "
        r"\mathbf{P}(t) = \mathbf{P_0} + (\mathbf{P_f} - \mathbf{P_0})\,s(t)"
        r"\qquad\dot{\mathbf{P}}(t) = (\mathbf{P_f} - \mathbf{P_0})\,\dot{s}(t)$",
        r"$\mathbf{Quintic\ Blend}:\quad "
        r"s(\tau)=10\tau^3-15\tau^4+6\tau^5"
        r"\qquad \dot{s}(\tau)=\tfrac{1}{T}\,(30\tau^2-60\tau^3+30\tau^4)"
        r"\qquad \tau=t/T$",
        r"$\mathbf{Damped\ IK\ Update}:\quad "
        r"\mathbf{q}_{k+1} = \mathbf{q}_k + \alpha\,J^{\mathsf{T}}\,"
        r"\bigl(J\,J^{\mathsf{T}} + \lambda^{2} I\bigr)^{-1}"
        r"\bigl[\mathbf{p}_{\mathrm{target}} - \mathrm{FK}(\mathbf{q}_k)\bigr]$",
        params_line,
    ])
    ax5.text(0.5, 0.5, equations, ha='center', va='center',
             fontsize=12.5, linespacing=1.7)

    fig.suptitle(
        "UR5 Task 2 — Cartesian Trajectory Planner & Visual Dashboard",
        fontsize=16, fontweight='bold',
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p_start = np.array([0.40, -0.30, 0.50])
    p_end   = np.array([0.40,  0.30, 0.50])
    duration = 5.0
    samples  = 200

    q_init = np.array([0.0, -math.pi / 2, math.pi / 2,
                       -math.pi / 2, -math.pi / 2, 0.0])

    print("[task2] generating quintic-blended Cartesian path …")
    t, positions, velocities, speed = generate_trajectory(
        p_start, p_end, duration, samples,
    )

    print("[task2] solving inverse kinematics along the path …")
    joints = solve_joint_trajectory(positions, q_init)

    residual = np.linalg.norm(
        np.array([fk_position(joints[i]) - positions[i] for i in range(samples)]),
        axis=1,
    )
    print(f"[task2] mean IK position residual : {residual.mean() * 1e3:.3f} mm")
    print(f"[task2] max  IK position residual : {residual.max()  * 1e3:.3f} mm")
    print(f"[task2] peak EE speed             : {speed.max():.3f} m/s")

    fig = build_dashboard(
        t, positions, velocities, speed, joints,
        p_start, p_end, duration,
    )
    plt.show()


if __name__ == "__main__":
    main()
