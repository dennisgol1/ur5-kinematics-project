# UR5 Robot Kinematics & Dynamics — Academic Project

Course project on a 6-DOF Universal Robots UR5: forward & inverse
kinematics, quintic-blended Cartesian trajectory planning,
multi-waypoint live motion in **NVIDIA Isaac Sim 6.0**, and an
analytical-Jacobian / static-torque dashboard for verification.

Every task ships with a self-contained Python script and a matplotlib
dashboard (some live, some static) so the math is observable next to
the simulator viewport.

---

## Status

| Task | Description | Implementation | Status |
|------|-------------|----------------|--------|
| 1   | Forward Kinematics (DH convention, 5 sample poses) | `task1_fk_validation.py` | ✅ Done |
| 1b  | Isaac Sim FK validation against `task1` math | `task1b_isaac_fk_validation.py` | ✅ Done |
| 2   | Linear Cartesian path + quintic blend + damped IK | `task2_trajectory_planner.py` | ✅ Done |
| 3   | (Linear Cartesian Path Planning) | absorbed into Task 2 / Task 5 | ✅ Done |
| 4   | (Smooth Velocity Profiling) | absorbed into Task 2 / Task 5 | ✅ Done |
| 5   | Multi-waypoint live motion + 6-panel dashboard (Jacobian check + static torque) | `task5_cartesian_motion.py` | ✅ Done |

---

## Requirements

```text
numpy
matplotlib
PyQt5            # only for the live dashboard in task5
```

`task1` and `task2` run on any system Python ≥ 3.10. `task1b` and
`task5` need **NVIDIA Isaac Sim 6.0** and must be launched with its
bundled Python:

```bash
/home/ubuntu/Simulators/isaacsim-6.0/python.sh <script>
```

---

## Robot

- **Model**: Universal Robots UR5 (6 revolute joints, ~0.85 m reach)
- **Simulator**: NVIDIA Isaac Sim 6.0
- **Convention**: standard DH (Craig)

DH parameters used throughout:

| Joint | a (m) | d (m) | α (rad) |
|-------|-------|-------|---------|
| 1 | 0       | 0.089159 |  π/2  |
| 2 | -0.425  | 0        |  0    |
| 3 | -0.39225| 0        |  0    |
| 4 | 0       | 0.10915  |  π/2  |
| 5 | 0       | 0.09465  | -π/2  |
| 6 | 0       | 0.0823   |  0    |

---

## Task 1 — Forward Kinematics

Computes the end-effector pose `(X, Y, Z, Roll, Pitch, Yaw)` for five
representative joint configurations using the DH chain
`T_0^6 = ∏ A_i(θ_i, d_i, a_i, α_i)`.

```bash
python3 task1_fk_validation.py
```

![Task 1 — five FK poses printed to the terminal](images/task1_fk_5poses.png)
_Placeholder — replace with a screenshot of the printed pose table._

### Task 1b — Isaac Sim FK validation

Spawns the UR5 in Isaac Sim, drives the joints to each of the five
configurations, and compares Isaac's reported EE pose against the
math from Task 1 (must agree to sub-millimetre).

```bash
/home/ubuntu/Simulators/isaacsim-6.0/python.sh task1b_isaac_fk_validation.py
```

![Task 1b — UR5 in Isaac Sim with reported residuals](images/task1b_isaac_viewport.png)
_Placeholder — replace with a viewport screenshot showing the UR5 in pose #3._

---

## Task 2 — Cartesian Trajectory Planner

Linear EE motion between two task-space points with a **quintic-blended
speed profile** so velocity and acceleration are zero at both endpoints.
Joints come from a **damped-pseudo-inverse IK** warm-started across the
path.

```bash
python3 task2_trajectory_planner.py
```

Dashboard (5 panels): 3-D stick figure at start/middle/end, EE position
vs time, |v| (the characteristic quintic bell curve), joint angles,
and a rendered-equations footer.

![Task 2 — single-segment quintic trajectory dashboard](images/task2_trajectory_dashboard.png)
_Placeholder — replace with a screenshot of the matplotlib window after running task2._

---

## Task 5 — Live Multi-Waypoint Cartesian Motion (+ Jacobian + Static Torque)

The "main event": six Cartesian waypoints (centre → 0° → 90° → 180° →
270° → centre) on a vertical Y-Z-plane circle at X = 0.40 m, connected
by quintic segments and streamed to the UR5 in **Isaac Sim 6.0** at
60 Hz. A live matplotlib dashboard builds up next to the viewport — no
pre-drawn ghost curves, no scrubbing cursors; every panel grows in
sync with the simulation.

```bash
# Default 12 s tour through the cardinal waypoints
/home/ubuntu/Simulators/isaacsim-6.0/python.sh task5_cartesian_motion.py

# Random 6-point tour
/home/ubuntu/Simulators/isaacsim-6.0/python.sh task5_cartesian_motion.py \
    --random 6 --seed 42 --motion-time 12
```

Dashboard panels (4 × 2 grid + full-width torque row + equations footer):

1. **3-D stick figure** — live arm + EE trace + planned path + waypoint markers
2. **EE position vs time** — X / Y / Z lines grow with the sim
3. **|v| comparison** — planner's quintic profile **overlaid** with `|J(q) · q̇|`
   from the analytical Jacobian (they must coincide)
4. **Joint angles vs time** — θ₁ … θ₆ in degrees
5. **Static joint torques** — τ₁ … τ₆ from `τ = J^T · F` with a 5 kg
   payload at the EE under gravity
6. **Equations footer** — Cartesian path, quintic blend, damped IK,
   geometric Jacobian, V = J·q̇ check, τ = J^T·F

![Task 5 — UR5 sweeping the cardinal waypoints in Isaac Sim](images/task5_isaac_viewport.png)
_Placeholder — replace with a viewport screenshot mid-sweep._

![Task 5 — 6-panel live dashboard](images/task5_dashboard.png)
_Placeholder — replace with a screenshot of the matplotlib dashboard at the end of a run._

---

## Math Reference

### Forward kinematics (DH chain)

$$T_0^6(\mathbf{q}) = \prod_{i=1}^{6} A_i(\theta_i, d_i, a_i, \alpha_i)$$

### Quintic blending function

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5,\qquad \tau = t/T$$

`s(0)=0`, `s(T)=1`, `s'(0)=s'(T)=0`, `s''(0)=s''(T)=0` — smooth start
and stop with no acceleration discontinuities.

### Damped Jacobian IK update

$$\mathbf{q}_{k+1} = \mathbf{q}_k + \alpha\, J^{T}\,(J J^{T} + \lambda^{2} I)^{-1}\,[\mathbf{p}_{\text{target}} - \text{FK}(\mathbf{q}_k)]$$

Damping `λ` keeps the update well-conditioned near singularities.

### Geometric Jacobian (analytical, cross-product form)

For an all-revolute manipulator like the UR5, each column of the
6 × 6 geometric Jacobian is

$$J_v^{(i)} = \hat{z}_{i-1} \times (\mathbf{p}_{ee} - \mathbf{p}_{i-1}),\qquad J_\omega^{(i)} = \hat{z}_{i-1}$$

where `ẑ_{i-1}` and `p_{i-1}` come from the FK chain. Task 5 verifies
this implementation by checking that `|J(q) · q̇|` (with `q̇` taken from
the numerical derivative of the planned joint path) matches the
planner's own `|v|` profile to sub-µm/s.

### Static joint torques

For an external wrench `F = [Fx, Fy, Fz, Mx, My, Mz]^T` applied at the
end-effector, the joint torques that hold the manipulator static are

$$\boldsymbol{\tau} = J^{T} \mathbf{F}$$

Task 5 evaluates this along the trajectory with a 5 kg gravity load:
`F = [0, 0, -M·g, 0, 0, 0]^T`.

---

## Repository layout

```
ur5-kinematics-project/
├── README.md
├── task1_fk_validation.py            # pure-math FK (Task 1)
├── task1b_isaac_fk_validation.py     # Isaac Sim FK cross-check (Task 1b)
├── task2_trajectory_planner.py       # quintic Cartesian + damped IK + geometric Jacobian
├── task5_cartesian_motion.py         # multi-waypoint live motion + 6-panel dashboard
├── ur5_scene.py                      # shared Isaac Sim scene helpers (desk, lights, camera)
└── images/                           # screenshots for this README (placeholders for now)
```

---

## Credits

UR5 Robot Kinematics & Dynamics course project. Author:
Dennis Golubitsky (`dennisgol101@gmail.com`).
