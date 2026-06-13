# UR5 Robot Kinematics & Dynamics — Academic Project

Course project simulating a 6-DOF UR5 robot performing linear Cartesian motion with smooth velocity profiles using NVIDIA Isaac Sim.

## Project Tasks

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Environment Setup & Forward Kinematics Validation | ✅ Done |
| Task 2 | Inverse Kinematics | 🔲 Upcoming |
| Task 3 | Linear Cartesian Path Planning | 🔲 Upcoming |
| Task 4 | Smooth Velocity Profiling (Trapezoidal / S-Curve) | 🔲 Upcoming |
| Task 5 | Isaac Sim Full Simulation & Deployment | 🔲 Upcoming |

## Requirements

```
numpy
```

Isaac Sim 5.1.0 is required for Task 5 (simulation). Tasks 1–4 run on plain Python 3.10+.

## Task 1 — Forward Kinematics

```bash
# Pure math — runs on plain Python 3
python3 task1_fk_validation.py

# Isaac Sim validation
~/Simulators/isaacsim/python.sh task1b_isaac_fk_validation.py
```

Computes the end-effector pose for 5 joint configurations using the standard DH convention and prints position (X, Y, Z) and orientation (Roll, Pitch, Yaw).

## Robot

- **Model**: Universal Robots UR5
- **DOF**: 6 revolute joints
- **Simulator**: NVIDIA Isaac Sim 5.1.0
- **Control**: `pykos` gRPC (physical deployment)
