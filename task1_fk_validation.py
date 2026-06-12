"""
Task 1: UR5 Forward Kinematics Validation
==========================================
Implements the standard DH (Denavit-Hartenberg) convention for the UR5 6-DOF
robot arm and validates end-effector poses for five joint configurations.

Standard DH convention (Craig / textbook form):
    T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)

UR5 kinematic parameters (Universal Robots product manual, Rev. E3):
    All lengths in metres.

    Joint | d_i (m)   | a_i (m)    | α_i (rad)
    ------|-----------|------------|----------
      1   | 0.089159  |  0.0       | +π/2
      2   | 0.0       | -0.425     |  0.0
      3   | 0.0       | -0.39225   |  0.0
      4   | 0.10915   |  0.0       | +π/2
      5   | 0.09465   |  0.0       | -π/2
      6   | 0.0823    |  0.0       |  0.0
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# UR5 Standard DH Parameters
# ---------------------------------------------------------------------------

# Each row: (d_i, a_i, alpha_i).  θ_i is the variable supplied at runtime.
UR5_DH_PARAMS: list[tuple[float, float, float]] = [
    (0.089159,  0.0,      math.pi / 2),   # Joint 1
    (0.0,      -0.425,    0.0),            # Joint 2
    (0.0,      -0.39225,  0.0),            # Joint 3
    (0.10915,   0.0,      math.pi / 2),   # Joint 4
    (0.09465,   0.0,     -math.pi / 2),   # Joint 5
    (0.0823,    0.0,      0.0),            # Joint 6
]


# ---------------------------------------------------------------------------
# Core kinematics functions
# ---------------------------------------------------------------------------

def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Return the 4×4 homogeneous DH transform for a single joint.

    Standard DH formula:
        T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)

    return np.array([
        [ct,  -st * ca,   st * sa,  a * ct],
        [st,   ct * ca,  -ct * sa,  a * st],
        [0.0,       sa,        ca,       d],
        [0.0,      0.0,       0.0,     1.0],
    ], dtype=float)


def forward_kinematics(joint_angles: list[float] | np.ndarray) -> np.ndarray:
    """Compute T_6^0 — the end-effector pose in the base frame.

    Args:
        joint_angles: Six joint angles in radians [θ₁ … θ₆].

    Returns:
        A 4×4 homogeneous transformation matrix representing the
        end-effector position and orientation in the base frame.
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}.")

    T = np.eye(4, dtype=float)
    for theta, (d, a, alpha) in zip(joint_angles, UR5_DH_PARAMS):
        T = T @ dh_transform(theta, d, a, alpha)
    return T


def rotation_to_euler_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Extract Roll-Pitch-Yaw (ZYX Euler / extrinsic XYZ) from a 3×3 rotation matrix.

    Convention used here matches ROS / Isaac Sim:
        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Returns:
        (roll, pitch, yaw) in radians.
    """
    # Clamp to [-1, 1] to guard against floating-point noise in arcsin
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))

    # Gimbal-lock guard: when cos(pitch) ≈ 0
    if abs(math.cos(pitch)) > 1e-10:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:
        # At gimbal lock set roll = 0 and absorb everything into yaw
        roll = 0.0
        yaw  = math.atan2(-R[0, 1], R[1, 1])

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

# Each entry: (label, [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]) — angles in degrees.
# The script converts to radians before calling forward_kinematics().
TEST_CONFIGURATIONS: list[tuple[str, list[float]]] = [
    (
        "Config 1 — Zero / Home position",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        "Config 2 — Elbow-up, arm pointing forward",
        [0.0, -90.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        "Config 3 — Base rotated 90°, elbow bent",
        [90.0, -90.0, 90.0, 0.0, 0.0, 0.0],
    ),
    (
        "Config 4 — Wrist involved",
        [0.0, -90.0, 90.0, -90.0, 0.0, 0.0],
    ),
    (
        "Config 5 — Mixed 45° angles",
        [45.0, -45.0, 45.0, -45.0, 45.0, 0.0],
    ),
]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def print_fk_result(label: str, angles_deg: list[float]) -> None:
    """Run FK and print EE position and orientation for one configuration."""
    angles_rad = [math.radians(a) for a in angles_deg]
    T = forward_kinematics(angles_rad)

    # Extract position (metres) and convert to mm for readability
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    # Extract RPY from the rotation sub-matrix
    R = T[:3, :3]
    roll, pitch, yaw = rotation_to_euler_rpy(R)

    print_separator()
    print(f"  {label}")
    print_separator()
    angles_str = "  ".join(f"θ{i+1}={a:>7.1f}°" for i, a in enumerate(angles_deg))
    print(f"  Joints : {angles_str}")
    print()
    print(f"  End-Effector Position (base frame)")
    print(f"    X = {x * 1000:>10.4f} mm   ({x:.6f} m)")
    print(f"    Y = {y * 1000:>10.4f} mm   ({y:.6f} m)")
    print(f"    Z = {z * 1000:>10.4f} mm   ({z:.6f} m)")
    print()
    print(f"  End-Effector Orientation — Roll-Pitch-Yaw (ZYX Euler)")
    print(f"    Roll  = {math.degrees(roll):>10.4f}°   ({roll:.6f} rad)")
    print(f"    Pitch = {math.degrees(pitch):>10.4f}°   ({pitch:.6f} rad)")
    print(f"    Yaw   = {math.degrees(yaw):>10.4f}°   ({yaw:.6f} rad)")
    print()
    print(f"  Full T_6^0 matrix:")
    for row in T:
        formatted = "  ".join(f"{v:>10.6f}" for v in row)
        print(f"    [ {formatted} ]")
    print()


# ---------------------------------------------------------------------------
# Isaac Sim integration guide
# ---------------------------------------------------------------------------

ISAAC_SIM_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║         HOW TO VALIDATE FK IN ISAAC SIM — STEP-BY-STEP GUIDE        ║
╚══════════════════════════════════════════════════════════════════════╝

PREREQUISITES
  • NVIDIA Isaac Sim 5.1.0 installed and licensed
  • Python path includes the Isaac Sim site-packages (see isaacsim.sh)

────────────────────────────────────────────────────────────────────────
STEP 1 — Load the UR5 asset
────────────────────────────────────────────────────────────────────────
In an Isaac Sim Python extension script (or the Script Editor tab):

    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    import numpy as np

    world = World(stage_units_in_meters=1.0)

    robot = world.scene.add(
        Robot(
            prim_path="/World/UR5",
            name="ur5",
            usd_path=(
                "omniverse://localhost/Isaac/Robots/"
                "UniversalRobots/ur5/ur5.usd"
            ),
            position=np.array([0.0, 0.0, 0.0]),
        )
    )
    world.reset()

────────────────────────────────────────────────────────────────────────
STEP 2 — Set joint positions
────────────────────────────────────────────────────────────────────────
Isaac Sim joint order for UR5: [shoulder_pan, shoulder_lift, elbow,
                                 wrist_1, wrist_2, wrist_3]
This matches θ₁…θ₆ in this script.

    import math
    angles_deg = [0.0, -90.0, 0.0, 0.0, 0.0, 0.0]   # Config 2 example
    angles_rad = [math.radians(a) for a in angles_deg]

    robot.set_joint_positions(np.array(angles_rad))
    world.step(render=True)   # advance one physics step

────────────────────────────────────────────────────────────────────────
STEP 3 — Read the End-Effector pose
────────────────────────────────────────────────────────────────────────
The UR5 USD model exposes an "ee_link" (or "tool0") prim:

    from omni.isaac.core.prims import RigidPrimView

    ee = RigidPrimView(
        prim_paths_expr="/World/UR5/ee_link",
        name="ee_view",
    )
    world.scene.add(ee)
    world.reset()

    # After stepping physics:
    pos, quat = ee.get_world_poses()   # pos: (N,3), quat: (N,4) [w,x,y,z]
    print("EE position (m):", pos[0])

────────────────────────────────────────────────────────────────────────
STEP 4 — Convert quaternion → Roll-Pitch-Yaw
────────────────────────────────────────────────────────────────────────
Isaac Sim returns quaternions as [w, x, y, z].

    from scipy.spatial.transform import Rotation

    w, x, y, z = quat[0]
    r = Rotation.from_quat([x, y, z, w])   # scipy wants [x,y,z,w]
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    print(f"RPY: Roll={roll:.4f}°  Pitch={pitch:.4f}°  Yaw={yaw:.4f}°")

────────────────────────────────────────────────────────────────────────
STEP 5 — Compare with this script
────────────────────────────────────────────────────────────────────────
Expected tolerance:  position ≤ 1 mm,  angles ≤ 0.1°

Possible sources of discrepancy:
  ① Frame offset — Isaac Sim's UR5 USD may place the base frame slightly
    above the ground (check the root prim transform).
  ② Axis convention — Isaac Sim is Y-up; DH model is Z-up.
    The base rotation Rx(-90°) bridges the two conventions if needed.
  ③ Joint zero offsets — verify that the USD default joint positions
    match the DH zero configuration (all axes aligned as per the table).

If you see a constant offset, apply a fixed base transform T_base and
pre-multiply: T_corrected = T_base @ T_6^0.
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 70)
    print("  UR5 FORWARD KINEMATICS VALIDATION — Task 1")
    print("  Standard DH Convention  |  numpy only  |  Python 3.10+")
    print("=" * 70)
    print()

    for label, angles_deg in TEST_CONFIGURATIONS:
        print_fk_result(label, angles_deg)

    print_separator("=")
    print("  All 5 configurations computed successfully.")
    print_separator("=")
    print(ISAAC_SIM_GUIDE)


if __name__ == "__main__":
    main()
