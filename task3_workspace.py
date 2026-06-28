#!/usr/bin/env python3
"""
task3_workspace.py  ·  UR5 reachable-workspace point cloud (Section 4.1.5)
==========================================================================
Monte Carlo sample of joint space -> forward kinematics -> 2 x 2 grid of
views of the resulting end-effector point cloud. Pure numpy + matplotlib,
no Isaac Sim required.

The FK and DH parameters come from Task 1 (``task1_fk_validation``) — the
same convention used by every other script in this project.

Usage
-----

    python3 task3_workspace.py
    python3 task3_workspace.py --samples 200000 --seed 42
    python3 task3_workspace.py --no-show          # headless PNG refresh

If the system Python complains about a numpy/matplotlib ABI mismatch
(``ModuleNotFoundError: No module named 'matplotlib.tri.triangulation'``
or similar), pick one of:

  1. Use Isaac Sim's bundled Python — its matplotlib + numpy are already
     compatible::

        /home/ubuntu/Simulators/isaacsim-6.0/python.sh \\
            /home/ubuntu/ur5-kinematics-project/task3_workspace.py

  2. Upgrade matplotlib in user-site so it picks up the numpy 2.x ABI::

        python3 -m pip install --user --upgrade matplotlib
        python3 task3_workspace.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

sys.path.insert(0, str(Path(__file__).parent))
from task1_fk_validation import UR5_DH_PARAMS, forward_kinematics

DEFAULT_N_SAMPLES: int = 150_000
JOINT_LIMIT: float = math.pi          # ±π per joint, per the Section 4.1.5 spec
DEFAULT_IMAGE_PATH: Path = Path(__file__).parent / "images" / "task3_workspace.png"


def sample_workspace(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``(n_samples, 3)`` array of EE positions for uniform joint draws."""
    qs = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(n_samples, 6))
    pts = np.empty((n_samples, 3), dtype=float)
    for i in range(n_samples):
        pts[i] = forward_kinematics(qs[i])[:3, 3]
    return pts


def build_figure(pts: np.ndarray) -> plt.Figure:
    """4-panel view of the EE point cloud: 3D isometric + 3 orthographic projections."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"UR5 Reachable Workspace  ·  Monte Carlo, N = {pts.shape[0]:,} samples",
        fontsize=15, fontweight="semibold",
    )

    # The (0, 0) slot needs to be 3-D; plt.subplots only makes 2-D axes,
    # so swap that one for a 3-D projection.
    axes[0, 0].remove()
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                 s=1, alpha=0.10, c=pts[:, 2], cmap="viridis")
    ax3d.set_title("3D Isometric View")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.view_init(elev=25, azim=35)
    ax3d.set_box_aspect((1, 1, 1))

    ax_xy = axes[0, 1]
    ax_xy.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.10, c="#264653")
    ax_xy.set_title("Top View  (X–Y plane)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)

    ax_xz = axes[1, 0]
    ax_xz.scatter(pts[:, 0], pts[:, 2], s=1, alpha=0.10, c="#2A9D8F")
    ax_xz.set_title("Front View  (X–Z plane)")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.3)

    ax_yz = axes[1, 1]
    ax_yz.scatter(pts[:, 1], pts[:, 2], s=1, alpha=0.10, c="#E76F51")
    ax_yz.set_title("Side View  (Y–Z plane)")
    ax_yz.set_xlabel("Y (m)")
    ax_yz.set_ylabel("Z (m)")
    ax_yz.set_aspect("equal", adjustable="box")
    ax_yz.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UR5 reachable workspace point cloud via Monte Carlo FK.",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_N_SAMPLES,
                        help=f"Monte Carlo sample count (default {DEFAULT_N_SAMPLES:,})")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducibility (default 0)")
    parser.add_argument("--save", type=Path, default=DEFAULT_IMAGE_PATH,
                        help="Where to save the figure PNG "
                             f"(default {DEFAULT_IMAGE_PATH})")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show() — useful in headless mode.")
    args = parser.parse_args()

    print(f"[task3] sanity      : UR5_DH_PARAMS has {len(UR5_DH_PARAMS)} rows")
    print(f"[task3] sampling    : {args.samples:,} joint configs "
          f"uniformly in [-π, π]")
    rng = np.random.default_rng(args.seed)
    pts = sample_workspace(args.samples, rng)
    print(f"[task3] EE bbox (m) :")
    for axis_label, col in zip("XYZ", range(3)):
        lo, hi = float(pts[:, col].min()), float(pts[:, col].max())
        print(f"            {axis_label}: [{lo:+.3f}, {hi:+.3f}]"
              f"   span {hi - lo:.3f} m")

    fig = build_figure(pts)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"[task3] saved figure -> {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
