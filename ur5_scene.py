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
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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


# ---------------------------------------------------------------------------
# Robot placement — works around USDs (like NVIDIA's UR5) whose ``base_link``
# origin sits inside the visible base cylinder instead of at its bottom face.
# Placing such a USD at ``z = TABLE_HEIGHT`` buries the lower half of the
# base in the desk; we measure the local Z-min and lift accordingly.
# ---------------------------------------------------------------------------
BASE_VERTICAL_OFFSET_FALLBACK: float = 0.089   # UR5 DH d1 (m) — used if bbox fails


def place_robot_on_desk(
    stage,
    robot_prim_path: str,
    desk_top_z: float,
    simulation_app=None,
    warmup_frames: int = 10,
) -> float:
    """Translate the robot so its geometry sits flush on ``desk_top_z``.

    NVIDIA's UR5 USD uses heavy instancing — the cylindrical base sits
    inside an instance, so ``UsdGeom.BBoxCache.ComputeLocalBound`` (which
    walks the direct prim hierarchy) under-reports the geometry extent
    by ~40 mm. Two compensations:

    1. If a ``simulation_app`` is passed, pump ``warmup_frames`` updates
       first so payloads and instance proxies fully resolve before we
       query the bbox.
    2. Take ``z_min = min(bbox_z_min, -BASE_VERTICAL_OFFSET_FALLBACK)``
       so we always lift by at least the UR5 DH ``d1`` (~89 mm) — that's
       the spec-correct base-to-shoulder distance and a safe lower bound
       for the visible base height.

    Returns the actual Z translate applied (logged to stdout for proof).
    """
    if simulation_app is not None:
        for _ in range(warmup_frames):
            simulation_app.update()

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )
    local_box = bbox_cache.ComputeLocalBound(robot_prim).ComputeAlignedBox()
    bbox_z_min = float(local_box.GetMin()[2])
    if not np.isfinite(bbox_z_min):
        bbox_z_min = 0.0   # treat NaN/inf as "no info"

    z_min = min(bbox_z_min, -BASE_VERTICAL_OFFSET_FALLBACK)
    z_translate = desk_top_z - z_min

    xf = UsdGeom.Xformable(robot_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_translate))
    print(f"[ur5_scene] robot placed at z = {z_translate:.4f} m  "
          f"(bbox z_min = {bbox_z_min:+.4f} m, "
          f"fallback = -{BASE_VERTICAL_OFFSET_FALLBACK:.4f} m, "
          f"used = {z_min:+.4f} m)")
    return z_translate


# ---------------------------------------------------------------------------
# Viewport camera — close-up of the desk + UR5
# ---------------------------------------------------------------------------
# Captured live from /OmniverseKit_Persp after the user framed the shot
# manually in the viewport (Script Editor readout).
DEFAULT_CAMERA_EYE:    tuple[float, float, float] = ( 0.289,  1.433, 2.217)
DEFAULT_CAMERA_TARGET: tuple[float, float, float] = (-0.339, -0.351, 0.750)


def set_camera_view(
    stage,
    eye:    tuple[float, float, float] = DEFAULT_CAMERA_EYE,
    target: tuple[float, float, float] = DEFAULT_CAMERA_TARGET,
    up:     tuple[float, float, float] = (0.0, 0.0, 1.0),
    camera_path: str = "/OmniverseKit_Persp",
) -> None:
    """Point the perspective viewport camera at ``target`` from ``eye``.

    Authoring a raw ``xformOp:transform`` on ``/OmniverseKit_Persp`` is
    silently overridden by the viewport navigator, which drives the
    camera through ``omni:kit:centerOfInterest`` + ``ViewportCameraState``.
    Delegate to the bundled helper that touches both.
    """
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Sdf

    viewport_api = get_active_viewport()
    if viewport_api is None:
        print("[ur5_scene] no active viewport; skipping camera set")
        return

    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        print(f"[ur5_scene] camera prim {camera_path} not found; skipping camera set")
        return

    coi_prop = cam_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_prop or not coi_prop.IsValid():
        cam_prim.CreateAttribute(
            "omni:kit:centerOfInterest",
            Sdf.ValueTypeNames.Vector3d,
            True,
            Sdf.VariabilityUniform,
        ).Set(Gf.Vec3d(0, 0, -10))

    cam_state = ViewportCameraState(camera_path, viewport_api)
    cam_state.set_position_world(Gf.Vec3d(*eye), True)
    cam_state.set_target_world(Gf.Vec3d(*target), True)
    print(f"[ur5_scene] viewport camera  eye=({eye[0]:.2f},{eye[1]:+.2f},{eye[2]:.2f})  "
          f"target=({target[0]:.2f},{target[1]:+.2f},{target[2]:.2f})")


# ---------------------------------------------------------------------------
# Articulation root pin — defends against USD-authored xform overrides
# ---------------------------------------------------------------------------
def pin_articulation_pose(articulation, z_world: float) -> None:
    """Force the articulation root to ``(0, 0, z_world)`` via the physics API.

    Without this, the PhysX articulation initializer reads the USD's
    authored xform — which may not match the translate we just set with
    ``UsdGeom.Xformable``. Setting through the articulation API anchors
    the physics root to our chosen Z and keeps the base flush on the desk.
    """
    pos = np.array([[0.0, 0.0, float(z_world)]], dtype=np.float32)
    articulation.set_world_poses(positions=pos)
    print(f"[ur5_scene] articulation root pinned at world z = {z_world:.4f} m")


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


def _apply_static_collision(stage, prim_path: str) -> None:
    """Make a prim a static PhysX collider (kinematic rigid body + collision).

    Matches the reference repo's `_apply_static_collision` pattern: a
    `UsdPhysics.CollisionAPI` for the contact surface plus a
    `UsdPhysics.RigidBodyAPI` with `kinematicEnabled = True` so the body
    does not fall under gravity but other rigid bodies still bounce off it.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    UsdPhysics.CollisionAPI.Apply(prim)
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb_api.CreateKinematicEnabledAttr(True)


def build_desk(stage) -> None:
    """Single Cube for the top + four Cube legs, all in warm wood color.

    ``Cube`` from ``isaacsim.core.experimental.objects`` wraps ``UsdGeom.Cube``
    whose ``size`` attribute defaults to **2** (USD spec). Pass ``sizes=[1.0]``
    so ``scales`` reads as actual metres.

    Each desk component is given a PhysX **static collider** so the UR5
    physically collides with the wood instead of passing through it
    (CollisionAPI + kinematic RigidBodyAPI, same pattern as the
    reference repo's ``_apply_static_collision``).
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

    # ---- physics: every desk piece becomes a static collider -----------
    _apply_static_collision(stage, "/World/Desk/Top")
    for i in range(4):
        _apply_static_collision(stage, f"/World/Desk/Leg_{i}")
    print("[ur5_scene] desk built (1 top + 4 legs, all kinematic colliders)")
