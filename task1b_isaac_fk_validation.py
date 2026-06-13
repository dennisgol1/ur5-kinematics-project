#!/usr/bin/env python3
"""
task1b_isaac_fk_validation.py — Task 1b: FK Validation inside Isaac Sim
=========================================================================
Scene layout
------------
  • A wooden workbench (FixedCuboid, 1.5 m × 1.2 m × 0.75 m) sits on the
    ground plane.  The UR5 is mounted on the table surface (base at z = 0.75 m).
  • Physics are enabled so gravity keeps the table on the floor.

What it does
------------
  1. Loads the UR5 from the Omniverse Nucleus asset library.
  2. Steps through the 5 test configurations from task1_fk_validation.py.
  3. Reads the EE world pose, converts it to the robot-base frame.
  4. Compares position / RPY against the DH-math FK result.
  5. Prints a side-by-side table, then keeps the simulation running so you
     can explore the last pose in the viewport.

Usage
-----
    ~/Simulators/isaacsim/python.sh task1b_isaac_fk_validation.py

    # Slower dwell (default 3 s per config):
    ~/Simulators/isaacsim/python.sh task1b_isaac_fk_validation.py --dwell 5

Notes
-----
• All Isaac Sim imports MUST come after SimulationApp() is constructed.
• If Nucleus is not reachable set UR5_USD_LOCAL_PATH to a local .usd file.
• The FK comparison is done in the robot-base frame (not the world frame)
  so the table height does not affect the error numbers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CLI — parsed before SimulationApp so Kit does not swallow our flags
# ---------------------------------------------------------------------------
import argparse
import sys

_parser = argparse.ArgumentParser(
    description="UR5 FK validation: DH math vs Isaac Sim",
    add_help=False,
)
_parser.add_argument(
    "--dwell", type=float, default=3.0,
    help="Seconds to hold each configuration in the viewport (default 3.0)",
)
_parser.add_argument(
    "--renderer", choices=("rtx", "pxr"), default="pxr",
    help="Hydra renderer: 'pxr' (Pixar Storm, default — this machine's "
         "RTX scenedb plugin crashes at startup because the GPU is on a "
         "PCIe Gen4 x1 link, 1/16 the expected bandwidth) or 'rtx'",
)
_parser.add_argument(
    "--motion", choices=("smooth", "snap"), default="smooth",
    help="How to move between FK configs: 'smooth' (cosine-eased "
         "interpolation, default) or 'snap' (instantaneous)",
)
_parser.add_argument(
    "--motion-time", type=float, default=2.0,
    help="Seconds to travel between configs in smooth mode (default 2.0)",
)
cli, _unknown = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# SimulationApp — MUST be the very first Isaac Sim call
# ---------------------------------------------------------------------------
try:
    from isaacsim import SimulationApp           # Isaac Sim 5.x
except ImportError:
    from omni.isaac.kit import SimulationApp     # Isaac Sim 4.x fallback

_app_cfg: dict = {
    "headless": False,
    "width": 1920,
    "height": 1080,
}
if cli.renderer == "pxr":
    # Fallback for machines where librtx.scenedb.plugin.so crashes at
    # carbOnPluginStartup. Surfaces render flat-grey (Storm cannot evaluate
    # MDL), but geometry and physics still work.
    _app_cfg["extra_args"] = [
        "--/app/extensions/excluded/0=omni.hydra.rtx",
        "--/app/extensions/excluded/1=rtx.scenedb",
        "--enable", "omni.hydra.pxr",
        "--/renderer/enabled=pxr",
        "--/renderer/active=pxr",
    ]
simulation_app = SimulationApp(_app_cfg)

# ---------------------------------------------------------------------------
# All remaining imports — AFTER SimulationApp()
# ---------------------------------------------------------------------------
import math
from pathlib import Path

import numpy as np

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

# Prefer the new isaacsim.* surface (Isaac Sim 5.x); fall back to omni.isaac.*.
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.api.robots import Robot
except ImportError:
    from omni.isaac.core import World                       # type: ignore
    from omni.isaac.core.objects import FixedCuboid         # type: ignore
    from omni.isaac.core.robots import Robot                # type: ignore

try:
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
except ImportError:
    try:
        from isaacsim.core.prims import XFormPrim           # type: ignore
    except ImportError:
        from omni.isaac.core.prims import XFormPrim         # type: ignore

try:
    from isaacsim.core.utils.rotations import quat_to_euler_angles
except ImportError:
    from omni.isaac.core.utils.rotations import quat_to_euler_angles  # type: ignore

try:
    from isaacsim.core.utils.stage import add_reference_to_stage
except ImportError:
    from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore

try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    try:
        from omni.isaac.nucleus import get_assets_root_path        # type: ignore
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path  # type: ignore

try:
    from isaacsim.core.utils.viewports import set_camera_view
    _HAS_VIEWPORT_HELPER = True
except ImportError:
    try:
        from omni.isaac.core.utils.viewports import set_camera_view  # type: ignore
        _HAS_VIEWPORT_HELPER = True
    except ImportError:
        _HAS_VIEWPORT_HELPER = False

# Re-use FK math + test configs from Task 1
sys.path.insert(0, str(Path(__file__).parent))
from task1_fk_validation import (
    forward_kinematics,
    rotation_to_euler_rpy,
    TEST_CONFIGURATIONS,
)

# ---------------------------------------------------------------------------
# Scene / robot constants
# ---------------------------------------------------------------------------

# Set to a local path if the Nucleus server is not running, e.g.:
# UR5_USD_LOCAL_PATH = "/home/user/assets/ur5.usd"
UR5_USD_LOCAL_PATH: str | None = None

# Desk geometry — top surface ends up exactly at TABLE_HEIGHT
TABLE_WIDTH:   float = 1.50   # m  (X)
TABLE_DEPTH:   float = 1.20   # m  (Y)
TABLE_HEIGHT:  float = 0.75   # m  (Z, from floor to top surface)
DESK_TOP_THICKNESS: float = 0.05   # m
DESK_LEG_THICKNESS: float = 0.07   # m  (square cross-section)
DESK_LEG_INSET:     float = 0.05   # m  (inset from desk edges)

# Floor slab
FLOOR_SIZE:      float = 8.0    # m  (X = Y)
FLOOR_THICKNESS: float = 0.02   # m

# The UR5 base is placed on the desk surface
ROBOT_BASE_Z: float = TABLE_HEIGHT

# Prim paths
FLOOR_PRIM_PATH  = "/World/Floor"
DESK_TOP_PATH    = "/World/Desk/Top"
DESK_LEGS_PATH   = "/World/Desk/Legs"
ROBOT_PRIM_PATH  = "/World/UR5"
ROBOT_NAME       = "ur5"

# Local textures (ship with Isaac Sim 5.1.0 — no Nucleus dependency)
WOOD_TEXTURE_PATH = (
    "/home/ubuntu/Simulators/isaacsim/extscache/"
    "omni.kit.usdz_export-1.0.9+69cbf6ad/data/test_stage/textures/"
    "mahogany_floorboards.png"
)
TILE_TEXTURE_PATH = (
    "/home/ubuntu/Simulators/isaacsim/extscache/"
    "omni.kit.asset_converter-5.0.17+107.3.1.lx64.r.cp311.u353/"
    "data/materials/C/F/tile.1001.png"
)

# EE prim candidates (tried in order; first valid one wins)
EE_CANDIDATE_PRIMS = [
    f"{ROBOT_PRIM_PATH}/ee_link",
    f"{ROBOT_PRIM_PATH}/tool0",
    f"{ROBOT_PRIM_PATH}/wrist_3_link",
]

# Base prim used to compute base-relative EE pose
BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/base_link"

# Physics settle steps after commanding joint positions
SETTLE_STEPS = 60   # 1 s at 60 Hz — enough for the articulation to snap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_usd() -> str:
    if UR5_USD_LOCAL_PATH:
        p = Path(UR5_USD_LOCAL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Local USD not found: {UR5_USD_LOCAL_PATH}")
        return str(p.resolve())
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Omniverse Nucleus is not reachable.\n"
            "Start the local Nucleus service or set UR5_USD_LOCAL_PATH."
        )
    return f"{root}/Isaac/Robots/UniversalRobots/ur5/ur5.usd"


def _find_ee_prim(stage) -> str:
    for path in EE_CANDIDATE_PRIMS:
        if stage.GetPrimAtPath(path).IsValid():
            return path
    raise RuntimeError(
        "Cannot find the EE prim. Expected one of:\n"
        + "\n".join(f"  {p}" for p in EE_CANDIDATE_PRIMS)
        + "\nCheck the stage hierarchy in Isaac Sim and update EE_CANDIDATE_PRIMS."
    )


def _add_studio_lighting(stage) -> None:
    """Two distant lights (key + fill) + a sphere fill light.

    Avoids ``UsdLux.DomeLight`` on purpose: without an HDR texture, Storm
    spams ``Dome light has no texture asset path`` every frame, and the
    contribution is uniform white anyway — which made the robot blend into
    the white background.
    """
    # Key light: warm directional from the front-upper-right. Wider angle =
    # softer shadows = less facet contrast on the UR5.
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Key"))
    key.CreateIntensityAttr(2000.0)
    key.CreateColorAttr(Gf.Vec3f(1.00, 0.96, 0.88))
    key.CreateAngleAttr(8.0)
    xf = UsdGeom.Xformable(key.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 35.0))

    # Fill light: cooler directional from the opposite side.
    fill = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Fill"))
    fill.CreateIntensityAttr(600.0)
    fill.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.00))
    fill.CreateAngleAttr(10.0)
    xf = UsdGeom.Xformable(fill.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 0.0, -150.0))

    # Overhead softbox — provides the bulk of the ambient term.
    sph = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/Lights/Bounce"))
    sph.CreateIntensityAttr(6000.0)
    sph.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sph.CreateRadiusAttr(1.0)
    xf = UsdGeom.Xformable(sph.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 3.0))
    print("[task1b] Added key + fill + bounce lights")


def _make_preview_material(
    stage,
    mat_path: str,
    diffuse: tuple[float, float, float],
    roughness: float = 0.6,
    metallic: float = 0.0,
) -> UsdShade.Material:
    """Author a ``UsdPreviewSurface`` Storm can shade without an IBL.

    ``metallic=0`` is deliberate: with only directional lights Storm has no
    environment to sample for the metallic specular and produces tiny
    facet-sized highlights that read as noise on faceted geometry.
    """
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*diffuse)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface",
    )
    return material


def _bind_material_recursive(
    stage, root_path: str, material: UsdShade.Material,
) -> int:
    """Bind ``material`` to every Mesh under ``root_path``. Returns count."""
    root = stage.GetPrimAtPath(root_path)
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Mesh":
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            count += 1
    return count


def _make_textured_material(
    stage,
    mat_path: str,
    tex_path: str,
    st_scale: tuple[float, float] = (1.0, 1.0),
    roughness: float = 0.6,
    metallic: float = 0.0,
) -> UsdShade.Material:
    """``UsdPreviewSurface`` whose ``diffuseColor`` is a tiled texture.

    Builds the standard Storm-compatible shader graph:
        UsdPrimvarReader_float2 ('st')
            → (scale)
            → UsdUVTexture (file=tex_path)
            → UsdPreviewSurface.diffuseColor
    """
    material = UsdShade.Material.Define(stage, mat_path)

    primvar = UsdShade.Shader.Define(stage, mat_path + "/PrimvarReader")
    primvar.CreateIdAttr("UsdPrimvarReader_float2")
    primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    primvar_out = primvar.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, mat_path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(st_scale[0], st_scale[1], 1.0, 1.0)
    )
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(primvar_out)
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f) \
          .ConnectToSource(tex_rgb)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface",
    )
    return material


def _add_planar_st_primvar(
    stage,
    cube_root_path: str,
    repeat: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Author an ``st`` primvar on every Mesh under ``cube_root_path``.

    ``isaacsim.core.api.objects.FixedCuboid`` wraps ``UsdGeom.Cube`` (and
    its tessellated Mesh proxy). The Mesh does not author ``st`` by
    default, so Storm's ``UsdUVTexture`` cannot sample our texture.

    This walks the prim tree, finds every Mesh, looks at its face counts
    and authors a per-face-vertex constant tiling using the mesh's XY
    bounds. ``repeat`` is how many times the texture tiles across the
    object's footprint.
    """
    root = stage.GetPrimAtPath(cube_root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        if not face_counts:
            continue
        # Quick-and-dirty planar UVs: each face gets [(0,0), (r,0), (r,r), (0,r)]
        # tiled by ``repeat``. Works for top-down look on a box; sides will
        # repeat but won't read as wrong since they're nearly hidden.
        r_u, r_v = repeat
        uvs = []
        for n in face_counts:
            if n == 4:
                uvs.extend(
                    [(0.0, 0.0), (r_u, 0.0), (r_u, r_v), (0.0, r_v)]
                )
            else:
                # Triangle or n-gon: distribute evenly around a unit triangle.
                uvs.extend([(0.0, 0.0), (r_u, 0.0), (0.0, r_v)] * (n // 3 + 1))
                uvs = uvs[: sum(face_counts)]
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        st = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st.Set([Gf.Vec2f(u, v) for (u, v) in uvs])


_ACCENT_LINK_HINTS: tuple[str, ...] = (
    "joint", "cap", "cover", "collar", "flange",
)


def _apply_robot_livery(stage, robot_prim_path: str) -> None:
    """Per-link UsdPreviewSurface mimicking the real UR5 livery.

    The UR5 is brushed aluminum with darker grey joint hubs. We bind two
    materials, picking by the nearest ancestor link name:

    * silver   — `(0.80, 0.80, 0.82)`, metallic=0.55, roughness=0.35
    * charcoal — `(0.18, 0.18, 0.20)`, metallic=0.55, roughness=0.45
    """
    silver = _make_preview_material(
        stage, "/World/Materials/UR5Silver",
        diffuse=(0.80, 0.80, 0.82), roughness=0.35, metallic=0.55,
    )
    charcoal = _make_preview_material(
        stage, "/World/Materials/UR5Charcoal",
        diffuse=(0.18, 0.18, 0.20), roughness=0.45, metallic=0.55,
    )
    root = stage.GetPrimAtPath(robot_prim_path)
    silver_count = 0
    charcoal_count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        # Walk up to find which link this mesh belongs to.
        path_str = str(prim.GetPath()).lower()
        accent = any(hint in path_str for hint in _ACCENT_LINK_HINTS)
        material = charcoal if accent else silver
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        if accent:
            charcoal_count += 1
        else:
            silver_count += 1
    print(
        f"[task1b] UR5 livery — silver: {silver_count} meshes, "
        f"charcoal: {charcoal_count} meshes"
    )


def _build_floor(world, stage, renderer: str) -> None:
    """Large flat slab at z=0 with a concrete-grey preview material."""
    world.scene.add(
        FixedCuboid(
            prim_path=FLOOR_PRIM_PATH,
            name="floor",
            position=np.array([0.0, 0.0, -FLOOR_THICKNESS / 2.0]),
            scale=np.array([FLOOR_SIZE, FLOOR_SIZE, FLOOR_THICKNESS]),
            color=np.array([0.62, 0.62, 0.64]),
        )
    )
    if renderer == "pxr":
        if Path(TILE_TEXTURE_PATH).exists():
            mat = _make_textured_material(
                stage, "/World/Materials/Floor",
                tex_path=TILE_TEXTURE_PATH,
                st_scale=(1.0, 1.0),       # texture already pre-tiled
                roughness=0.55, metallic=0.05,
            )
            _add_planar_st_primvar(
                stage, FLOOR_PRIM_PATH, repeat=(8.0, 8.0)
            )
            _bind_material_recursive(stage, FLOOR_PRIM_PATH, mat)
            print("[task1b] Floor: tile texture applied (8x8 repeats)")
        else:
            mat = _make_preview_material(
                stage, "/World/Materials/Floor",
                diffuse=(0.62, 0.62, 0.64), roughness=0.85, metallic=0.0,
            )
            _bind_material_recursive(stage, FLOOR_PRIM_PATH, mat)
            print(f"[task1b] Floor: tile texture missing; using flat grey")


def _build_desk(world, stage, renderer: str) -> None:
    """Thin top slab + four square legs, all matte wood."""
    top_z = TABLE_HEIGHT - DESK_TOP_THICKNESS / 2.0
    world.scene.add(
        FixedCuboid(
            prim_path=DESK_TOP_PATH,
            name="desk_top",
            position=np.array([0.0, 0.0, top_z]),
            scale=np.array([TABLE_WIDTH, TABLE_DEPTH, DESK_TOP_THICKNESS]),
            color=np.array([0.55, 0.35, 0.18]),
        )
    )

    # Four legs at the corners, inset slightly from the top edges.
    leg_h = TABLE_HEIGHT - DESK_TOP_THICKNESS
    leg_z = leg_h / 2.0
    half_x = TABLE_WIDTH / 2.0 - DESK_LEG_INSET - DESK_LEG_THICKNESS / 2.0
    half_y = TABLE_DEPTH / 2.0 - DESK_LEG_INSET - DESK_LEG_THICKNESS / 2.0
    for i, (sx, sy) in enumerate(
        [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    ):
        world.scene.add(
            FixedCuboid(
                prim_path=f"{DESK_LEGS_PATH}/Leg{i}",
                name=f"desk_leg_{i}",
                position=np.array([sx * half_x, sy * half_y, leg_z]),
                scale=np.array(
                    [DESK_LEG_THICKNESS, DESK_LEG_THICKNESS, leg_h]
                ),
                color=np.array([0.45, 0.28, 0.13]),
            )
        )

    if renderer == "pxr":
        # Top: textured wood. Legs: flat brown (texture won't read at 7 cm).
        if Path(WOOD_TEXTURE_PATH).exists():
            top_mat = _make_textured_material(
                stage, "/World/Materials/DeskWoodTex",
                tex_path=WOOD_TEXTURE_PATH,
                st_scale=(1.0, 1.0),
                roughness=0.45, metallic=0.0,
            )
            _add_planar_st_primvar(
                stage, DESK_TOP_PATH, repeat=(2.0, 1.5)
            )
            _bind_material_recursive(stage, DESK_TOP_PATH, top_mat)
            print("[task1b] Desk top: wood texture applied")
        else:
            top_mat = _make_preview_material(
                stage, "/World/Materials/DeskWood",
                diffuse=(0.55, 0.35, 0.18), roughness=0.55, metallic=0.0,
            )
            _bind_material_recursive(stage, DESK_TOP_PATH, top_mat)
            print("[task1b] Desk top: wood texture missing; using flat brown")

        legs_mat = _make_preview_material(
            stage, "/World/Materials/DeskLegs",
            diffuse=(0.42, 0.26, 0.13), roughness=0.6, metallic=0.0,
        )
        _bind_material_recursive(stage, DESK_LEGS_PATH, legs_mat)


def _smooth_blend(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
    """Cosine-eased interpolation between two joint vectors.

    alpha in [0, 1].  At the endpoints the velocity is zero, which makes
    the motion read as a deliberate move rather than a teleport.
    """
    s = 0.5 - 0.5 * math.cos(math.pi * float(np.clip(alpha, 0.0, 1.0)))
    return (1.0 - s) * q_start + s * q_end


_MESH_VARIANT_CANDIDATES: tuple[str, ...] = (
    "Quality", "Default", "Visual", "High",
)


def _select_mesh_variant(robot_prim) -> None:
    """Pick the visual-quality mesh variant on the UR5 USD.

    Without an explicit selection the stage falls back to collision proxy
    hulls, which render as a sparse wireframe.
    """
    vsets = robot_prim.GetVariantSets()
    names = vsets.GetNames()
    print(f"[task1b] UR5 variant sets: {names}")
    if "Mesh" not in names:
        print("[task1b] No 'Mesh' variant set on UR5 USD — skipping")
        return
    vset = vsets.GetVariantSet("Mesh")
    options = vset.GetVariantNames()
    chosen = next(
        (v for v in _MESH_VARIANT_CANDIDATES if v in options),
        options[0] if options else None,
    )
    if chosen:
        vset.SetVariantSelection(chosen)
        print(f"[task1b] UR5 mesh variant: {chosen}  (options: {options})")


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix (numpy only)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def _world_to_base(
    ee_pos_w:   np.ndarray,
    ee_quat_w:  np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Express the EE pose in the robot-base frame.
    Inputs / outputs use [w, x, y, z] quaternion convention.

    Returns (pos_in_base [m], quat_in_base [w,x,y,z]).
    """
    R_base = _quat_wxyz_to_rot(base_quat_w)
    R_ee   = _quat_wxyz_to_rot(ee_quat_w)

    # T_base_inv = [ R_base^T | -R_base^T * p_base ]
    R_base_T  = R_base.T
    pos_rel   = R_base_T @ (ee_pos_w - base_pos_w)
    R_rel     = R_base_T @ R_ee

    # R_rel → quaternion [w, x, y, z]
    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R_rel[2, 1] - R_rel[1, 2]) * s
        qy = (R_rel[0, 2] - R_rel[2, 0]) * s
        qz = (R_rel[1, 0] - R_rel[0, 1]) * s
    elif R_rel[0, 0] > R_rel[1, 1] and R_rel[0, 0] > R_rel[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R_rel[0, 0] - R_rel[1, 1] - R_rel[2, 2])
        qw = (R_rel[2, 1] - R_rel[1, 2]) / s
        qx = 0.25 * s
        qy = (R_rel[0, 1] + R_rel[1, 0]) / s
        qz = (R_rel[0, 2] + R_rel[2, 0]) / s
    elif R_rel[1, 1] > R_rel[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R_rel[1, 1] - R_rel[0, 0] - R_rel[2, 2])
        qw = (R_rel[0, 2] - R_rel[2, 0]) / s
        qx = (R_rel[0, 1] + R_rel[1, 0]) / s
        qy = 0.25 * s
        qz = (R_rel[1, 2] + R_rel[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R_rel[2, 2] - R_rel[0, 0] - R_rel[1, 1])
        qw = (R_rel[1, 0] - R_rel[0, 1]) / s
        qx = (R_rel[0, 2] + R_rel[2, 0]) / s
        qy = (R_rel[1, 2] + R_rel[2, 1]) / s
        qz = 0.25 * s

    return pos_rel, np.array([qw, qx, qy, qz])


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

_C           = 16   # numeric column width
_POS_TOL_MM  = 1.0  # position error threshold (mm)
_RPY_TOL_DEG = 0.5  # angle error threshold (degrees)

_ROW_META: list[tuple[str, bool]] = [
    ("X (mm)",    False),
    ("Y (mm)",    False),
    ("Z (mm)",    False),
    ("Roll (°)",  True),
    ("Pitch (°)", True),
    ("Yaw (°)",   True),
]

def _print_table(results: list[dict]) -> None:
    sep  = "=" * 78
    sep2 = "-" * 78
    print("\n" + sep)
    print("  FK COMPARISON  —  DH Math  vs  Isaac Sim  (robot-base frame)")
    print(sep)

    for r in results:
        print(f"\n  {r['label']}")
        joints = "  ".join(f"θ{i+1}={a:>6.1f}°" for i, a in enumerate(r["angles_deg"]))
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
            mark = " ✓" if err < tol else " !"
            print(f"  {axis:<12} {dv:>{_C}.4f}  {sv:>{_C}.4f}  {err:>{_C}.4f}{mark}")
        print()

    print(sep)
    print(f"  ✓ pos < {_POS_TOL_MM} mm  |  ✓ angle < {_RPY_TOL_DEG}°  |  ! = outside tolerance")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    usd_path = _resolve_usd()
    print(f"\n[task1b] UR5 asset : {usd_path}")
    print(f"[task1b] Table height: {TABLE_HEIGHT} m — robot base at z={ROBOT_BASE_Z} m")
    print(f"[task1b] Dwell per config: {cli.dwell} s\n")

    # ── World (60 Hz physics) ─────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)

    # ── Stage (built before world.reset() so xforms author cleanly) ───────
    stage = omni.usd.get_context().get_stage()
    _build_floor(world, stage, cli.renderer)
    _build_desk(world, stage, cli.renderer)

    # ── UR5 robot — base on the table surface ────────────────────────────
    # Capture the prim so we can pick the visual-quality variant and place
    # the robot via UsdGeom.Xformable (the reference-repo pattern).
    robot_prim = add_reference_to_stage(
        usd_path=usd_path, prim_path=ROBOT_PRIM_PATH,
    )
    _select_mesh_variant(robot_prim)

    xf = UsdGeom.Xformable(robot_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ROBOT_BASE_Z))

    robot: Robot = world.scene.add(
        Robot(prim_path=ROBOT_PRIM_PATH, name=ROBOT_NAME)
    )

    world.reset()

    # ── Lighting — key/fill/bounce (no DomeLight: Storm spams without HDR) ─
    _add_studio_lighting(stage)

    # ── Force Storm-compatible material on the robot meshes ───────────────
    # The UR5 USD has only MDL bindings, which Storm cannot evaluate.
    if cli.renderer == "pxr":
        _apply_robot_livery(stage, ROBOT_PRIM_PATH)

    # Confirm where the robot actually ended up after world.reset().
    base_world_pos, _ = robot.get_world_pose()
    print(f"[task1b] Robot world pose: {tuple(round(float(v), 3) for v in base_world_pos)}")

    # ── Camera — clear angled view from front-right, robot fills frame ────
    if _HAS_VIEWPORT_HELPER:
        set_camera_view(
            eye=np.array([2.5, -1.8, 2.2]),
            target=np.array([0.0, 0.0, TABLE_HEIGHT + 0.4]),
            camera_prim_path="/OmniverseKit_Persp",
        )

    # Step a few frames so Storm re-renders the new materials
    for _ in range(5):
        world.step(render=True)
        simulation_app.update()

    # ── Locate EE and base prims ─────────────────────────────────────────
    ee_prim_path = _find_ee_prim(stage)
    ee_xform     = XFormPrim(ee_prim_path)

    # Use base_link if present; fall back to the robot root prim
    base_path = BASE_PRIM_PATH if stage.GetPrimAtPath(BASE_PRIM_PATH).IsValid() \
                else ROBOT_PRIM_PATH
    base_xform = XFormPrim(base_path)

    print(f"[task1b] EE prim  : {ee_prim_path}")
    print(f"[task1b] Base prim: {base_path}\n")

    # ── Iterate test configurations ───────────────────────────────────────
    results: list[dict] = []

    # Track previous target so smooth motion has a start point. The first
    # config interpolates from the home pose returned by the articulation.
    q_prev: np.ndarray = np.asarray(robot.get_joint_positions(), dtype=np.float64)

    motion_steps = max(1, int(cli.motion_time * 60.0))   # 60 Hz physics

    for idx, (label, angles_deg) in enumerate(TEST_CONFIGURATIONS, start=1):
        angles_rad = np.array([math.radians(a) for a in angles_deg], dtype=np.float64)

        print(f"[{idx}/5] {label}")
        print(f"        Joints: {[f'{a}°' for a in angles_deg]}")

        if cli.motion == "smooth":
            # Cosine-eased blend from previous joint vector to the target,
            # commanding intermediate joint positions every physics step.
            for k in range(1, motion_steps + 1):
                q_cmd = _smooth_blend(q_prev, angles_rad, k / motion_steps)
                robot.set_joint_positions(q_cmd)
                world.step(render=True)
                simulation_app.update()
        else:
            robot.set_joint_positions(angles_rad)
            for _ in range(SETTLE_STEPS):
                world.step(render=True)
                simulation_app.update()

        q_prev = angles_rad

        # Read world poses
        ee_pos_w,   ee_quat_w   = ee_xform.get_world_pose()
        base_pos_w, base_quat_w = base_xform.get_world_pose()

        # Express EE in robot-base frame
        ee_pos_b, ee_quat_b = _world_to_base(
            ee_pos_w, ee_quat_w, base_pos_w, base_quat_w,
        )

        sim_pos_mm  = tuple(float(v) * 1000.0 for v in ee_pos_b)
        sim_rpy_deg = tuple(
            math.degrees(float(v))
            for v in quat_to_euler_angles(ee_quat_b)
        )

        # DH FK (base frame, origin at robot base)
        T      = forward_kinematics(angles_rad.tolist())
        dh_pos = tuple(T[i, 3] * 1000.0 for i in range(3))
        dh_rpy = tuple(math.degrees(a) for a in rotation_to_euler_rpy(T[:3, :3]))

        results.append(
            {
                "label":       label,
                "angles_deg":  angles_deg,
                "dh_pos_mm":   dh_pos,
                "sim_pos_mm":  sim_pos_mm,
                "dh_rpy_deg":  dh_rpy,
                "sim_rpy_deg": sim_rpy_deg,
            }
        )

        print(f"        DH  (mm) X={dh_pos[0]:8.2f}  Y={dh_pos[1]:8.2f}  Z={dh_pos[2]:8.2f}")
        print(f"        Sim (mm) X={sim_pos_mm[0]:8.2f}  Y={sim_pos_mm[1]:8.2f}  Z={sim_pos_mm[2]:8.2f}")
        print()

        # Hold the pose so it's visible in the viewport
        dwell_steps = max(1, int(cli.dwell * 60))
        for _ in range(dwell_steps):
            world.step(render=True)
            simulation_app.update()

    # ── Print full comparison table ───────────────────────────────────────
    _print_table(results)

    # Keep the simulation alive — close the window to quit
    print("[task1b] All configurations shown.  Close the Isaac Sim window to exit.\n")
    while simulation_app.is_running():
        world.step(render=True)
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
