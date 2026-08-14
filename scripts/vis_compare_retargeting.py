"""Compare retargeting algorithms side-by-side in one MuJoCo scene.

Loads the BVH human motion together with the robot motions produced by several
retargeting algorithms (COLMO / GMR / OmniRetarget, stored under
``results/lafan1/<algo>/<motion>.pkl``) and renders them all in a single
MuJoCo viewer:

  * one copy of the robot per algorithm, laid out in a row, and
  * the source BVH human skeleton next to them,

with a floating label carrying the algorithm name above each robot's head.

Example
-------
    python scripts/vis_compare_retargeting.py --motion dance1_subject1

    # record a video and drop OmniRetarget from the comparison
    python scripts/vis_compare_retargeting.py --motion dance1_subject1 \
        --algos colmo gmr --record_video --video_path videos/dance1.mp4

    # PNG stills at three frames, human drawn at its true captured size
    # (no viewer needed -- works over ssh with MUJOCO_GL=egl)
    python scripts/vis_compare_retargeting.py --motion dance1_subject1 \
        --human_mode original --snapshot_frames 0 60 120 --snapshot_dir figures

    # every frame from 0 to 500 (inclusive), human alone, no robots
    python scripts/vis_compare_retargeting.py --motion dance1_subject1 \
        --algos --snapshot_frames 0-500 --snapshot_dir figures

``--human_mode`` picks how the human skeleton is drawn: ``scaled`` (default, the
IK config's per-bone ``human_scale_table``, i.e. the reference the retargeter
tracks), ``original`` (the captured human at its true size), or
``scaled_with_keypoint`` (scaled, keybody markers only).

The layout mirrors ``scripts/vis_colmo_with_bvh.py``: the human skeleton is drawn
with ``user_scn`` markers (red joints, yellow bones) while the robots are real
MJCF meshes composed into one model via ``mujoco.MjSpec`` attachment.
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import json

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    ROBOT_XML_DICT, ROBOT_BASE_DICT, IK_CONFIG_DICT, saturate_root_xy)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-algorithm defaults: where the pkl lives, the filename suffix used when
# saving, the on-screen label, and a distinctive color (used both to tint the
# robot mesh and to color the head label marker).
ALGO_TABLE = {
    "colmo": dict(subdir="colmo", suffix="", label="COLMO",
                  color=(0.20, 0.80, 0.35, 1.0)),
    "colmo_shoulder_yaw": dict(subdir="colmo_shoulder_yaw", suffix="",
                               label="COLMO (shoulder-yaw)",
                               color=(0.80, 0.35, 0.85, 1.0)),
    "gmr": dict(subdir="gmr", suffix="", label="GMR",
                color=(0.25, 0.50, 1.00, 1.0)),
    "omniretarget": dict(subdir="omniretarget", suffix="_original",
                         label="OmniRetarget", color=(1.00, 0.55, 0.10, 1.0)),
}

HUMAN_JOINT_COLOR = (1.00, 0.20, 0.20, 1.0)
HUMAN_ROOT_COLOR = (1.00, 0.40, 0.20, 1.0)
HUMAN_BONE_COLOR = (0.90, 0.90, 0.20, 1.0)
HUMAN_LABEL_COLOR = (1.00, 0.30, 0.30, 1.0)


# ---------------------------------------------------------------------------
# Robust pickle loading across numpy versions.
#
# The COLMO pkls are typically written with numpy 1.x (``numpy.core.*``) while
# the GMR / OmniRetarget pkls come from tools running numpy 2.x
# (``numpy._core.*``). A plain ``pickle.load`` in a numpy-1.x interpreter fails
# on the 2.x files with ``ModuleNotFoundError: numpy._core``. Remapping the
# module name *only* during class resolution (never touching ``sys.modules``,
# which corrupts numpy internals and segfaults) reads both directions safely.
# ---------------------------------------------------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                return super().find_class(
                    "numpy.core" + module[len("numpy._core"):], name)
            if module.startswith("numpy.core"):
                return super().find_class(
                    "numpy._core" + module[len("numpy.core"):], name)
            raise


def load_pkl(path):
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


# ---------------------------------------------------------------------------
# Facing alignment.
#
# Each algorithm stores the root in its own world convention (COLMO/GMR face
# one way, OmniRetarget the opposite), so a raw side-by-side comparison shows
# the characters pointing in different directions. ``yaw_align_root`` rotates a
# whole root trajectory about the *vertical line through its first frame* so the
# character's frame-0 heading points at ``target_yaw``. Rotating about frame 0
# (not per-frame) keeps the motion coherent: body orientation AND horizontal
# travel are turned together, and the frame-0 root xy / every z stay fixed.
# ---------------------------------------------------------------------------
def _heading_yaw(forward_xy):
    """World yaw (rad) of a horizontal forward vector."""
    return float(np.arctan2(forward_xy[1], forward_xy[0]))


def yaw_align_root(root_pos, root_rot_wxyz, target_yaw, forward_axis=(1.0, 0.0, 0.0)):
    """Return (root_pos, root_rot) rotated so frame-0 heading == target_yaw.

    ``forward_axis`` is the robot base body's forward direction (+x for the
    Unitree G1/H1 pelvis). ``root_rot_wxyz`` is scalar-first, as MuJoCo wants.
    """
    rot = R.from_quat(root_rot_wxyz, scalar_first=True)
    heading0 = _heading_yaw(rot.apply(forward_axis)[0])
    Rz = R.from_euler("z", target_yaw - heading0)
    p0 = root_pos[0].copy()
    new_pos = Rz.apply(root_pos - p0) + p0            # z untouched (rot is about z)
    new_rot = (Rz * rot).as_quat(scalar_first=True)
    return new_pos, new_rot


# ---------------------------------------------------------------------------
# Left-right mirroring of a robot motion.
#
# Some result pkls store the motion mirrored relative to the others (e.g.
# OmniRetarget vs COLMO/GMR). A left-right mirror of a symmetric humanoid is a
# reflection across the body's sagittal plane, which on the same robot model is:
#   * swap each joint with its left/right partner, and
#   * negate the joints whose axis is odd under the reflection (roll about the
#     forward x-axis and yaw about the up z-axis); pitch (about the lateral
#     y-axis, incl. knee/elbow) keeps its sign,
# together with reflecting the free-base pose across a vertical plane. The
# G1/H1 joints all have pure x/y/z axes with matching left/right conventions,
# so the rule reduces to a name-based swap + a sign flip on roll/yaw joints.
# ---------------------------------------------------------------------------
def build_mirror_spec(xml_path):
    """Return (perm, sign) over the robot's actuated dofs (in qpos order) so
    that ``mirrored_dof = dof[:, perm] * sign``."""
    m = mj.MjModel.from_xml_path(str(xml_path))
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j)
             for j in range(m.njnt)
             if m.jnt_type[j] != mj.mjtJoint.mjJNT_FREE]
    idx = {n: i for i, n in enumerate(names)}
    perm = np.arange(len(names))
    sign = np.ones(len(names))
    for i, n in enumerate(names):
        if n.startswith("left_"):
            partner = "right_" + n[len("left_"):]
        elif n.startswith("right_"):
            partner = "left_" + n[len("right_"):]
        else:
            partner = n                      # centered joint (e.g. waist_*)
        perm[i] = idx.get(partner, i)
        sign[i] = -1.0 if ("roll" in n or "yaw" in n) else 1.0
    return perm, sign


def mirror_root(root_pos, root_rot_wxyz):
    """Reflect a free-base trajectory across the world x-z plane (y -> -y)."""
    pos = root_pos.copy()
    pos[:, 1] *= -1.0
    rot = root_rot_wxyz.copy()               # wxyz: negate x and z, keep w and y
    rot[:, 1] *= -1.0
    rot[:, 3] *= -1.0
    return pos, rot


def build_effective_scale(bones, parents, base_scale):
    """Per-bone scale, mirroring GMR.scale_human_data / vis_colmo_with_bvh.

    Bones absent from the JSON ``human_scale_table`` inherit their nearest
    scaled ancestor's factor (BVH order guarantees parents precede children),
    so intermediate joints move with their limb instead of tearing apart.
    """
    eff = {}
    for idx, b in enumerate(bones):
        if b in base_scale:
            eff[b] = base_scale[b]
        elif parents[idx] >= 0:
            eff[b] = eff[bones[parents[idx]]]
        else:
            eff[b] = 1.0
    for side in ("Left", "Right"):
        mod = f"{side}FootMod"
        if mod not in eff:
            eff[mod] = base_scale.get(mod, eff.get(f"{side}Foot", 1.0))
    return eff


def build_scaled_skeleton(args, frames, bones, parents, base_scale, h_root):
    """Per-bone scale for --human_mode scaled, plus COLMO's base-speed cap.

    The cap makes the drawn skeleton travel at the same (capped) speed as the COLMO
    robot. It reads ``max_base_horizontal_speed`` and ``base_speed_cap_mode`` from the
    --robot's collision_cfg and is a no-op when unset:
      per_frame -> precompute the saturated root xy trajectory with the SAME helper the
                   retargeter uses (returned as ``root_xy_traj``, looked up per frame).
      clip      -> shrink the root xy scale once, as adjust_hips_scale_for_motion does.

    Returns (eff_scale, root_xy_traj); ``root_xy_traj`` is None outside per_frame mode.
    """
    root_xy_traj = None
    try:
        import yaml
        _pp = yaml.safe_load(
            open(ROBOT_XML_DICT[args.robot].parent / "collision_cfg.yaml"))["parameters"]
        _limit = _pp.get("max_base_horizontal_speed")
        _fps = float(_pp.get("motion_fps", 30))
        _mode = str(_pp.get("base_speed_cap_mode", "per_frame")).lower()
    except Exception:
        _limit, _fps, _mode = None, 30.0, "per_frame"
    if _limit and float(_limit) > 0 and h_root in base_scale and len(frames) > 1:
        _rxy = np.array([np.asarray(frames[t][h_root][0], float)[:2]
                         for t in range(len(frames))])
        _s = np.array(base_scale[h_root], dtype=float)
        if _s.ndim == 0:
            _s = np.array([float(_s)] * 3)
        if _mode == "per_frame":
            root_xy_traj, _info = saturate_root_xy(_rxy, _s[:2], _limit, _fps)
            print(f"[skeleton] base-speed cap {_limit} m/s (per_frame) -> saturated "
                  f"{100 * _info['saturated_frac']:.1f}% of frames, base peak "
                  f"{_info['base_peak']:.2f} m/s")
        else:
            _peak = float(np.percentile(
                np.linalg.norm(np.diff(_rxy, axis=0), axis=1) * _fps, 99.5))
            if _peak > 1e-9:
                _cap = float(_limit) / _peak
                _s[0] = min(float(_s[0]), _cap)
                _s[1] = min(float(_s[1]), _cap)
                base_scale[h_root] = _s
                print(f"[skeleton] base-speed cap {_limit} m/s (clip) -> Hips xy scale "
                      f"{float(_s[0]):.4f} (peak {_peak:.2f} m/s)")
    return build_effective_scale(bones, parents, base_scale), root_xy_traj


def build_keypoint_edges(bones, parents, keypoint_set):
    """Reduced skeleton over ``keypoint_set``: connect each keypoint to its NEAREST
    ancestor keypoint in the raw bone hierarchy (skipping non-keypoint bones), so a
    clean connected sub-skeleton is drawn even when the chosen keypoints are not
    directly adjacent (e.g. LeftArm -> LeftShoulder -> Spine2 collapses to LeftArm ->
    Spine2). Computed keybodies ending in 'FootMod' stand in for their raw 'Foot' bone.
    Returns a list of (child_kp, parent_kp) name pairs."""
    name_to_idx = {b: i for i, b in enumerate(bones)}
    kp_to_raw, raw_to_kp = {}, {}
    for kp in keypoint_set:
        raw = kp[:-3] if kp.endswith("FootMod") else kp   # LeftFootMod -> LeftFoot
        if raw in name_to_idx:
            kp_to_raw[kp] = raw
            raw_to_kp[raw] = kp
    edges = []
    for kp, raw in kp_to_raw.items():
        p = parents[name_to_idx[raw]]
        while p >= 0:
            anc = bones[p]
            if anc in raw_to_kp:
                edges.append((kp, raw_to_kp[anc]))
                break
            p = parents[p]
    return edges


# ---------------------------------------------------------------------------
# Scene drawing helpers. They operate on a generic ``mjvScene`` so the same
# calls populate both the interactive ``viewer.user_scn`` and the offscreen
# ``renderer.scene`` used for video recording.
# ---------------------------------------------------------------------------
def add_sphere(scene, pos, radius, rgba, label=None):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    # Always assign the label (empty when None) so a re-used geom slot from a
    # previous frame does not leak a stale label string.
    geom.label = label if label is not None else ""
    scene.ngeom += 1


def add_capsule(scene, from_pos, to_pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=radius,
        from_=np.asarray(from_pos, dtype=np.float64),
        to=np.asarray(to_pos, dtype=np.float64),
    )
    geom.label = ""
    scene.ngeom += 1


def resolve_robot_sources(args, motion):
    """Return an ordered list of robot source dicts for the algorithms present."""
    results_dir = Path(args.results_dir)
    explicit = {"colmo": args.colmo_pkl, "gmr": args.gmr_pkl,
                "omniretarget": args.omni_pkl,}
    sources = []
    for i, key in enumerate(args.algos):
        if key not in ALGO_TABLE:
            print(f"[yellow]Unknown algo '{key}', skipping.[/yellow]")
            continue
        spec = ALGO_TABLE[key]
        if explicit.get(key) is not None:
            path = Path(explicit[key])
        else:
            path = results_dir / spec["subdir"] / f"{motion}{spec['suffix']}.pkl"
        if not path.exists():
            print(f"[yellow]{spec['label']}: {path} not found, skipping.[/yellow]")
            continue
        data = load_pkl(path)
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        # Stored root rotation is xyzw; MuJoCo wants scalar-first (wxyz).
        root_rot = np.asarray(data["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
        sources.append(dict(
            key=key,
            label=spec["label"],
            color=spec["color"],
            prefix=f"a{i}_",
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            n=len(root_pos),
            fps=int(data.get("fps", 30)),
        ))
        print(f"[green]{spec['label']}[/green]: {path.name} "
              f"({len(root_pos)} frames, {dof_pos.shape[1]} dof)")
    return sources


def build_scene_model(xml_path, sources):
    """Compose one MuJoCo model containing all robots (each name-prefixed).

    With no robots (``--algos`` passed no names, or none of them resolved to a pkl)
    the scene becomes the robot XML's own backdrop -- floor, light, skybox and
    <visual> settings -- with the robot itself removed, so the human skeleton is
    drawn on the usual ground rather than into an empty void. Deleting the robot
    body orphans everything that references its joints, so the actuator / sensor /
    keyframe / equality / tendon lists go with it or the compile fails.
    """
    if not sources:
        spec = mj.MjSpec.from_file(str(xml_path))
        for body in list(spec.worldbody.bodies):
            spec.delete(body)
        for referencing in (spec.actuators, spec.sensors, spec.keys,
                            spec.equalities, spec.tendons):
            for item in list(referencing):
                spec.delete(item)
        return spec.compile()

    parent = mj.MjSpec()
    for src in sources:
        child = mj.MjSpec.from_file(str(xml_path))
        # Attach at the origin: a free joint fully overrides any frame offset,
        # so the lateral layout is applied to root_pos at runtime instead.
        frame = parent.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
        parent.attach(child, prefix=src["prefix"], frame=frame)
    return parent.compile()


def tint_robot(model, src):
    """Recolor a robot's visible geoms with its algorithm color."""
    rgb = np.asarray(src["color"][:3], dtype=np.float32)
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            continue
        bid = model.geom_bodyid[gid]
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(src["prefix"]):
            model.geom_rgba[gid, :3] = rgb  # keep original alpha


def null_geom_names(model):
    """Redirect every geom name to a null byte so MuJoCo draws no geom labels.

    Only the head-marker labels we attach to ``user_scn`` geoms should show;
    this suppresses the viewer's automatic per-geom labels on the robot mesh.
    """
    names_arr = np.frombuffer(model.names, dtype=np.uint8)
    null_pos = int(np.where(names_arr == 0)[0][0]) if (names_arr == 0).any() else 0
    for gid in range(model.ngeom):
        model.name_geomadr[gid] = null_pos


NAMED_COLORS = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0),
                "gray": (0.5, 0.5, 0.5), "grey": (0.5, 0.5, 0.5)}


def parse_frame_spec(text):
    """argparse type for one --snapshot_frames token: ``N``, ``A-B`` or ``A-B:STEP``.

    Returns the frame indices the token stands for, with ``A-B`` INCLUSIVE of B, so
    ``--snapshot_frames 0 60 100-200:10`` means frame 0, frame 60, and every 10th
    frame from 100 through 200.
    """
    t = text.strip()
    if "-" not in t:
        try:
            i = int(t)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"bad frame {text!r}; use N, A-B or A-B:STEP")
        if i < 0:
            raise argparse.ArgumentTypeError(f"negative frame {text!r}")
        return [i]
    span, _, step_text = t.partition(":")
    start_text, _, end_text = span.partition("-")
    try:
        start, end = int(start_text), int(end_text)
        step = int(step_text) if step_text else 1
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad frame range {text!r}; use A-B or A-B:STEP")
    if start < 0:
        raise argparse.ArgumentTypeError(f"negative frame in {text!r}")
    if step <= 0:
        raise argparse.ArgumentTypeError(f"step must be positive in {text!r}")
    if end < start:
        raise argparse.ArgumentTypeError(f"empty range {text!r} (end before start)")
    return list(range(start, end + 1, step))


def parse_color(s):
    """argparse type: 'white'/'black'/'gray', '#rrggbb', or 'r,g,b' (0-1 or
    0-255). Returns an (r, g, b) tuple in 0-1."""
    t = s.strip().lower()
    if t in NAMED_COLORS:
        return NAMED_COLORS[t]
    if t.startswith("#") and len(t) == 7:
        return tuple(int(t[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    vals = [float(p) for p in t.replace(",", " ").split()]
    if len(vals) == 3:
        scale = 255.0 if max(vals) > 1.0 else 1.0
        return tuple(v / scale for v in vals)
    raise argparse.ArgumentTypeError(
        f"bad color {s!r}; use a name, #rrggbb, or 'r,g,b'")


def set_skybox_color(model, rgb):
    """Overwrite every skybox texture with a solid color so the rendered
    background is that flat color instead of the scene's gradient sky. Must be
    called BEFORE the GL context/renderer is created (textures upload then)."""
    col = np.clip(np.asarray(rgb, dtype=float) * 255.0, 0, 255).astype(np.uint8)
    for i in range(model.ntex):
        if model.tex_type[i] != mj.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        adr = int(model.tex_adr[i])
        nch = int(model.tex_nchannel[i])
        n = int(model.tex_height[i]) * int(model.tex_width[i]) * nch
        block = model.tex_data[adr:adr + n].reshape(-1, nch)
        block[:, :3] = col[:3]


def apply_background(model, rgb):
    """Make the ENTIRE backdrop a single solid color. The skybox is unlit, so we
    recolor it to ``rgb`` and then HIDE the floor plane(s) (a lit surface would
    shade to gray even when painted white); with the floor gone the white skybox
    fills the whole frame, top to bottom. Shadows are disabled too (nothing left
    to catch them). Must run BEFORE the GL context/renderer is created."""
    set_skybox_color(model, rgb)
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[gid, 3] = 0.0       # hide floor -> skybox shows through
    if model.nlight:
        model.light_castshadow[:] = 0


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BVH human motion and multiple retargeted robot "
                    "motions (COLMO / GMR / OmniRetarget) in one MuJoCo scene.")
    parser.add_argument("--motion", type=str, default="dance1_subject1",
                        help="Motion name (basename shared by the result pkls "
                             "and the BVH file), e.g. dance1_subject1. Use "
                             "'all' to batch over every .bvh in --motion_dir.")
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        choices=list(ROBOT_XML_DICT.keys()))
    parser.add_argument("--background", type=parse_color, default=None,
                        metavar="COLOR",
                        help="Solid background color: a name (white/black/gray), "
                             "a hex '#rrggbb', or 'r,g,b' (0-1 or 0-255). "
                             "Default: keep the scene's gradient sky.")
    parser.add_argument("--algos", nargs="*",
                        default=["colmo", "gmr", "omniretarget"],
                        help="Which algorithms to show, in left-to-right order. "
                             "Pass '--algos' with no names to show no robots at all, "
                             "i.e. the human skeleton alone on the usual ground.")

    # Data locations (defaults resolve relative to the repo root).
    parser.add_argument("--results_dir", type=str,
                        default=str(REPO_ROOT / "results" / "lafan1"))
    parser.add_argument("--motion_dir", type=str,
                        default=str(REPO_ROOT / "human_motion" / "lafan1"))
    parser.add_argument("--colmo_pkl", type=str, default=None)
    parser.add_argument("--gmr_pkl", type=str, default=None)
    parser.add_argument("--omni_pkl", type=str, default=None)
    parser.add_argument("--bvh_file", type=str, default=None,
                        help="Override BVH path (default: "
                             "<motion_dir>/<motion>.bvh).")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--no_human", action="store_true",
                        help="Do not draw the BVH human skeleton.")
    parser.add_argument("--human_mode",
                        choices=["original", "scaled", "scaled_with_keypoint"],
                        default="scaled",
                        help="How the human skeleton is drawn. scaled (default): "
                             "shrunk onto the robot's proportions with the IK config's "
                             "human_scale_table, i.e. the reference the retargeter "
                             "actually tracks. original: the captured human at its true "
                             "size -- no per-bone scaling and no base-speed cap, so it "
                             "is taller than the robots and travels at its own speed. "
                             "scaled_with_keypoint: scaled, but only the retarget "
                             "keybodies are drawn (same as --keypoints_from_config).")
    parser.add_argument("--keypoints", nargs="+", default=None, metavar="BONE",
                        help="If given, draw the human skeleton markers ONLY for these "
                             "bone names (e.g. --keypoints Hips LeftHand RightHand "
                             "LeftFoot RightFoot Head). A bone (capsule) is drawn only "
                             "when BOTH of its endpoints are in the list. Default: draw "
                             "the full skeleton.")
    parser.add_argument("--keypoints_from_config", action="store_true",
                        help="Draw markers only for the retarget keybodies, i.e. the "
                             "keys of human_scale_table in the robot's IK config JSON "
                             "(bvh_<format>_to_<robot>.json). Takes precedence over "
                             "--keypoints. Implied by "
                             "--human_mode scaled_with_keypoint.")

    # Layout / playback.
    parser.add_argument("--spacing", type=float, default=1.5,
                        help="Lateral gap (m) between adjacent characters.")
    parser.add_argument("--axis", choices=["x", "y"], default="y",
                        help="World axis the characters are lined up along.")
    parser.add_argument("--root_mode", choices=["lock", "recenter", "absolute"],
                        default="lock",
                        help="How each character's horizontal root is placed. "
                             "lock: pinned to its lane every frame (in-place, "
                             "best for pose comparison); recenter: recentered on "
                             "its lane at t=0 then free to move; absolute: raw "
                             "world translation plus the lane offset.")
    parser.add_argument("--motion_fps", type=int, default=None,
                        help="Override playback FPS (default: pkl fps).")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no_tint", action="store_true",
                        help="Do not recolor the robots per algorithm.")
    parser.add_argument("--no_labels", action="store_true",
                        help="Hide the floating algorithm labels.")
    parser.add_argument("--label_offset", type=float, nargs=3,
                        default=[0.0, 0.0, 0.22], metavar=("X", "Y", "Z"),
                        help="Offset (m) of the motion-name label from the human's head "
                             "(default: 0 0 0.22 = 0.22 m straight up). Raise Z to lift "
                             "it, use X/Y to slide it sideways.")

    # Facing. Different algorithms store the root in different world
    # conventions, so by default we rotate every character (robots + human) so
    # its frame-0 heading points at the camera. The whole trajectory turns with
    # it, so subsequent turning motion is preserved.
    parser.add_argument("--no_face_forward", action="store_true",
                        help="Keep each character's original world heading "
                             "instead of rotating them to face the camera.")
    parser.add_argument("--face_yaw", type=float, default=None,
                        help="World yaw (deg) every character's front should "
                             "point at, at frame 0 (default: face the camera). "
                             "Add 180 if characters end up facing away.")
    parser.add_argument("--mirror", nargs="*",
                        default=["omniretarget"],
                        metavar="ALGO",
                        help="Algorithms whose motion is shown left-right "
                             "mirrored (default: omniretarget"
                             "which are stored mirrored vs COLMO/GMR). Pass "
                             "'--mirror' with no names to disable.")

    # Camera.
    parser.add_argument("--cam_distance", type=float, default=None,
                        help="Camera distance (m). Default: 4.5 to span the row of "
                             "robots, or 2.8 when the human skeleton is alone on "
                             "screen (--algos with no names).")
    parser.add_argument("--cam_azimuth", type=float, default=None)
    parser.add_argument("--cam_elevation", type=float, default=-15.0)
    parser.add_argument("--no_follow_camera", action="store_true")
    parser.add_argument("--lookat_shift", type=float, default=None,
                        help="Slide the camera reference (lookat) point sideways "
                             "in the image plane: positive = left, negative = "
                             "right (meters). The scene appears to shift the "
                             "opposite way. Default: 0 for a robots-only run "
                             "(camera centered between the robots), 0.73 when the "
                             "human skeleton is drawn (to keep it in frame).")

    # Video recording / snapshots. Both go through the same offscreen renderer, so
    # --video_width/--video_height and --msaa size the PNGs too.
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output video path. Default: videos/<motion>.mp4, "
                             "or videos/all.mp4 when --motion all.")
    parser.add_argument("--snapshot_frames", type=parse_frame_spec, nargs="+",
                        default=None, metavar="SPEC",
                        help="Frames to save as PNG stills into --snapshot_dir. Each "
                             "SPEC is a single index (60), an inclusive range "
                             "(0-500), or a strided range (0-500:10); mix them freely "
                             "('0 60 100-200:10'). Without --record_video the run "
                             "stops after the last requested frame instead of playing "
                             "the whole clip. In batch mode (--motion all) these "
                             "frames are captured for EVERY motion.")
    parser.add_argument("--snapshot_all", action="store_true",
                        help="Save EVERY rendered frame as a PNG still into "
                             "--snapshot_dir.")
    parser.add_argument("--snapshot_dir", type=str, default="figures",
                        help="Where --snapshot_frames / --snapshot_all PNGs are "
                             "written (as <motion>_<robot>_<human_mode>_<frame>.png).")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open the interactive viewer (needed over ssh; "
                             "pair with MUJOCO_GL=egl and --record_video / "
                             "--snapshot_frames). Implied by either output mode, "
                             "which already render offscreen only.")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--video_quality", type=int, default=8,
                        help="Video compression quality (imageio/ffmpeg, 0-10; "
                             "higher = less compression artifacts, bigger file).")
    parser.add_argument("--msaa", type=int, default=8,
                        help="Anti-aliasing samples for the render (MuJoCo "
                             "offsamples). Higher = smoother edges. Try 8 or 16.")
    parser.add_argument("--font_scale", type=int, default=150,
                        choices=[50, 100, 150, 200, 250, 300],
                        help="Label text size in the recorded video (MuJoCo "
                             "font scale, percent). Larger = bigger labels. The "
                             "live viewer always uses MuJoCo's default 150.")

    args = parser.parse_args()

    # A headless GL backend (MUJOCO_GL=egl/osmesa) has no window system, so the live
    # passive viewer cannot open -- creating its context fails and takes the process
    # down with it. Auto-switch to offscreen rendering instead.
    gl_backend = os.environ.get("MUJOCO_GL", "").lower()
    if gl_backend in ("egl", "osmesa") and not args.headless:
        args.headless = True
        print(f"[yellow]MUJOCO_GL={gl_backend}: no on-screen window available -> "
              f"forcing --headless (offscreen render).[/yellow]")

    # Each --snapshot_frames token parsed to a list of indices (ranges expand); flatten
    # them into one sorted, de-duplicated frame list.
    if args.snapshot_frames:
        args.snapshot_frames = sorted(
            {i for spec in args.snapshot_frames for i in spec})

    snapshots = bool(args.snapshot_frames) or args.snapshot_all
    if args.headless and not (args.record_video or snapshots):
        raise SystemExit("--headless with neither --record_video nor "
                         "--snapshot_frames/--snapshot_all would render nothing.")
    if snapshots:
        os.makedirs(args.snapshot_dir, exist_ok=True)
        count = "every frame" if args.snapshot_all \
            else f"{len(args.snapshot_frames)} frame(s)"
        print(f"[cyan]Snapshots ({count}) -> {args.snapshot_dir}/[/cyan]")

    # Default video name follows the motion: videos/<motion>.mp4, or
    # videos/all.mp4 for a full batch (an explicit --video_path overrides this).
    if args.video_path is None:
        name = "all" if args.motion == "all" else args.motion
        args.video_path = f"videos/{name}.mp4"

    # Resolve which motions to compare. "--motion all" batches over every BVH
    # basename found in --motion_dir; otherwise it is just the one motion.
    if args.motion == "all":
        motion_dir = Path(args.motion_dir)
        motions = sorted(p.stem for p in motion_dir.glob("*.bvh"))
        if not motions:
            raise SystemError(f"No .bvh files found in {motion_dir}")
        print(f"[cyan]Batch mode: {len(motions)} motions from "
              f"{motion_dir}[/cyan]")
    else:
        motions = [args.motion]
    batch = len(motions) > 1

    # A single video for the whole run: every motion is appended back-to-back
    # into args.video_path (one file, not one-per-motion).
    mp4_writer = None
    if args.record_video:
        import imageio
        video_dir = os.path.dirname(args.video_path)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir)
        # One fixed fps for the whole file (LAFAN1 is 30 fps throughout).
        # quality raises the bitrate; macro_block_size=None keeps the exact
        # requested resolution (no rounding up to a multiple of 16).
        mp4_writer = imageio.get_writer(args.video_path,
                                        fps=args.motion_fps or 30,
                                        quality=args.video_quality,
                                        macro_block_size=None)
        print(f"[cyan]Recording {len(motions)} motion(s) into one file: "
              f"{args.video_path}[/cyan]")

    try:
        for mi, motion in enumerate(motions):
            if batch:
                print(f"\n[bold cyan]=== [{mi + 1}/{len(motions)}] {motion} "
                      f"===[/bold cyan]")
            # --loop / --bvh_file name a single motion, so ignore them in batch.
            run_comparison(args, motion, mp4_writer=mp4_writer,
                           loop=args.loop and not batch,
                           bvh_override=None if batch else args.bvh_file)
    finally:
        if mp4_writer is not None:
            mp4_writer.close()
            print(f"[cyan]Video saved to {args.video_path}[/cyan]")


def run_comparison(args, motion, mp4_writer, loop, bvh_override):
    """Render one motion's human + robot comparison (interactive and/or video).

    ``mp4_writer`` is a shared, already-open imageio writer (or None): this
    function only appends frames to it and never opens or closes it, so several
    motions can be concatenated into one file by the caller.
    """
    # Resolve the camera azimuth up front: it is both the viewer's azimuth and,
    # by default, the yaw the characters are turned to face (so they look at the
    # camera). Characters line up along y -> camera on the x-axis, and vice versa.
    # The camera looks from `azimuth + 180`, so the characters' fronts must point
    # there (not at `azimuth`) to face it.
    cam_azimuth = args.cam_azimuth if args.cam_azimuth is not None \
        else (180.0 if args.axis == "y" else 90.0)
    face_forward = not args.no_face_forward
    target_yaw = np.deg2rad(args.face_yaw if args.face_yaw is not None
                            else cam_azimuth + 180.0)

    # World-space vector pointing to the *left* of the image, so `--lookat_shift`
    # can slide the camera's reference point sideways. The horizontal view
    # direction is -(cos, sin) of the azimuth; screen-left = view x up.
    _az = np.deg2rad(cam_azimuth)
    _fwd = np.array([-np.cos(_az), -np.sin(_az), 0.0])
    left_dir = np.cross(_fwd, [0.0, 0.0, 1.0])   # screen-left; scaled by the
    # resolved lookat_shift once we know whether the human is drawn (below).

    # --- Load robot motions -------------------------------------------------
    # Zero robots is a legitimate request ("--algos" with no names, to look at the
    # human alone), so it is only fatal when there is no human to fall back on.
    sources = resolve_robot_sources(args, motion)
    if not sources:
        reason = ("no algorithms requested" if not args.algos
                  else f"no robot motions found for {args.algos}")
        if args.no_human:
            print(f"[yellow]{motion}: {reason}, and --no_human leaves nothing "
                  f"to draw -- skipping.[/yellow]")
            return
        print(f"[yellow]{motion}: {reason} -- drawing the human skeleton "
              f"alone.[/yellow]")

    # Left-right mirror the requested algorithms (before facing alignment, so
    # the mirrored motion is then re-oriented to face the camera like the rest).
    if args.mirror and sources:
        mirror_perm, mirror_sign = build_mirror_spec(ROBOT_XML_DICT[args.robot])
        for src in sources:
            if src["key"] not in args.mirror:
                continue
            if src["dof_pos"].shape[1] != len(mirror_perm):
                print(f"[yellow]{src['label']}: {src['dof_pos'].shape[1]} dof "
                      f"!= robot's {len(mirror_perm)}, skipping mirror.[/yellow]")
                continue
            src["root_pos"], src["root_rot"] = mirror_root(
                src["root_pos"], src["root_rot"])
            src["dof_pos"] = src["dof_pos"][:, mirror_perm] * mirror_sign[None, :]
            print(f"[magenta]{src['label']}: left-right mirrored.[/magenta]")

    # Turn every robot to face the target heading at frame 0 (see
    # ``yaw_align_root``). This also lines OmniRetarget up with COLMO/GMR, which
    # it otherwise faces 180 deg away from.
    if face_forward:
        for src in sources:
            src["root_pos"], src["root_rot"] = yaw_align_root(
                src["root_pos"], src["root_rot"], target_yaw)

    # --- Load BVH human motion ---------------------------------------------
    human = None
    if not args.no_human:
        bvh_path = Path(bvh_override) if bvh_override \
            else Path(args.motion_dir) / f"{motion}.bvh"
        if bvh_path.exists():
            frames, height = load_bvh_file(str(bvh_path), format=args.format)
            raw = read_bvh(str(bvh_path))
            bones = list(raw.bones)
            parents = list(raw.parents)
            edges = [(bones[i], bones[p]) for i, p in enumerate(parents)
                     if p >= 0 and bones[i] in frames[0] and bones[p] in frames[0]]

            # Per-bone scale, exactly as GMR would build it for this robot: read
            # the same IK-config human_scale_table and rescale it by the actual
            # human height, so the drawn skeleton matches vis_colmo_with_bvh.py.
            with open(IK_CONFIG_DICT[f"bvh_{args.format}"][args.robot],
                      encoding="utf-8") as f:
                ik_config = json.load(f)
            ratio = height / ik_config["human_height_assumption"]
            base_scale = {k: np.asarray(v, dtype=float) * ratio
                          for k, v in ik_config["human_scale_table"].items()}
            h_root = ik_config["human_root_name"]

            # --human_mode original: draw the captured human as it was recorded. Both the
            # per-bone scaling and the base-speed cap below exist ONLY to bring the
            # skeleton onto the robot's proportions, so neither applies here -- the human
            # then stands taller than the robots and travels at its own (uncapped) speed,
            # which is the point of the mode.
            if args.human_mode == "original":
                eff_scale = {b: 1.0 for b in frames[0]}
                root_xy_traj = None
                print(f"[skeleton] original size: no per-bone scaling, no base-speed cap "
                      f"(human {height:.2f} m)")
            else:
                eff_scale, root_xy_traj = build_scaled_skeleton(
                    args, frames, bones, parents, base_scale, h_root)

            # Frame-0 facing from the hip line (convention-independent, so it
            # agrees with the robots' +x heading). Rotate the whole skeleton
            # about the vertical line through the scaled root at frame 0.
            root0 = np.asarray(frames[0][h_root][0], dtype=np.float64)
            pivot = eff_scale.get(h_root, 1.0) * root0
            Rz_human = R.identity()
            if face_forward and "LeftUpLeg" in frames[0] and "RightUpLeg" in frames[0]:
                r_lr = (np.asarray(frames[0]["RightUpLeg"][0])
                        - np.asarray(frames[0]["LeftUpLeg"][0]))
                fwd = np.cross([0.0, 0.0, 1.0], r_lr)
                if np.linalg.norm(fwd[:2]) > 1e-6:
                    Rz_human = R.from_euler(
                        "z", target_yaw - _heading_yaw(fwd))

            human = dict(frames=frames, bones=bones, edges=edges, parents=parents,
                         n=len(frames), height=height,
                         eff_scale=eff_scale, h_root=h_root,
                         pivot=pivot, Rz=Rz_human, root_xy_traj=root_xy_traj,
                         # Retarget keybodies = the IK config's human_scale_table keys
                         # (used by --keypoints_from_config). May include computed bodies
                         # like LeftFootMod that are present in the frames but not raw bones.
                         scale_bones=list(ik_config["human_scale_table"].keys()))
            print(f"[red]BVH human[/red]: {bvh_path.name} "
                  f"({len(frames)} frames, height {height:.2f} m, "
                  f"mode {args.human_mode})")
        else:
            print(f"[yellow]BVH {bvh_path} not found, human skeleton disabled."
                  f"[/yellow]")

    # Optional keypoint filter: when set, only these bones get a marker, and a
    # bone-capsule is drawn only between two shown keypoints. None -> full skeleton.
    # --keypoints_from_config (IK config human_scale_table keys) wins over --keypoints,
    # and --human_mode scaled_with_keypoint is exactly that filter.
    keypoint_set = None
    from_config = (args.keypoints_from_config
                   or args.human_mode == "scaled_with_keypoint")
    if from_config:
        if human is not None:
            keypoint_set = set(human["scale_bones"])
            print(f"[cyan]keypoints from IK config human_scale_table: "
                  f"{sorted(keypoint_set)}[/cyan]")
        else:
            print("[yellow]keypoint filter ignored (no human skeleton).[/yellow]")
    elif args.keypoints:
        keypoint_set = set(args.keypoints)
    if keypoint_set is not None and human is not None:
        # Validate against the bones present in the motion frames (which, unlike the raw
        # BVH bones, include computed keybodies such as LeftFootMod).
        valid = set(human["frames"][0].keys())
        unknown = keypoint_set - valid
        if unknown:
            print(f"[yellow]keypoints: unknown bone(s) {sorted(unknown)}. "
                  f"Valid: {sorted(valid)}[/yellow]")

    # Bones to draw between the kept keypoints: a reduced skeleton connecting each
    # keypoint to its nearest ancestor keypoint (so the sub-skeleton stays connected).
    keypoint_edges = None
    if keypoint_set is not None and human is not None:
        keypoint_edges = build_keypoint_edges(
            human["bones"], human["parents"], keypoint_set)

    if not sources and human is None:
        print(f"[yellow]{motion}: no robot motions and no BVH -- nothing to draw, "
              f"skipping.[/yellow]")
        return

    # --- Column layout ------------------------------------------------------
    # Human (if any) sits at column 0, robots follow. Columns are centered on
    # the origin along the chosen axis.
    columns = (["human"] if human else []) + [s["key"] for s in sources]
    ncol = len(columns)
    axis_idx = 0 if args.axis == "x" else 1

    def column_offset(col_index):
        off = np.zeros(3)
        off[axis_idx] = (col_index - (ncol - 1) / 2.0) * args.spacing
        return off

    # ``horizontal_shift`` maps a character's horizontal root at frame ``t`` to
    # its on-screen position for the chosen root_mode. The vertical axis (z) is
    # always left untouched so the feet keep contact with the floor.
    def horizontal_shift(offset, root0_xy, cur_xy):
        if args.root_mode == "lock":       # pin root to the lane every frame
            return np.array([offset[0] - cur_xy[0], offset[1] - cur_xy[1], 0.0])
        if args.root_mode == "recenter":   # recenter on the lane at t=0
            return np.array([offset[0] - root0_xy[0], offset[1] - root0_xy[1], 0.0])
        return np.array([offset[0], offset[1], 0.0])  # absolute

    if human:
        human["offset"] = column_offset(0)
        # Frame-0 root after scaling + facing rotation: the rotation pivots on
        # this point, so it equals ``pivot`` (the recenter/lock anchor).
        human["root0_xy"] = human["pivot"][:2]
    for k, src in enumerate(sources):
        src["offset"] = column_offset((1 if human else 0) + k)
        src["root0_xy"] = src["root_pos"][0, :2]

    def human_root_world(i):
        """On-screen position of the human's root at frame ``i``.

        The same expression ``draw_overlays`` evaluates for the root marker, i.e.
        ``rp(h_root) + shift`` -- the root's own local offset is zero, so scaling
        collapses to the root term. Only the follow-camera needs it outside the
        draw path (to track the human when no robot is on screen).
        """
        frame = human["frames"][min(i, human["n"] - 1)]
        h_root = human["h_root"]
        raw_root = np.asarray(frame[h_root][0], dtype=np.float64)
        scaled_root = np.asarray(human["eff_scale"].get(h_root, 1.0) * raw_root,
                                 dtype=np.float64)
        if human["root_xy_traj"] is not None:
            scaled_root = scaled_root.copy()
            scaled_root[:2] = human["root_xy_traj"][min(i, human["n"] - 1)]
        p = human["pivot"] + human["Rz"].apply(scaled_root - human["pivot"])
        return p + horizontal_shift(human["offset"], human["root0_xy"], p[:2])

    # Camera anchor: the fixed horizontal center of the ROBOT lanes (the human
    # column, if any, is excluded), so the view sits between the N robots for any
    # N. With no robots the human IS the scene, so it anchors on the human column
    # instead. lookat_shift then slides it sideways; its default is 0 for a
    # robots-only run (dead-centered) and 0.73 when the human is drawn alongside
    # robots (nudged so the human column stays in frame) -- but 0 again when the
    # human is alone, since there is nothing to make room for.
    robots_center = np.mean([s["offset"] for s in sources], axis=0) if sources \
        else human["offset"]
    shift_val = args.lookat_shift if args.lookat_shift is not None \
        else (0.73 if (human and sources) else 0.0)
    lookat_offset = left_dir * shift_val

    # --- Compose and prepare the MuJoCo model ------------------------------
    xml_path = ROBOT_XML_DICT[args.robot]
    robot_base = ROBOT_BASE_DICT[args.robot]
    model = build_scene_model(xml_path, sources)
    data = mj.MjData(model)

    # Body to hang the algorithm label on, with the extra z-clearance needed to
    # float it above the head. Robots differ: G1 has ``head_mocap`` above the
    # head, H1 has neither and tops out at ``torso_link``; fall back to the base
    # body with generous clearance so the label still clears any humanoid head.
    head_anchors = (("head_mocap", 0.15), ("head_link", 0.35),
                    ("torso_link", 0.65))
    for src in sources:
        free_jid = next(
            jid for jid in range(model.njnt)
            if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE
            and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or "")
            .startswith(src["prefix"]))
        src["qadr"] = int(model.jnt_qposadr[free_jid])
        src["ndof"] = src["dof_pos"].shape[1]
        # The pkl's dof count must match this robot's model, otherwise the qpos
        # write would spill into the neighbouring robot's block (or past nq).
        model_ndof = sum(
            1 for jid in range(model.njnt)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or "")
            .startswith(src["prefix"])
            and model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE)
        if src["ndof"] != model_ndof:
            raise SystemError(
                f"{src['label']}: pkl has {src['ndof']} dof but robot "
                f"'{args.robot}' expects {model_ndof}. The result pkl must "
                f"match the --robot model.")
        src["base_bid"] = model.body(src["prefix"] + robot_base).id
        src["head_bid"] = src["base_bid"]
        src["label_z"] = 0.95
        for cand, dz in head_anchors:
            try:
                src["head_bid"] = model.body(src["prefix"] + cand).id
                src["label_z"] = dz
                break
            except KeyError:
                continue
        if not args.no_tint:
            tint_robot(model, src)

    null_geom_names(model)

    # Solid background color (skybox override), if requested. Must run before the
    # viewer / offscreen renderer is created, since textures upload to the GL
    # context at that point.
    if args.background is not None:
        apply_background(model, args.background)

    # Floor grid + lighting are defined in the robot scene XML (a flat floor and
    # a directional light), so nothing extra is needed here.

    total_frames = max([s["n"] for s in sources]
                       + ([human["n"]] if human else []))
    # Playback rate comes from the first robot's pkl; with no robots the BVH sets
    # it (LAFAN1 is 30 fps, which is also the fallback the pkls carry).
    motion_fps = args.motion_fps or (sources[0]["fps"] if sources else 30)

    # --- Viewer / camera setup ----------------------------------------------
    # When writing a video or PNG stills we render OFFSCREEN ONLY. Opening the
    # live GLFW window at the same time as the offscreen mj.Renderer makes the
    # two share/fight over the GL context, which tears or corrupts the captured
    # frames (much worse under a mismatched GPU driver). So while rendering we
    # skip the passive viewer and drive a standalone camera + option instead.
    # For a robust headless GL context, run with ``MUJOCO_GL=egl``.
    snapshot_set = set(args.snapshot_frames or [])
    need_render = mp4_writer is not None or snapshot_set or args.snapshot_all
    paused = [False]

    def key_callback(keycode):
        if keycode == 32:  # GLFW_KEY_SPACE
            paused[0] = not paused[0]
            print(f"[{'PAUSED' if paused[0] else 'RESUMED'}] (SPACE toggles)")

    if need_render or args.headless:
        viewer = None
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        opt = mj.MjvOption()
    else:
        viewer = mjv.launch_passive(model=model, data=data,
                                    show_left_ui=False, show_right_ui=False,
                                    key_callback=key_callback)
        cam = viewer.cam
        opt = viewer.opt

    opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    opt.geomgroup[2] = 0  # hide collision geoms
    # Model-geom labels off; the explicitly-set user_scn head labels still draw.
    opt.label = mj.mjtLabel.mjLABEL_NONE

    # Default framing spans the row of robots; with the human alone on screen there
    # is no row to span, so pull in to a single-character distance.
    if args.cam_distance is not None:
        cam.distance = args.cam_distance
    else:
        cam.distance = 4.5 if sources else 2.8
    cam.azimuth = cam_azimuth
    cam.elevation = args.cam_elevation
    cam.lookat[:] = robots_center + lookat_offset

    # --- Offscreen renderer (video + snapshots) -----------------------------
    # The mp4 writer is owned by the caller (shared across motions); here we only
    # build this motion's offscreen renderer and append frames to it. Snapshots
    # come out of the same renderer, so they share the resolution and AA settings.
    renderer = None
    if need_render:
        # MjSpec.attach does not carry over the child XML's <visual><global>
        # offscreen buffer size, so the composed model keeps MuJoCo's 640x480
        # default. mj.Renderer refuses any render larger than that buffer, so
        # grow it to fit the requested video resolution before constructing it.
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth),
                                         args.video_width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight),
                                          args.video_height)
        # Anti-aliasing: MjSpec.attach drops the child <visual><quality>, so set
        # the multisample count on the composed model before building the
        # renderer (the renderer reads it when creating its GL context).
        model.vis.quality.offsamples = max(int(model.vis.quality.offsamples),
                                           args.msaa)
        renderer = mj.Renderer(model, height=args.video_height,
                               width=args.video_width,
                               font_scale=mj.mjtFontScale(args.font_scale))

    def draw_overlays(scene):
        """Draw the human skeleton and floating algorithm labels into a scene."""
        # Human skeleton (markers + bones), scaled to robot proportions (like
        # vis_colmo_with_bvh) and turned to face the camera, at its column.
        if human:
            frames = human["frames"]
            frame = frames[min(cur_i, human["n"] - 1)]
            h_root = human["h_root"]
            eff_scale = human["eff_scale"]
            pivot, Rz = human["pivot"], human["Rz"]
            raw_root = np.asarray(frame[h_root][0], dtype=np.float64)
            scaled_root = np.asarray(eff_scale.get(h_root, 1.0) * raw_root, dtype=np.float64)
            # "per_frame" base-speed cap: the horizontal root is the pre-integrated saturated
            # trajectory, not a scalar multiply (see saturate_root_xy). None in "clip" mode,
            # where the shrink is already baked into eff_scale.
            if human["root_xy_traj"] is not None:
                scaled_root = scaled_root.copy()
                scaled_root[:2] = human["root_xy_traj"][min(cur_i, human["n"] - 1)]

            def rp(bone):
                # Scale about the root (vis_colmo_with_bvh convention), then rotate
                # the whole skeleton about the frame-0 root's vertical line.
                local = (np.asarray(frame[bone][0]) - raw_root) \
                    * eff_scale.get(bone, 1.0)
                return pivot + Rz.apply((scaled_root - pivot) + local)

            hips_xy = rp(h_root)[:2]
            shift = horizontal_shift(human["offset"], human["root0_xy"], hips_xy)

            def wp(bone):
                return rp(bone) + shift

            # Iterate the keypoint set directly (so computed keybodies like LeftFootMod,
            # absent from raw bones but present in the frame, still draw); else full skeleton.
            marker_bones = human["bones"] if keypoint_set is None else keypoint_set
            for bone in marker_bones:
                if bone not in frame:
                    continue
                if bone == h_root:
                    add_sphere(scene, wp(bone), 0.045, HUMAN_ROOT_COLOR)
                else:
                    add_sphere(scene, wp(bone), 0.030, HUMAN_JOINT_COLOR)
            # Full skeleton edges, or the reduced keypoint sub-skeleton when filtering.
            draw_edges = human["edges"] if keypoint_set is None else keypoint_edges
            for child, parent in draw_edges:
                if child not in frame or parent not in frame:
                    continue
                add_capsule(scene, wp(parent), wp(child), 0.012, HUMAN_BONE_COLOR)
            if not args.no_labels and "Head" in frame:
                add_sphere(scene, wp("Spine2") + np.array([0, 0, 0.15]),
                           0.05, HUMAN_LABEL_COLOR,
                           label=f"{motion}"
                        #    label=f"BVH (human): {motion}"
                           )

        # Floating algorithm labels above each robot head.
        if not args.no_labels:
            for src in sources:
                head_pos = data.xpos[src["head_bid"]] + \
                    np.array([0, 0, src["label_z"]])
                add_sphere(scene, head_pos, 0.05, src["color"], label=src["label"])

    # PNG stills. Named per motion so a batch run (and repeated runs with a
    # different --human_mode) never overwrite each other's figures.
    snap_count = 0
    # Naming each PNG as it lands is useful for a handful of figures and pure noise
    # for a whole range, so past a few frames only the closing tally is printed.
    verbose_snaps = not args.snapshot_all and 0 < len(snapshot_set) <= 12
    if snapshot_set or args.snapshot_all:
        import imageio
        tag = "" if args.no_human else f"_{args.human_mode}"
        snap_name = f"{motion}_{args.robot}{tag}"
        late = sorted(i for i in snapshot_set if i >= total_frames)
        if late:
            shown = late if len(late) <= 8 else late[:8] + ["..."]
            print(f"[yellow]{len(late)} snapshot frame(s) {shown} are past the end "
                  f"of {motion} ({total_frames} frames) -- not captured.[/yellow]")
    # With snapshots only, there is nothing to capture past the last requested
    # frame, so stop there instead of playing out the whole clip.
    snapshot_stop = max(snapshot_set) if (
        snapshot_set and mp4_writer is None and not args.snapshot_all) else None

    pbar = tqdm(total=(total_frames if snapshot_stop is None
                       else min(total_frames, snapshot_stop + 1)), desc="compare")
    rate_limiter = RateLimiter(frequency=motion_fps, warn=False)
    cur_i = 0

    try:
        while (viewer.is_running() if viewer is not None else True):
            # 1) Push each robot's pose into its qpos block.
            if not paused[0]:
                for src in sources:
                    t = min(cur_i, src["n"] - 1)
                    adr, ndof = src["qadr"], src["ndof"]
                    root = src["root_pos"][t]
                    shift = horizontal_shift(src["offset"], src["root0_xy"],
                                             root[:2])
                    data.qpos[adr:adr + 3] = root + shift
                    data.qpos[adr + 3:adr + 7] = src["root_rot"][t]
                    data.qpos[adr + 7:adr + 7 + ndof] = src["dof_pos"][t]
                mj.mj_forward(model, data)

            # 2) Camera stays horizontally centered on the fixed robot-lane
            # center (so it never drifts toward one robot) and follows only the
            # vertical motion of the robot bases, keeping the row framed on jumps
            # / crouches. With no robots it tracks the human's root height instead,
            # which is the same behaviour applied to the only character on screen.
            if not args.no_follow_camera:
                z = float(np.mean([data.xpos[s["base_bid"]][2] for s in sources])) \
                    if sources else float(human_root_world(cur_i)[2])
                cam.lookat[:] = np.array(
                    [robots_center[0], robots_center[1], z]) + lookat_offset

            # 3) Overlays for the live viewer (the offscreen recorder re-draws
            #    them into its own render scene below).
            if viewer is not None:
                viewer.user_scn.ngeom = 0
                draw_overlays(viewer.user_scn)
                viewer.sync()

            # 4) Offscreen frame -- one render feeds both the video and the PNG
            #    stills (overlays must be re-drawn into the render scene). Skipped
            #    while paused so the recording is not padded with dupes.
            if renderer is not None and not paused[0]:
                renderer.update_scene(data, camera=cam,
                                      scene_option=opt)
                draw_overlays(renderer.scene)
                pixels = renderer.render()
                if mp4_writer is not None:
                    mp4_writer.append_data(pixels)
                if args.snapshot_all or cur_i in snapshot_set:
                    out = os.path.join(args.snapshot_dir,
                                       f"{snap_name}_{cur_i:05d}.png")
                    imageio.imwrite(out, pixels)
                    snap_count += 1
                    if verbose_snaps:
                        print(f"[cyan]snapshot[/cyan] {out}")

            # Cap to real time only for live viewing; when rendering, go as fast
            # as possible (the mp4 fps is fixed) so batches finish quickly.
            if renderer is None:
                rate_limiter.sleep()

            if paused[0]:
                continue
            pbar.update(1)

            cur_i += 1
            if snapshot_stop is not None and cur_i > snapshot_stop:
                break
            if cur_i >= total_frames:
                if loop:
                    cur_i = 0
                    pbar.reset()
                else:
                    break
    finally:
        pbar.close()
        if snap_count and not verbose_snaps:
            print(f"[cyan]{snap_count} snapshot(s) -> "
                  f"{os.path.join(args.snapshot_dir, snap_name)}_*.png[/cyan]")
        if viewer is not None:
            viewer.close()
        # Free this motion's GL renderer (the caller keeps the shared writer
        # open so the next motion appends to the same file).
        if renderer is not None:
            renderer.close()
        time.sleep(0.3)


if __name__ == "__main__":
    main()
