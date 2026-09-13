"""Compare the SAME COLMO retarget across DIFFERENT robots, side-by-side.

Where ``vis_compare_retargeting.py`` puts one robot next to itself for several
retarget *algorithms*, this script puts several *robots* (G1 / KAPEX / H1 / T1 /
Go2) next to each other for the same COLMO motion, together with the source BVH
human skeleton, all in one MuJoCo scene:

  * the BVH human skeleton at column 0, then one robot per column, and
  * a floating label above each character's head.

Every robot's own model lights and (all but one) floor are stripped and replaced
by a SINGLE shared light + floor, so the lighting and framing are identical for
every robot (e.g. Go2's directional light no longer makes it look different from
the humanoids' point light). The camera is one shared view of the whole row.

Layout order (default): human, G1, KAPEX(lite), H1, T1, Go2.

Example
-------
    python scripts/vis_compare_robots.py --motion aiming1_subject1

    # record a video, drop the human, only three robots
    python scripts/vis_compare_robots.py --motion aiming1_subject1 \
        --robots g1 h1 go2 --no_human \
        --record_video --video_path videos/compare_aiming1.mp4

    # lossless PNG stills for a figure (stops after the last requested frame)
    python scripts/vis_compare_robots.py --motion aiming1_subject1 \
        --robots g1 kapex --snapshot_frames 0 60 120-200:20 \
        --snapshot_dir figures \
        --video_width 2560 --video_height 1440 --msaa 16 --font_scale 300

Both output modes render OFFSCREEN ONLY (no live window); run them with
``MUJOCO_GL=egl`` for a robust headless GL context over ssh.

The result pkls live in ``results/<DIR>/<motion>.pkl`` (G1_COLMO, kapex_COLMO,
h1_COLMO, t1_COLMO, go2_COLMO); the BVH lives in ``<motion_dir>/<motion>.bvh``.
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

from collision_free_motion_retargeting import (
    ROBOT_XML_DICT, ROBOT_BASE_DICT, IK_CONFIG_DICT)
from collision_free_motion_retargeting.utils.lafan1 import load_bvh_file
from collision_free_motion_retargeting.utils.lafan_vendor.extract import read_bvh

REPO_ROOT = Path(__file__).resolve().parent.parent

# LAFAN1 BVH (Ubisoft La Forge) default location.
DEFAULT_MOTION_DIR = Path(
    "/home/user/ubisoft-laforge-animation-dataset/lafan1/lafan1")

# Per-robot defaults, in the requested left-to-right order. ``robot`` is the
# key into ROBOT_XML_DICT / ROBOT_BASE_DICT (which model to draw); ``subdir`` is
# the results folder the COLMO pkl is read from; ``label`` / ``color`` are the
# on-screen name and marker/tint color. ``kapex`` is drawn with the low-poly
# ``kapex_lite`` model (identical kinematics, faster) but its pkl lives in
# results/kapex_COLMO.
ROBOT_TABLE = {
    "g1":    dict(robot="unitree_g1",  subdir="G1_COLMO",    label="G1",
                  color=(0.20, 0.80, 0.35, 1.0)),
    "kapex": dict(robot="kapex_lite",  subdir="kapex_COLMO", label="KAPEX",
                  color=(0.95, 0.75, 0.15, 1.0)),
    "h1":    dict(robot="unitree_h1",  subdir="h1_COLMO",    label="H1",
                  color=(0.25, 0.50, 1.00, 1.0)),
    "t1":    dict(robot="booster_t1", subdir="t1_COLMO", label="T1",
                  color=(0.85, 0.30, 0.85, 1.0)),
    "go2":   dict(robot="unitree_go2", subdir="go2_COLMO",   label="Go2",
                  color=(1.00, 0.45, 0.10, 1.0), forward=(0.0, 0.0, -1.0)),
}
# Body-frame axis that points "forward" (nose) for facing alignment. The
# humanoids' base (pelvis) uses +x; go2's free-joint base frame is defined with
# a different convention -- its local +x points nearly straight UP, so aligning
# on +x yields a near-vertical, unstable heading and go2 ends up facing wrong.
# go2's true horizontal forward is local -z (verified to match the humanoids'
# +x heading to <0.1 deg across all clips). Robots without an explicit
# ``forward`` default to +x.
DEFAULT_FORWARD = (1.0, 0.0, 0.0)
DEFAULT_ORDER = ["g1", "kapex", "h1", "t1", "go2"]

HUMAN_JOINT_COLOR = (1.00, 0.20, 0.20, 1.0)
HUMAN_ROOT_COLOR = (1.00, 0.40, 0.20, 1.0)
HUMAN_BONE_COLOR = (0.90, 0.90, 0.20, 1.0)
HUMAN_LABEL_COLOR = (1.00, 0.30, 0.30, 1.0)


# ---------------------------------------------------------------------------
# Robust pickle loading across numpy versions (COLMO pkls may be written with
# numpy 1.x, GMR/others with numpy 2.x). Remap the module name only during class
# resolution, never touching sys.modules. (Copied from vis_compare_retargeting.)
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
# Facing alignment: rotate a whole root trajectory about the vertical line
# through its first frame so the frame-0 heading points at ``target_yaw``.
# ---------------------------------------------------------------------------
def _heading_yaw(forward_xy):
    return float(np.arctan2(forward_xy[1], forward_xy[0]))


def yaw_align_root(root_pos, root_rot_wxyz, target_yaw, forward_axis=(1.0, 0.0, 0.0)):
    rot = R.from_quat(root_rot_wxyz, scalar_first=True)
    heading0 = _heading_yaw(rot.apply(forward_axis)[0])
    Rz = R.from_euler("z", target_yaw - heading0)
    p0 = root_pos[0].copy()
    new_pos = Rz.apply(root_pos - p0) + p0
    new_rot = (Rz * rot).as_quat(scalar_first=True)
    return new_pos, new_rot


# ---------------------------------------------------------------------------
# Left-right mirroring of a robot motion (same idea as vis_compare_retargeting,
# where OmniRetarget is stored mirrored vs COLMO/GMR). A sagittal-plane
# reflection of a symmetric robot is: swap each joint with its left/right
# partner, negate the joints whose rotation axis is odd under the reflection
# (roll about x / yaw about z; pitch about y keeps its sign), and reflect the
# free base across the world x-z plane (y -> -y). The sign is read from each
# joint's axis (not its name), so this works for the humanoids' left_/right_
# joints AND the go2 quadruped's FL/FR/RL/RR joints (hip=abduction is roll->x).
# ---------------------------------------------------------------------------
def _mirror_partner(name):
    """Left/right partner joint name under a sagittal mirror, or the name itself
    for a centered joint. Handles humanoid (left_/right_) and go2 (FL/FR/RL/RR)
    naming conventions."""
    pairs = (("left_", "right_"), ("right_", "left_"),
             ("FL_", "FR_"), ("FR_", "FL_"),
             ("RL_", "RR_"), ("RR_", "RL_"))
    for a, b in pairs:
        if name.startswith(a):
            return b + name[len(a):]
    return name


def build_mirror_spec(xml_path):
    """Return (perm, sign) over the robot's actuated dofs (in qpos order) so that
    ``mirrored_dof = dof[:, perm] * sign``."""
    m = mj.MjModel.from_xml_path(str(xml_path))
    jids = [j for j in range(m.njnt)
            if m.jnt_type[j] != mj.mjtJoint.mjJNT_FREE]
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in jids]
    idx = {n: i for i, n in enumerate(names)}
    perm = np.arange(len(names))
    sign = np.ones(len(names))
    for i, (j, n) in enumerate(zip(jids, names)):
        perm[i] = idx.get(_mirror_partner(n), i)
        dom = int(np.argmax(np.abs(m.jnt_axis[j])))   # 0=roll(x) 1=pitch(y) 2=yaw(z)
        sign[i] = -1.0 if dom in (0, 2) else 1.0
    return perm, sign


def mirror_root(root_pos, root_rot_wxyz):
    """Reflect a free-base trajectory across the world x-z plane (y -> -y)."""
    pos = root_pos.copy()
    pos[:, 1] *= -1.0
    rot = root_rot_wxyz.copy()               # wxyz: negate x and z, keep w and y
    rot[:, 1] *= -1.0
    rot[:, 3] *= -1.0
    return pos, rot


# ---------------------------------------------------------------------------
# Human-skeleton scaling helpers (mirror GMR.scale_human_data /
# vis_colmo_with_bvh), unchanged from vis_compare_retargeting.
# ---------------------------------------------------------------------------
def build_effective_scale(bones, parents, base_scale):
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


def build_keypoint_edges(bones, parents, keypoint_set):
    name_to_idx = {b: i for i, b in enumerate(bones)}
    kp_to_raw, raw_to_kp = {}, {}
    for kp in keypoint_set:
        raw = kp[:-3] if kp.endswith("FootMod") else kp
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
# Scene drawing helpers (generic mjvScene, so they populate both the live
# user_scn and the offscreen render scene).
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
    """Ordered list of robot source dicts for the requested --robots."""
    results_dir = Path(args.results_dir)
    sources = []
    for key in args.robots:
        if key not in ROBOT_TABLE:
            print(f"[yellow]Unknown robot '{key}', skipping.[/yellow]")
            continue
        spec = ROBOT_TABLE[key]
        path = results_dir / spec["subdir"] / f"{motion}.pkl"
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
            robot=spec["robot"],
            label=spec["label"],
            color=spec["color"],
            forward=spec.get("forward", DEFAULT_FORWARD),
            prefix=f"{key}_",
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            n=len(root_pos),
            fps=int(data.get("fps", 30)),
            z_offset=0.0,   # set by compute_ground_offsets (unless --no_ground)
        ))
        print(f"[green]{spec['label']}[/green] ({spec['robot']}): {path.name} "
              f"({len(root_pos)} frames, {dof_pos.shape[1]} dof)")
    return sources


def strip_lights_and_floors(spec, keep_floor):
    """Delete every light and (unless keep_floor) every floor/plane geom from a
    model spec, recursing through all bodies. Lets the composed scene use one
    shared light + one shared floor for identical lighting across robots."""
    def visit(body):
        for light in list(body.lights):
            spec.delete(light)
        for geom in list(body.geoms):
            if geom.type == mj.mjtGeom.mjGEOM_PLANE or geom.name == "floor":
                if not keep_floor:
                    spec.delete(geom)
        for child in list(body.bodies):
            visit(child)
    visit(spec.worldbody)


def build_scene_model(sources):
    """Compose one MuJoCo model containing all robots (each name-prefixed), with
    a single shared light + floor so lighting/framing match for every robot."""
    parent = mj.MjSpec()
    for i, src in enumerate(sources):
        child = mj.MjSpec.from_file(str(ROBOT_XML_DICT[src["robot"]]))
        # Keep exactly one robot's floor; strip everyone's lights (a shared one
        # is added below) so per-robot lighting (e.g. Go2's directional light)
        # can't make one character look different from the rest.
        strip_lights_and_floors(child, keep_floor=(i == 0))
        # Attach at the origin: the free joint fully overrides any frame offset,
        # so the lateral layout is applied to root_pos at runtime instead.
        frame = parent.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
        parent.attach(child, prefix=src["prefix"], frame=frame)
    # One shared light for the whole scene (matches the humanoids' default).
    parent.worldbody.add_light(pos=[-3.0, -3.0, 5.0], dir=[3.0, 3.0, -5.0],
                               diffuse=[0.5, 0.5, 0.5], castshadow=True)
    return parent.compile()


def make_floor_opaque(model):
    """Force the shared floor plane to alpha=1 and return how many planes changed.

    The robot XMLs ship a semi-transparent floor (g1: rgba alpha 0.5). MuJoCo
    draws transparent geoms in a separate pass that receives no shadow map, so
    the light's castshadow=True has no visible effect until the floor is opaque
    -- the robots were casting shadows all along, onto a surface that could not
    show them.
    """
    n = 0
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE and \
                model.geom_rgba[gid, 3] < 1.0:
            model.geom_rgba[gid, 3] = 1.0
            n += 1
    return n


def tint_robot(model, src):
    """Recolor a robot's visible geoms with its per-robot color."""
    rgb = np.asarray(src["color"][:3], dtype=np.float32)
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            continue
        bid = model.geom_bodyid[gid]
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(src["prefix"]):
            model.geom_rgba[gid, :3] = rgb  # keep original alpha


def _geom_lowest_z(model, data, gid):
    """World-space lowest z of a geom's axis-aligned local bounding box."""
    c = model.geom_aabb[gid, :3]
    h = model.geom_aabb[gid, 3:]
    xp = data.geom_xpos[gid]
    Rm = data.geom_xmat[gid].reshape(3, 3)
    zmin = np.inf
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = c + np.array([sx * h[0], sy * h[1], sz * h[2]])
                zmin = min(zmin, float((xp + Rm @ corner)[2]))
    return zmin


def compute_ground_offsets(model, data, sources, sample=200):
    """Per-robot z-lift so each robot's lowest visual-mesh point over the WHOLE
    clip rests on the floor (z=0), written into src['z_offset'].

    Different robots' COLMO pkls ground-align inconsistently (e.g. H1 sits ~8-10
    cm below the floor for its whole clip while G1 is nearly flush), which reads
    as "H1's feet are buried". A single constant lift per robot removes that
    static penetration/float while preserving the robot's relative vertical
    motion (jumps, crouches stay intact). The lane offsets don't affect foot z,
    so this pre-pass can pose every robot at the origin.
    """
    mesh_gids = {
        src["key"]: [
            g for g in range(model.ngeom)
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
            and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
                 or "").startswith(src["prefix"])]
        for src in sources}
    nfr = min(src["n"] for src in sources)
    step = max(1, nfr // sample)
    mins = {src["key"]: np.inf for src in sources}
    for t in range(0, nfr, step):
        for src in sources:
            adr, ndof = src["qadr"], src["ndof"]
            tt = min(t, src["n"] - 1)
            data.qpos[adr:adr + 3] = src["root_pos"][tt]
            data.qpos[adr + 3:adr + 7] = src["root_rot"][tt]
            data.qpos[adr + 7:adr + 7 + ndof] = src["dof_pos"][tt]
        mj.mj_forward(model, data)
        for src in sources:
            lo = min(_geom_lowest_z(model, data, g)
                     for g in mesh_gids[src["key"]])
            mins[src["key"]] = min(mins[src["key"]], lo)
    for src in sources:
        src["z_offset"] = -mins[src["key"]] if np.isfinite(mins[src["key"]]) else 0.0
        print(f"[cyan]{src['label']}: ground lift {src['z_offset']:+.3f} m"
              f"[/cyan]")


def hide_collision_geoms(model):
    """Make every collision proxy invisible, leaving only the visual mesh shell.

    In a composed multi-robot scene the geom GROUP numbers mean different things
    per robot (G1 puts collision in group 2, others in group 3, and H1's VISUAL
    mesh sits in group 2), so a group-based ``geomgroup`` toggle would hide one
    robot's mesh while showing another's capsules. Classify by geom TYPE instead
    (mesh = visual shell, primitive capsule/box/sphere = collision proxy) and
    zero the alpha of the primitives, which is per-geom and robot-agnostic.
    """
    for gid in range(model.ngeom):
        gtype = model.geom_type[gid]
        if gtype in (mj.mjtGeom.mjGEOM_PLANE, mj.mjtGeom.mjGEOM_MESH):
            continue
        model.geom_rgba[gid, 3] = 0.0


def null_geom_names(model):
    """Redirect every geom name to a null byte so MuJoCo draws no geom labels;
    only the head-marker labels attached to user_scn geoms should show."""
    names_arr = np.frombuffer(model.names, dtype=np.uint8)
    null_pos = int(np.where(names_arr == 0)[0][0]) if (names_arr == 0).any() else 0
    for gid in range(model.ngeom):
        model.name_geomadr[gid] = null_pos


def parse_frame_spec(text):
    """argparse type for one --snapshot_frames token: ``N``, ``A-B`` or ``A-B:STEP``.

    Returns the frame indices the token stands for, with ``A-B`` INCLUSIVE of B, so
    ``--snapshot_frames 0 60 100-200:10`` means frame 0, frame 60, and every 10th
    frame from 100 through 200. (Same spec as vis_compare_retargeting.)
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare one COLMO motion across several robots "
                    "(G1 / KAPEX / H1 / T1 / Go2) + the BVH human, one scene.")
    parser.add_argument("--motion", type=str, default="aiming1_subject1",
                        help="Motion name shared by the result pkls and the BVH "
                             "file, e.g. aiming1_subject1. 'all' batches over "
                             "every .bvh in --motion_dir.")
    parser.add_argument("--robots", nargs="+", default=DEFAULT_ORDER,
                        help=f"Robots to show, left-to-right. Choices: "
                             f"{list(ROBOT_TABLE.keys())}. "
                             f"Default: {DEFAULT_ORDER}.")

    # Data locations.
    parser.add_argument("--results_dir", type=str,
                        default=str(REPO_ROOT / "results"))
    parser.add_argument("--motion_dir", type=str, default=str(DEFAULT_MOTION_DIR))
    parser.add_argument("--bvh_file", type=str, default=None,
                        help="Override BVH path (default: "
                             "<motion_dir>/<motion>.bvh).")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--human_ref_robot", type=str, default="unitree_g1",
                        choices=list(ROBOT_XML_DICT.keys()),
                        help="Which robot's bvh_<format>_to_<robot>.json supplies "
                             "the human_scale_table used to size the drawn human "
                             "skeleton (default: unitree_g1).")
    parser.add_argument("--no_human", action="store_true",
                        help="Do not draw the BVH human skeleton.")
    parser.add_argument("--keypoints", nargs="+", default=None, metavar="BONE",
                        help="Draw the human skeleton markers ONLY for these bone "
                             "names. A bone (capsule) is drawn only when BOTH "
                             "endpoints are in the list. Default: full skeleton.")
    parser.add_argument("--keypoints_from_config", action="store_true",
                        help="Draw markers only for the retarget keybodies (the "
                             "human_scale_table keys of --human_ref_robot's IK "
                             "config). Takes precedence over --keypoints.")

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
    parser.add_argument("--tint", action="store_true",
                        help="Recolor each robot with its per-robot color "
                             "(default: keep the robots' natural colors).")
    parser.add_argument("--no_labels", action="store_true",
                        help="Hide the floating robot labels (the head marker "
                             "sphere carries the text, so this hides both).")
    parser.add_argument("--shadow", action="store_true",
                        help="Cast ground shadows under every character. The robot "
                             "XMLs ship a semi-transparent floor, and MuJoCo draws "
                             "transparent geoms in a pass that receives no shadow "
                             "map, so shadows never appear until it is made opaque "
                             "-- this does that. Recommended for paper figures: the "
                             "contact shadow is what reads as 'standing on the "
                             "floor' rather than floating.")
    parser.add_argument("--shadow_size", type=int, default=8192,
                        help="Shadow map resolution used with --shadow. Higher = "
                             "crisper shadow edges. Try 4096 / 8192 / 16384.")
    parser.add_argument("--shadow_skew", type=float, default=0.45,
                        help="How far --shadow throws the shadows to the side, as "
                             "a fraction of the straight-back direction. 0 casts "
                             "them straight away from the camera (mostly hidden "
                             "behind each character); negative throws them the "
                             "other way.")
    parser.add_argument("--no_ground", action="store_true",
                        help="Do not ground-align the robots. By default each "
                             "robot is lifted by a constant so its lowest foot "
                             "point over the clip rests on the floor (fixes e.g. "
                             "H1 sitting ~8-10 cm below the floor).")

    # Facing.
    parser.add_argument("--no_face_forward", action="store_true",
                        help="Keep each character's original world heading "
                             "instead of rotating them to face the camera.")
    parser.add_argument("--face_yaw", type=float, default=None,
                        help="World yaw (deg) every character's front should point "
                             "at, at frame 0 (default: face the camera). Add 180 "
                             "if characters end up facing away.")
    parser.add_argument("--flip", nargs="*", default=[], metavar="ROBOT",
                        help="Robots (by --robots key) whose frame-0 heading is "
                             "turned an extra 180 deg during facing alignment "
                             "(default: none; the per-robot forward axes in "
                             "ROBOT_TABLE already orient every robot correctly). "
                             "Use only as a manual override.")
    parser.add_argument("--mirror", nargs="*", default=[], metavar="ROBOT",
                        help="Robots (by --robots key) shown left-right mirrored "
                             "before facing alignment, matching how OmniRetarget "
                             "is mirrored in vis_compare_retargeting (default: "
                             "none). Pass e.g. '--mirror go2' to enable.")

    # Camera (one shared view of the whole row).
    parser.add_argument("--cam_distance", type=float, default=None,
                        help="Camera distance (m). Default: auto-fit the whole "
                             "row = max(4.0, (ncol-1)*spacing + 1.0).")
    parser.add_argument("--cam_azimuth", type=float, default=None)
    parser.add_argument("--cam_elevation", type=float, default=-15.0)
    parser.add_argument("--no_follow_camera", action="store_true")
    parser.add_argument("--lookat_shift", type=float, default=0.0,
                        help="Slide the camera reference (lookat) point sideways "
                             "in the image plane: positive = left, negative = "
                             "right (meters). Default 0 keeps the symmetric row "
                             "centered.")

    # Video recording / snapshots. Both go through the same offscreen renderer, so
    # they share --video_width/--video_height/--msaa/--font_scale.
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output video path. Default: "
                             "videos/compare_robots_<motion>.mp4.")
    parser.add_argument("--snapshot_frames", type=parse_frame_spec, nargs="+",
                        default=None, metavar="SPEC",
                        help="Frames to save as lossless PNG stills into "
                             "--snapshot_dir. Each token is a single frame (60), an "
                             "inclusive range (0-500), or a strided range (0-500:10); "
                             "mix them freely ('0 60 100-200:10'). Without "
                             "--record_video the run stops after the last requested "
                             "frame instead of playing the whole clip. In batch mode "
                             "(--motion all) these frames are captured for EVERY "
                             "motion.")
    parser.add_argument("--snapshot_all", action="store_true",
                        help="Save EVERY rendered frame as a PNG still into "
                             "--snapshot_dir.")
    parser.add_argument("--snapshot_dir", type=str, default="figures",
                        help="Where --snapshot_frames / --snapshot_all PNGs are "
                             "written (as <motion>_<robots>_<frame>.png).")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--video_quality", type=int, default=8,
                        help="imageio/ffmpeg quality (0-10; higher = less "
                             "compression, bigger file).")
    parser.add_argument("--msaa", type=int, default=8,
                        help="Anti-aliasing samples (MuJoCo offsamples). 8 or 16.")
    parser.add_argument("--font_scale", type=int, default=150,
                        choices=[50, 100, 150, 200, 250, 300],
                        help="Label text size in the recorded video (percent).")

    args = parser.parse_args()

    # Each --snapshot_frames token parsed to a list of indices (ranges expand);
    # flatten them into one sorted, de-duplicated frame list.
    if args.snapshot_frames:
        args.snapshot_frames = sorted(
            {i for spec in args.snapshot_frames for i in spec})

    snapshots = bool(args.snapshot_frames) or args.snapshot_all
    if snapshots:
        os.makedirs(args.snapshot_dir, exist_ok=True)
        count = "every frame" if args.snapshot_all \
            else f"{len(args.snapshot_frames)} frame(s)"
        print(f"[cyan]Snapshots ({count}) -> {args.snapshot_dir}/[/cyan]")

    if args.video_path is None:
        name = "all" if args.motion == "all" else args.motion
        args.video_path = f"videos/compare_robots_{name}.mp4"

    # Resolve which motions to compare.
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

    # One video for the whole run (motions appended back-to-back).
    mp4_writer = None
    if args.record_video:
        import imageio
        video_dir = os.path.dirname(args.video_path)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir)
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
            run_comparison(args, motion, mp4_writer=mp4_writer,
                           loop=args.loop and not batch,
                           bvh_override=None if batch else args.bvh_file)
    finally:
        if mp4_writer is not None:
            mp4_writer.close()
            print(f"[cyan]Video saved to {args.video_path}[/cyan]")


def run_comparison(args, motion, mp4_writer, loop, bvh_override):
    """Render one motion's human + multi-robot comparison (viewer and/or video)."""
    cam_azimuth = args.cam_azimuth if args.cam_azimuth is not None \
        else (180.0 if args.axis == "y" else 90.0)
    face_forward = not args.no_face_forward
    target_yaw = np.deg2rad(args.face_yaw if args.face_yaw is not None
                            else cam_azimuth + 180.0)

    _az = np.deg2rad(cam_azimuth)
    _fwd = np.array([-np.cos(_az), -np.sin(_az), 0.0])
    lookat_offset = np.cross(_fwd, [0.0, 0.0, 1.0]) * args.lookat_shift

    # --- Load robot motions -------------------------------------------------
    sources = resolve_robot_sources(args, motion)
    if not sources:
        print(f"[yellow]{motion}: no robot motions found for "
              f"{args.robots}, skipping.[/yellow]")
        return

    # Left-right mirror the requested robots (before facing alignment, so the
    # mirrored motion is then re-oriented to face the camera like the rest).
    # Each robot gets its own mirror spec built from its own model.
    if args.mirror:
        for src in sources:
            if src["key"] not in args.mirror:
                continue
            perm, msign = build_mirror_spec(ROBOT_XML_DICT[src["robot"]])
            if src["dof_pos"].shape[1] != len(perm):
                print(f"[yellow]{src['label']}: {src['dof_pos'].shape[1]} dof "
                      f"!= robot's {len(perm)}, skipping mirror.[/yellow]")
                continue
            src["root_pos"], src["root_rot"] = mirror_root(
                src["root_pos"], src["root_rot"])
            src["dof_pos"] = src["dof_pos"][:, perm] * msign[None, :]
            print(f"[magenta]{src['label']}: left-right mirrored.[/magenta]")

    # Turn every robot to face the target heading at frame 0, each using its own
    # forward axis (go2's base frame differs from the humanoids', see
    # ROBOT_TABLE). --flip adds an extra 180 deg for any robot that still ends up
    # facing away (empty by default now that the forward axes are correct).
    if face_forward:
        for src in sources:
            yaw = target_yaw + (np.pi if src["key"] in args.flip else 0.0)
            src["root_pos"], src["root_rot"] = yaw_align_root(
                src["root_pos"], src["root_rot"], yaw,
                forward_axis=src["forward"])
            if src["key"] in args.flip:
                print(f"[magenta]{src['label']}: heading flipped 180 deg."
                      f"[/magenta]")

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

            # Per-bone scale from the reference robot's IK config, rescaled by the
            # actual human height (matches vis_colmo_with_bvh).
            with open(IK_CONFIG_DICT[f"bvh_{args.format}"][args.human_ref_robot],
                      encoding="utf-8") as f:
                ik_config = json.load(f)
            ratio = height / ik_config["human_height_assumption"]
            base_scale = {k: np.asarray(v, dtype=float) * ratio
                          for k, v in ik_config["human_scale_table"].items()}
            h_root = ik_config["human_root_name"]
            eff_scale = build_effective_scale(bones, parents, base_scale)

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
                         pivot=pivot, Rz=Rz_human,
                         scale_bones=list(ik_config["human_scale_table"].keys()))
            print(f"[red]BVH human[/red]: {bvh_path.name} "
                  f"({len(frames)} frames, height {height:.2f} m)")
        else:
            print(f"[yellow]BVH {bvh_path} not found, human skeleton disabled."
                  f"[/yellow]")

    # Optional keypoint filter.
    keypoint_set = None
    if args.keypoints_from_config:
        if human is not None:
            keypoint_set = set(human["scale_bones"])
            print(f"[cyan]keypoints from IK config human_scale_table: "
                  f"{sorted(keypoint_set)}[/cyan]")
        else:
            print("[yellow]--keypoints_from_config ignored (no human skeleton).[/yellow]")
    elif args.keypoints:
        keypoint_set = set(args.keypoints)
    if keypoint_set is not None and human is not None:
        valid = set(human["frames"][0].keys())
        unknown = keypoint_set - valid
        if unknown:
            print(f"[yellow]keypoints: unknown bone(s) {sorted(unknown)}. "
                  f"Valid: {sorted(valid)}[/yellow]")

    keypoint_edges = None
    if keypoint_set is not None and human is not None:
        keypoint_edges = build_keypoint_edges(
            human["bones"], human["parents"], keypoint_set)

    # --- Column layout ------------------------------------------------------
    columns = (["human"] if human else []) + [s["key"] for s in sources]
    ncol = len(columns)
    axis_idx = 0 if args.axis == "x" else 1

    def column_offset(col_index):
        off = np.zeros(3)
        off[axis_idx] = (col_index - (ncol - 1) / 2.0) * args.spacing
        return off

    def horizontal_shift(offset, root0_xy, cur_xy):
        if args.root_mode == "lock":
            return np.array([offset[0] - cur_xy[0], offset[1] - cur_xy[1], 0.0])
        if args.root_mode == "recenter":
            return np.array([offset[0] - root0_xy[0], offset[1] - root0_xy[1], 0.0])
        return np.array([offset[0], offset[1], 0.0])

    if human:
        human["offset"] = column_offset(0)
        human["root0_xy"] = human["pivot"][:2]
    for k, src in enumerate(sources):
        src["offset"] = column_offset((1 if human else 0) + k)
        src["root0_xy"] = src["root_pos"][0, :2]

    # --- Compose and prepare the MuJoCo model ------------------------------
    model = build_scene_model(sources)
    if args.shadow:
        make_floor_opaque(model)
        # Crisper contact shadows at print resolution (default is 4096).
        model.vis.quality.shadowsize = max(int(model.vis.quality.shadowsize),
                                           args.shadow_size)
        # Put the light on the CAMERA's side of the row so shadows fall away from
        # the viewer (front-to-back). The shared light sits opposite the camera by
        # default, which throws every shadow forward, toward the reader.
        _az = np.deg2rad(cam_azimuth)
        view = np.array([np.cos(_az), np.sin(_az), 0.0])   # camera -> scene
        side = np.array([-np.sin(_az), np.cos(_az), 0.0])  # image-right in world
        light_dir = view + args.shadow_skew * side + np.array([0.0, 0.0, -1.1])
        light_dir /= np.linalg.norm(light_dir)
        model.light_dir[0] = light_dir
        row_mid = 0.5 * (column_offset(0) + column_offset(ncol - 1))
        model.light_pos[0] = row_mid - light_dir * 7.0
    data = mj.MjData(model)

    # Body to hang each robot's label on, with z-clearance to float above the
    # head. Robots differ, so fall back to the base body with generous clearance.
    head_anchors = (("head_mocap", 0.15), ("head_link", 0.35),
                    ("head_pitch_link", 0.35), ("Head", 0.35),
                    ("torso_link", 0.65), ("Trunk", 0.5))
    for src in sources:
        robot_base = ROBOT_BASE_DICT[src["robot"]]
        # Match the free joint by its BODY prefix: several robots (go2, t1, h1)
        # leave the base free joint unnamed, so a joint-name check would miss it.
        free_jid = next(
            jid for jid in range(model.njnt)
            if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE
            and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY,
                               model.jnt_bodyid[jid]) or "")
            .startswith(src["prefix"]))
        src["qadr"] = int(model.jnt_qposadr[free_jid])
        src["ndof"] = src["dof_pos"].shape[1]
        model_ndof = sum(
            1 for jid in range(model.njnt)
            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or "")
            .startswith(src["prefix"])
            and model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE)
        if src["ndof"] != model_ndof:
            raise SystemError(
                f"{src['label']}: pkl has {src['ndof']} dof but robot "
                f"'{src['robot']}' expects {model_ndof}. The result pkl must "
                f"match the robot model.")
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
        if args.tint:
            tint_robot(model, src)

    # Ground-align each robot (constant z-lift) unless disabled. Done after the
    # qadr/ndof setup above (the pre-pass needs them) and before hiding geoms.
    if not args.no_ground:
        compute_ground_offsets(model, data, sources)

    hide_collision_geoms(model)
    null_geom_names(model)

    total_frames = max([s["n"] for s in sources]
                       + ([human["n"]] if human else []))
    motion_fps = args.motion_fps or sources[0]["fps"]

    # Horizontal center of the whole row: the midpoint between the first and last
    # column lanes (0 by construction, since column_offset centers on the origin).
    # Framing on this — not on the robot-only centroid — keeps the end columns
    # (the human on the far left, Go2 on the far right) inside the frame.
    row_center = 0.5 * (column_offset(0) + column_offset(ncol - 1))

    # --- Viewer setup -------------------------------------------------------
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

    if need_render:
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
    # Collision proxies were already made invisible per-geom (hide_collision_geoms),
    # so no group toggle here — a group toggle would misfire across robots.
    opt.label = mj.mjtLabel.mjLABEL_NONE

    # Distance: pull back far enough to frame the whole row (all ncol columns),
    # not just the robots. --cam_distance overrides the auto fit.
    row_span = (ncol - 1) * args.spacing
    if args.cam_distance is not None:
        cam.distance = args.cam_distance
    else:
        cam.distance = max(5.0, row_span + 1.0)
    cam.azimuth = cam_azimuth
    cam.elevation = args.cam_elevation
    cam.lookat[:] = row_center + lookat_offset

    # --- Offscreen renderer (video + snapshots) -----------------------------
    # The mp4 writer is owned by the caller (shared across motions); here we only
    # build this motion's offscreen renderer. Snapshots come out of the same
    # renderer, so they share the resolution and AA settings.
    renderer = None
    if need_render:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth),
                                         args.video_width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight),
                                          args.video_height)
        model.vis.quality.offsamples = max(int(model.vis.quality.offsamples),
                                           args.msaa)
        renderer = mj.Renderer(model, height=args.video_height,
                               width=args.video_width,
                               font_scale=mj.mjtFontScale(args.font_scale))

    def draw_overlays(scene):
        """Draw the human skeleton and floating robot labels into a scene."""
        if human:
            frames = human["frames"]
            frame = frames[min(cur_i, human["n"] - 1)]
            h_root = human["h_root"]
            eff_scale = human["eff_scale"]
            pivot, Rz = human["pivot"], human["Rz"]
            raw_root = np.asarray(frame[h_root][0], dtype=np.float64)
            scaled_root = eff_scale.get(h_root, 1.0) * raw_root

            def rp(bone):
                local = (np.asarray(frame[bone][0]) - raw_root) \
                    * eff_scale.get(bone, 1.0)
                return pivot + Rz.apply((scaled_root - pivot) + local)

            hips_xy = rp(h_root)[:2]
            shift = horizontal_shift(human["offset"], human["root0_xy"], hips_xy)

            def wp(bone):
                return rp(bone) + shift

            marker_bones = human["bones"] if keypoint_set is None else keypoint_set
            for bone in marker_bones:
                if bone not in frame:
                    continue
                if bone == h_root:
                    add_sphere(scene, wp(bone), 0.045, HUMAN_ROOT_COLOR)
                else:
                    add_sphere(scene, wp(bone), 0.030, HUMAN_JOINT_COLOR)
            draw_edges = human["edges"] if keypoint_set is None else keypoint_edges
            for child, parent in draw_edges:
                if child not in frame or parent not in frame:
                    continue
                add_capsule(scene, wp(parent), wp(child), 0.012, HUMAN_BONE_COLOR)
            if not args.no_labels and "Spine2" in frame:
                # Label the human column with the motion name (like the g1
                # comparison video), floated well above the skeleton's head.
                add_sphere(scene, wp("Spine2") + np.array([0, 0, 0.45]),
                           0.05, HUMAN_LABEL_COLOR, label=f"{motion}")

        if not args.no_labels:
            for src in sources:
                head_pos = data.xpos[src["head_bid"]] + \
                    np.array([0, 0, src["label_z"]])
                add_sphere(scene, head_pos, 0.05, src["color"], label=src["label"])

    # PNG stills. Named per motion + robot row so a batch run (and repeated runs
    # with a different --robots) never overwrite each other's figures.
    snap_count = 0
    # Naming each PNG as it lands is useful for a handful of figures and pure noise
    # for a whole range, so past a few frames only the closing tally is printed.
    verbose_snaps = not args.snapshot_all and 0 < len(snapshot_set) <= 12
    if snapshot_set or args.snapshot_all:
        import imageio
        snap_name = f"{motion}_{'-'.join(s['key'] for s in sources)}"
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
                       else min(total_frames, snapshot_stop + 1)),
                desc="compare-robots")
    rate_limiter = RateLimiter(frequency=motion_fps, warn=False)
    cur_i = 0

    try:
        while (viewer.is_running() if viewer is not None else True):
            if not paused[0]:
                for src in sources:
                    t = min(cur_i, src["n"] - 1)
                    adr, ndof = src["qadr"], src["ndof"]
                    root = src["root_pos"][t]
                    shift = horizontal_shift(src["offset"], src["root0_xy"],
                                             root[:2])
                    shift[2] = src["z_offset"]   # ground-align lift (z)
                    data.qpos[adr:adr + 3] = root + shift
                    data.qpos[adr + 3:adr + 7] = src["root_rot"][t]
                    data.qpos[adr + 7:adr + 7 + ndof] = src["dof_pos"][t]
                mj.mj_forward(model, data)

            if not args.no_follow_camera:
                # Keep the row horizontally centered (so no end column drifts out
                # of frame); follow only the vertical motion of the robot bases.
                z = float(np.mean([data.xpos[s["base_bid"]][2] for s in sources]))
                cam.lookat[:] = np.array(
                    [row_center[0], row_center[1], z]) + lookat_offset

            # Overlays for the live viewer (the offscreen recorder re-draws them
            # into its own render scene below).
            if viewer is not None:
                viewer.user_scn.ngeom = 0
                draw_overlays(viewer.user_scn)
                viewer.sync()

            # One render feeds both the video and the PNG stills (overlays must be
            # re-drawn into the render scene). Skipped while paused so the
            # recording is not padded with dupes.
            if renderer is not None and not paused[0]:
                renderer.update_scene(data, camera=cam, scene_option=opt)
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
