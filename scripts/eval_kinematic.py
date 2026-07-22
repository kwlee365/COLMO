"""Kinematic evaluation of retargeted robot motions.

Scores the reference motions produced by several retargeting sources for one robot
against seven mesh-level kinematic quality metrics (all LOWER = better):

  1. Ground penetration frame fraction   P_ground
  2. Mean ground penetration depth        D_ground
  3. Self penetration frame fraction      P_self
  4. Mean self penetration depth          D_self
  5. Mean foot sliding velocity           V_slide
  6. Joint velocity violation frame frac. P_vel
  7. Shoulder roll saturation fraction    P_sat

Key modelling choices (documented so the numbers are reproducible):

* REAL ROBOT MESH, not COLMO's simplified capsules. The evaluator builds a
  dedicated collision model in which ONLY the robot's real link meshes (the
  visual group-1 meshes covering every link incl. feet/hands) collide; the
  coarse capsule/sphere collision primitives and the floor are disabled. This is
  identical for every source, so comparisons are fair.

* GROUND penetration uses the exact lowest real-mesh VERTEX below the floor
  (z=0), avoiding convex-hull inflation. SELF penetration uses MuJoCo's convex
  mesh-mesh contacts (penetration depth = -contact.dist).

* STRUCTURAL self-overlaps: some non-adjacent link meshes overlap even at the
  neutral pose (e.g. knee<->ankle_roll, waist_yaw<->torso) because MuJoCo
  collides the meshes' convex hulls. Those body pairs are modelling artifacts,
  not real self-collisions, so they are auto-detected at the neutral pose and
  excluded from the self-penetration metric for ALL frames and ALL sources.
  MuJoCo's parent-child filtering (filterparent) is left ON.

* Foot CONTACT (stance) is detected from the shared SOURCE human motion (BVH
  LeftToe/RightToe): a foot is in contact at frame t when its toe's horizontal
  displacement since t-1 is <= contact_thresh (default 0.01 m) at 30 fps. Robot
  foot sliding is then the horizontal speed of the robot's toe link over the
  contact foot-frames.

* Joint velocity limits come from the robot URDF (<limit velocity=...>); the
  per-frame joint velocity is 30*(q_t - q_{t-1}) with a shortest-angle wrap.

Usage:
    python scripts/eval_kinematic.py --robot unitree_g1 \
        --results_dir results/g1/lafan1 --motion_dir human_motion/lafan1 \
        --algos gmr omniretarget colmo --out results/g1/lafan1/kinematic_eval.csv

By default only motions present for EVERY requested source are scored (a fair
paired comparison over an identical motion set).
"""

import argparse
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco as mj
from tqdm import tqdm

from collision_free_motion_retargeting.params import ROBOT_XML_DICT
from collision_free_motion_retargeting.utils.lafan1 import load_bvh_file

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-source pkl location and filename suffix (mirrors vis_compare_retargeting).
ALGO_TABLE = {
    "gmr":          dict(subdir="gmr",          suffix="",          label="GMR"),
    "omniretarget": dict(subdir="omniretarget", suffix="_original", label="OmniRetarget"),
    "colmo":        dict(subdir="colmo",        suffix="",          label="COLMO"),
    "unitree":      dict(subdir="unitree",      suffix="",          label="Unitree LAFAN1"),
}


# --------------------------------------------------------------------------- #
# Robust pickle loading across numpy 1.x / 2.x (same shim as vis_compare).
# --------------------------------------------------------------------------- #
class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                return super().find_class("numpy.core" + module[len("numpy._core"):], name)
            if module.startswith("numpy.core"):
                return super().find_class("numpy._core" + module[len("numpy.core"):], name)
            raise


def load_pkl(path):
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


# --------------------------------------------------------------------------- #
# Robot model / metadata.
# --------------------------------------------------------------------------- #
def build_collision_model(robot):
    """Compile a model where ONLY the real link meshes (visual group 1) collide.

    Returns (model, mesh_geom_ids): the group-1 mesh geom ids are also the geoms
    whose vertices define the ground-penetration lowest point.
    """
    spec = mj.MjSpec.from_file(str(ROBOT_XML_DICT[robot].resolve()))
    for g in spec.geoms:
        real_mesh = (g.type == mj.mjtGeom.mjGEOM_MESH and g.group == 1)
        if real_mesh:
            g.contype, g.conaffinity = 1, 1
        else:                                   # capsules/spheres/floor/other -> off
            g.contype, g.conaffinity = 0, 0
    model = spec.compile()
    mesh_geom_ids = [g for g in range(model.ngeom)
                     if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
                     and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return model, mesh_geom_ids


def urdf_velocity_limits(robot):
    """joint_name -> max |qdot| [rad/s] from the robot URDF <limit velocity=...>."""
    urdf = next((ROBOT_XML_DICT[robot].resolve().parent).glob("*.urdf"), None)
    if urdf is None:
        raise FileNotFoundError(f"No URDF next to {ROBOT_XML_DICT[robot]}")
    root = ET.parse(str(urdf)).getroot()
    vlim = {}
    for j in root.iter("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("velocity") is not None:
            vlim[j.get("name")] = float(lim.get("velocity"))
    return vlim, urdf.name


def robot_metadata(robot):
    """Collect everything the metrics need from the model + URDF."""
    model, mesh_geom_ids = build_collision_model(robot)
    vlim_map, urdf_name = urdf_velocity_limits(robot)

    # Actuated joints in qpos/dof order (skip the free base).
    dof_names, dof_qadr = [], []
    free_qadr = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            free_qadr = int(model.jnt_qposadr[j])
            continue
        dof_names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j))
        dof_qadr.append(int(model.jnt_qposadr[j]))
    n_dof = len(dof_names)
    if free_qadr != 0 or dof_qadr != list(range(7, 7 + n_dof)):
        raise SystemError("Unexpected qpos layout (expected [free(7), dof...]).")

    vmax = np.array([vlim_map[n] for n in dof_names], dtype=np.float64)

    # Joint ranges (for shoulder saturation) and the shoulder-roll dof indices.
    jnt_range = np.array([model.jnt_range[model.joint(n).id] for n in dof_names])
    shoulder_roll_idx = [i for i, n in enumerate(dof_names)
                         if "shoulder_roll" in n]

    # Toe bodies for foot sliding (fall back to ankle_roll if no toe link).
    def body_id(cands):
        for c in cands:
            try:
                return model.body(c).id
            except KeyError:
                continue
        return None
    toe_bid = {
        "L": body_id(["left_toe_link", "left_ankle_roll_link"]),
        "R": body_id(["right_toe_link", "right_ankle_roll_link"]),
    }

    # Per-mesh-geom local vertices (for exact ground lowest-point).
    geom_verts = {}
    for gid in mesh_geom_ids:
        mid = int(model.geom_dataid[gid])
        adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        geom_verts[gid] = model.mesh_vert[adr:adr + num].astype(np.float64)

    # Floor height (plane z) in the ORIGINAL model, so ground pen is measured
    # against the real floor even though the collision model disables it.
    m_full = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot].resolve()))
    planes = [g for g in range(m_full.ngeom)
              if m_full.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE]
    floor_z = float(m_full.geom_pos[planes[0], 2]) if planes else 0.0

    # Per-foot real-mesh geoms (left/right ankle_roll link plate) for the foot-floating metric.
    foot_geoms = {"L": [], "R": []}
    for gid in mesh_geom_ids:
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if "left_ankle_roll" in bname:
            foot_geoms["L"].append(gid)
        elif "right_ankle_roll" in bname:
            foot_geoms["R"].append(gid)

    # Hand ground-penetration override: the detailed rubber-hand finger MESH pokes through
    # the floor in floor / get-up motions and dominates the ground-penetration metric, so
    # for the HANDS ground penetration uses the coarse collision PRIMITIVE (the Hand1 sphere)
    # instead of the finger mesh. Only override a hand that actually has such a primitive;
    # otherwise fall back to its mesh (never drop coverage). Spheres are kept (collision
    # disabled) in the compiled model, so mj_kinematics still gives their world pose.
    hand_mesh_skip = set()                     # mesh geom ids to EXCLUDE from ground pen.
    hand_pen_geoms = []                        # (geom_id, radius) collision spheres to USE
    for hb in range(model.nbody):
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, hb) or ""
        if "hand" not in bname.lower():
            continue
        spheres = [(g, float(model.geom_size[g][0])) for g in range(model.ngeom)
                   if int(model.geom_bodyid[g]) == hb
                   and model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE]
        if not spheres:
            continue                           # no collision primitive -> keep the mesh
        hand_pen_geoms.extend(spheres)
        hand_mesh_skip.update(gid for gid in mesh_geom_ids
                              if int(model.geom_bodyid[gid]) == hb)

    return dict(model=model, data=mj.MjData(model),
                mesh_geom_ids=mesh_geom_ids, geom_verts=geom_verts,
                dof_names=dof_names, n_dof=n_dof, vmax=vmax,
                jnt_range=jnt_range, shoulder_roll_idx=shoulder_roll_idx,
                toe_bid=toe_bid, foot_geoms=foot_geoms,
                hand_mesh_skip=hand_mesh_skip, hand_pen_geoms=hand_pen_geoms,
                floor_z=floor_z, urdf_name=urdf_name)


def structural_self_pairs(meta):
    """Body-pairs that overlap at the NEUTRAL pose (dof=0) -> modelling artifacts
    to exclude from the self-penetration metric. Returns a set of frozenset pairs."""
    model, data = meta["model"], meta["data"]
    q = np.zeros(model.nq)
    q[3] = 1.0            # identity quaternion (wxyz)
    q[2] = 1.0            # lift off the floor so only self-contacts appear
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    mj.mj_collision(model, data)
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if -c.dist <= 0:
            continue
        b1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
        b2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
        pairs.add(frozenset((b1, b2)))
    return pairs


def _ancestors(model, b):
    """Body ids on the path from body b up to (and including) the world root."""
    chain = set()
    while True:
        chain.add(b)
        p = int(model.body_parentid[b])
        if p == b:                       # world (id 0) is its own parent
            break
        b = p
    return chain


def same_chain_pairs(meta):
    """Body-name pairs where one body is an ANCESTOR of the other on the kinematic tree
    (i.e. on the same serial chain). A link overlapping the convex hull of its own
    ancestor/descendant is a structural false positive, so such pairs are excluded from the
    self-penetration metric (cf. ReactOR: "collisions within same kinematic chain ignored").
    This subsumes the parent-child cases filterparent already drops, plus the deeper
    same-limb overlaps (foot<->hip, upper-arm<->torso, thigh<->pelvis). Cross-chain
    collisions (hand<->hand, hand<->opposite leg, arm<->same-side leg) are KEPT."""
    model = meta["model"]
    anc = {b: _ancestors(model, b) for b in range(model.nbody)}
    pairs = set()
    for b1 in range(1, model.nbody):                       # skip world
        n1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
        for b2 in range(b1 + 1, model.nbody):
            if b1 in anc[b2] or b2 in anc[b1]:             # one is ancestor of the other
                n2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)
                pairs.add(frozenset((n1, n2)))
    return pairs


# --------------------------------------------------------------------------- #
# Per-frame geometry.
# --------------------------------------------------------------------------- #
def ground_penetration_depth(meta):
    """Depth (m, >=0) the lowest ground-collision point sits below the floor this frame.

    Uses each link's real mesh VERTICES, EXCEPT the hands: their detailed finger mesh is
    replaced by the coarse Hand1 collision SPHERE (see robot_metadata), so finger tips do not
    register as ground penetration."""
    model, data = meta["model"], meta["data"]
    min_z = np.inf
    skip = meta["hand_mesh_skip"]
    for gid, v_local in meta["geom_verts"].items():
        if gid in skip:                                  # hand mesh -> use its sphere below
            continue
        xpos = data.geom_xpos[gid]
        zrow = data.geom_xmat[gid].reshape(3, 3)[2]      # world-z row
        z = v_local @ zrow + xpos[2]                     # world z of each vertex
        mz = z.min()
        if mz < min_z:
            min_z = mz
    for gid, radius in meta["hand_pen_geoms"]:           # hands: lowest point of Hand1 sphere
        mz = float(data.geom_xpos[gid][2]) - radius
        if mz < min_z:
            min_z = mz
    return max(0.0, meta["floor_z"] - min_z)


def self_penetration_depth(meta, structural):
    """Max convex mesh-mesh penetration depth (m, >=0) among non-structural,
    non-parent-child link pairs this frame (0 if none)."""
    model, data = meta["model"], meta["data"]
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        depth = -c.dist
        if depth <= 0:
            continue
        b1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
        b2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
        if frozenset((b1, b2)) in structural:
            continue
        if depth > worst:
            worst = depth
    return worst


# --------------------------------------------------------------------------- #
# One motion, one source.
# --------------------------------------------------------------------------- #
def qpos_from(data_dict, meta, t):
    q = np.zeros(meta["model"].nq)
    q[:3] = data_dict["root_pos"][t]
    q[3:7] = np.asarray(data_dict["root_rot"][t], dtype=np.float64)[[3, 0, 1, 2]]  # xyzw->wxyz
    q[7:] = data_dict["dof_pos"][t]
    return q


def human_contact_labels(bvh_frames, contact_thresh, fps):
    """(N,) bool per foot: toe horizontal displacement since t-1 <= thresh (stance)."""
    N = len(bvh_frames)
    labels = {"L": np.zeros(N, bool), "R": np.zeros(N, bool)}
    bone = {"L": "LeftToe", "R": "RightToe"}
    for f in ("L", "R"):
        if bone[f] not in bvh_frames[0]:
            continue
        xy = np.array([np.asarray(bvh_frames[t][bone[f]][0], np.float64)[:2]
                       for t in range(N)])
        disp = np.linalg.norm(np.diff(xy, axis=0), axis=1)   # (N-1,)
        labels[f][1:] = disp <= contact_thresh
    return labels


def evaluate_motion(data_dict, bvh_frames, meta, structural, args, mirrored=False):
    model, data = meta["model"], meta["data"]
    N = len(data_dict["root_pos"])
    fps = int(data_dict.get("fps", 30))
    thr = args.pen_thresh                                  # self-penetration threshold
    gthr = args.ground_thresh if args.ground_thresh is not None else args.pen_thresh

    ground = np.zeros(N)
    self_pen = np.zeros(N)
    toe_xy = {"L": np.full((N, 2), np.nan), "R": np.full((N, 2), np.nan)}
    foot_gap = {"L": np.full(N, np.nan), "R": np.full(N, np.nan)}   # foot-mesh lowest z - floor

    for t in range(N):
        data.qpos[:] = qpos_from(data_dict, meta, t)
        mj.mj_kinematics(model, data)
        mj.mj_collision(model, data)
        ground[t] = ground_penetration_depth(meta)
        self_pen[t] = self_penetration_depth(meta, structural)
        for f in ("L", "R"):
            if meta["toe_bid"][f] is not None:
                toe_xy[f][t] = data.xpos[meta["toe_bid"][f]][:2]
            lo = np.inf
            for gid in meta["foot_geoms"][f]:
                zrow = data.geom_xmat[gid].reshape(3, 3)[2]           # world-z row
                lo = min(lo, float((meta["geom_verts"][gid] @ zrow
                                    + data.geom_xpos[gid][2]).min()))
            if np.isfinite(lo):
                foot_gap[f][t] = lo - meta["floor_z"]                  # >0 float, <0 penetrate

    # 1-2) ground penetration frame fraction + mean depth
    g_hit = ground > gthr
    P_ground = float(g_hit.mean())
    D_ground = float(ground[g_hit].mean()) if g_hit.any() else np.nan

    # 3-4) self penetration frame fraction + mean depth
    s_hit = self_pen > thr
    P_self = float(s_hit.mean())
    D_self = float(self_pen[s_hit].mean()) if s_hit.any() else np.nan

    # 5) foot sliding over source-contact foot-frames. For sources stored left-right
    # MIRRORED (e.g. OmniRetarget), the robot's LEFT foot executes the human's RIGHT
    # foot motion, so pair robot foot f with the human contact of the mirrored foot.
    labels = human_contact_labels(bvh_frames, args.contact_thresh, fps)
    Nc = min(N, len(bvh_frames))
    swap = {"L": "R", "R": "L"}
    slide_vals = []
    for f in ("L", "R"):
        hf = swap[f] if mirrored else f
        v = fps * np.linalg.norm(np.diff(toe_xy[f], axis=0), axis=1)   # (N-1,)
        c = labels[hf][1:Nc]                                           # align to v[0:Nc-1]
        vv = v[:Nc - 1][c]
        slide_vals.append(vv[~np.isnan(vv)])
    slide_all = np.concatenate(slide_vals) if slide_vals else np.array([])
    V_slide = float(slide_all.mean()) if slide_all.size else np.nan

    # 5b) foot floating: mean of the robot foot's closest distance to the ground over the
    # reference-contact frames of that foot (same mirror-aware pairing as foot sliding).
    # Positive = the foot hovers above the ground while the human foot is planted.
    float_vals = []
    for f in ("L", "R"):
        hf = swap[f] if mirrored else f
        c = labels[hf][:Nc]                                           # per-frame stance mask
        g = foot_gap[f][:Nc][c]
        float_vals.append(g[~np.isnan(g)])
    float_all = np.concatenate(float_vals) if float_vals else np.array([])
    Foot_float = float(float_all.mean()) if float_all.size else np.nan

    # 6) joint velocity violation frame fraction (shortest-angle diff, 30 fps)
    dof = np.asarray(data_dict["dof_pos"], np.float64)
    dq = np.diff(dof, axis=0)
    dq = (dq + np.pi) % (2 * np.pi) - np.pi
    qdot = fps * dq                                                    # (N-1, n_dof)
    viol = (np.abs(qdot) > meta["vmax"][None, :]).any(axis=1)
    P_vel = float(viol.mean()) if viol.size else np.nan

    # 7) shoulder roll saturation fraction (eta band of joint range, both sides)
    eta = args.eta
    sr = meta["shoulder_roll_idx"]
    sat_frac = np.nan
    if sr:
        lo = meta["jnt_range"][sr, 0]
        hi = meta["jnt_range"][sr, 1]
        delta = eta * (hi - lo)
        q_sr = dof[:, sr]                                             # (N, n_sr)
        sat = (q_sr <= lo + delta) | (q_sr >= hi - delta)
        sat_frac = float(sat.mean())                                 # averages over joints & frames

    return dict(N=N, P_ground=P_ground, D_ground=D_ground, Foot_float=Foot_float,
                P_self=P_self, D_self=D_self, V_slide=V_slide,
                P_vel=P_vel, P_sat=sat_frac)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def resolve_path(results_dir, key, motion):
    spec = ALGO_TABLE[key]
    return Path(results_dir) / spec["subdir"] / f"{motion}{spec['suffix']}.pkl"


def common_motions(results_dir, algos, motion_dir):
    """Motion basenames present for EVERY requested source (and with a BVH)."""
    sets = []
    for key in algos:
        spec = ALGO_TABLE[key]
        d = Path(results_dir) / spec["subdir"]
        names = set()
        for p in d.glob(f"*{spec['suffix']}.pkl" if spec["suffix"] else "*.pkl"):
            stem = p.stem
            if spec["suffix"] and stem.endswith(spec["suffix"]):
                stem = stem[:-len(spec["suffix"])]
            names.add(stem)
        sets.append(names)
    inter = set.intersection(*sets) if sets else set()
    bvh = {p.stem for p in Path(motion_dir).glob("*.bvh")}
    return sorted(inter & bvh)


def main():
    ap = argparse.ArgumentParser(description="Kinematic evaluation of retargeted motions.")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "g1" / "lafan1"))
    ap.add_argument("--motion_dir", default=str(REPO_ROOT / "human_motion" / "lafan1"))
    ap.add_argument("--algos", nargs="+",
                    default=["gmr", "omniretarget", "colmo"])
    ap.add_argument("--pen_thresh", type=float, default=0.01, help="Self-penetration threshold [m].")
    ap.add_argument("--ground_thresh", type=float, default=0.01,
                    help="Ground-penetration threshold [m] (default: --pen_thresh).")
    ap.add_argument("--contact_thresh", type=float, default=0.01,
                    help="Human toe stance displacement threshold [m/frame].")
    ap.add_argument("--eta", type=float, default=0.05, help="Shoulder saturation margin (range frac).")
    ap.add_argument("--mirror", nargs="*", default=["omniretarget"], metavar="ALGO",
                    help="Sources stored left-right mirrored vs the source human "
                         "(default: omniretarget); their foot-contact L/R pairing is "
                         "swapped so foot-sliding stays correct. Pass '--mirror' alone to disable.")
    ap.add_argument("--motions", nargs="+", default=None,
                    help="Explicit motion list (default: all present for every source).")
    ap.add_argument("--max_motions", type=int, default=None, help="Cap for a quick run.")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (>1 = faster, approximate).")
    ap.add_argument("--out", default=None, help="Per-motion CSV output path.")
    args = ap.parse_args()

    meta = robot_metadata(args.robot)
    structural = structural_self_pairs(meta)
    same_chain = same_chain_pairs(meta)                 # ReactOR: ignore same-chain collisions
    ignore_self = structural | same_chain               # combined self-penetration exclusion
    print(f"[eval] robot={args.robot}  dof={meta['n_dof']}  "
          f"vel-limits from {meta['urdf_name']}  floor_z={meta['floor_z']:.3f}")
    print(f"[eval] real-mesh collision geoms: {len(meta['mesh_geom_ids'])} | "
          f"structural self-pairs EXCLUDED: "
          f"{sorted('<->'.join(sorted(p)) for p in structural)}")
    print(f"[eval] same-kinematic-chain pairs also EXCLUDED (ancestor-descendant): "
          f"{len(same_chain)} pairs")

    motions = args.motions or common_motions(args.results_dir, args.algos, args.motion_dir)
    if args.max_motions:
        motions = motions[:args.max_motions]
    print(f"[eval] scoring {len(motions)} motion(s) present for {args.algos}\n")

    # BVH (source human) contact frames are shared by all sources of a motion.
    per_rows = []          # (motion, algo, metrics...)
    agg = {k: {m: [] for m in ("P_ground", "D_ground", "Foot_float", "P_self", "D_self",
                               "V_slide", "P_vel", "P_sat")} for k in args.algos}

    for motion in tqdm(motions, desc="motions"):
        bvh_path = Path(args.motion_dir) / f"{motion}.bvh"
        bvh_frames, _ = load_bvh_file(str(bvh_path), format="lafan1")
        if args.stride > 1:
            bvh_frames = bvh_frames[::args.stride]
        for key in args.algos:
            path = resolve_path(args.results_dir, key, motion)
            if not path.exists():
                continue
            d = load_pkl(path)
            if args.stride > 1:
                d = {**d, "root_pos": d["root_pos"][::args.stride],
                     "root_rot": np.asarray(d["root_rot"])[::args.stride],
                     "dof_pos": d["dof_pos"][::args.stride]}
            r = evaluate_motion(d, bvh_frames, meta, ignore_self, args,
                                mirrored=(key in args.mirror))
            per_rows.append((motion, key, r))
            for m in agg[key]:
                agg[key][m].append(r[m])

    # ---- Aggregate: report mean +/- std ACROSS motions (macro-level) --------
    def stats(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            return np.nan, np.nan
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0   # sample std
        return m, s

    def fmt_ms(vals, scale, nd):
        m, s = stats(vals)
        if np.isnan(m):
            return "n/a"
        return f"{m * scale:.{nd}f}±{s * scale:.{nd}f}"        # "mean±std"

    # metric key, row label (with unit), scale, decimals. Metrics-as-rows so the
    # mean±std cells stay readable (paper-style: ground/self penetration, foot
    # sliding, foot floating -- all reported as mean +/- std over motions).
    METRICS = [
        ("P_ground",   "Ground penetration %", 100, 2),
        ("D_ground",   "Ground pen depth cm",  100, 3),
        ("P_self",     "Self penetration %",   100, 2),
        ("D_self",     "Self pen depth cm",    100, 3),
        ("V_slide",    "Foot sliding cm/s",    100, 2),
        ("Foot_float", "Foot floating cm",     100, 3),
        ("P_vel",      "Vel violation %",      100, 2),
        ("P_sat",      "Shoulder sat %",       100, 2),
    ]
    order = [k for k in ("gmr", "omniretarget", "unitree", "colmo") if k in args.algos]
    labels = [ALGO_TABLE[k]["label"] for k in order]
    cw = 17
    width = 22 + cw * len(order)
    print("\n" + "=" * width)
    print(f"{args.robot}: Kinematic Quality  (mean ± std over {len(motions)} motions; "
          f"all metrics LOWER = better)")
    print("=" * width)
    print(f"{'Metric':<22}" + "".join(f"{lab:>{cw}}" for lab in labels))
    print("-" * width)
    for key, label, scale, nd in METRICS:
        cells = "".join(f"{fmt_ms(agg[k][key], scale, nd):>{cw}}" for k in order)
        print(f"{label:<22}{cells}")
    print("=" * width)

    if args.out:
        import csv
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["motion", "source", "frames", "P_ground", "D_ground_m",
                        "FootFloat_m", "P_self", "D_self_m", "V_slide_mps", "P_vel", "P_sat"])
            for motion, key, r in per_rows:
                w.writerow([motion, ALGO_TABLE[key]["label"], r["N"],
                            r["P_ground"], r["D_ground"], r["Foot_float"],
                            r["P_self"], r["D_self"],
                            r["V_slide"], r["P_vel"], r["P_sat"]])
        print(f"[eval] per-motion metrics -> {out}")


if __name__ == "__main__":
    main()
