"""Render the SMPL-X human *body* and the retargeted robot in one MuJoCo scene.

Paper-figure companion to ``scripts/vis_colmo_with_smplx.py``: instead of drawing the
human as sphere/capsule markers on ``user_scn``, the actual SMPL-X surface (10475 verts,
20908 faces) is registered as a MuJoCo **skin** asset and skinned to 55 mocap bodies --
one per SMPL-X joint. Every frame the mocap bodies are placed at that frame's SMPL-X
joint poses, so MuJoCo's linear-blend skinning reproduces the SMPL-X mesh directly.

Because the human is a real model asset (not a ``user_scn`` overlay), it lights, shades
and z-buffers against the robot exactly like the robot's own meshes -- which is what you
want in a figure.

``--scale_mode`` picks which human is drawn:

* ``original`` (default) -- true SMPL-X size and proportions.
* ``uniform``            -- the same body shrunk isotropically to the robot's stature.
* ``scaled``             -- the per-joint ``human_scale_table`` reference the IK actually
  tracks. Faithful to the retargeter, but visibly distorted: the config pulls the
  shoulders to 0.55 of their distance from the pelvis while the torso stays at 0.9, so
  the (unshrunk) arm mesh telescopes into the chest.

Examples
--------
    # live retargeting, human next to the robot, interactive viewer
    python scripts/vis_smplx_body_with_robot.py \
        --smplx_file human_motion/kimodo/amass_00.npz --robot unitree_g1

    # figure snapshots (no viewer -- works over ssh with MUJOCO_GL=egl)
    python scripts/vis_smplx_body_with_robot.py \
        --smplx_file human_motion/kimodo/amass_00.npz --headless \
        --snapshot_frames 0 60 120 --snapshot_dir figures/teaser

    # semi-transparent human overlaid ON the robot + video
    python scripts/vis_smplx_body_with_robot.py \
        --smplx_file human_motion/kimodo/amass_00.npz \
        --layout overlay --record_video --video_path videos/overlay.mp4

Notes
-----
* MuJoCo skinning is plain LBS, so SMPL-X's pose-corrective blend shapes are not
  reproduced (a sub-millimeter difference around bent joints -- invisible at figure size).
* The robot is always retargeted with the full IK config; ``--scale_mode`` only changes
  how the human is *drawn*, never how the robot is solved.
"""

import argparse
import hashlib
import importlib.util
import os
import pathlib
import pickle
import time

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rich import print
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# torch, smplx and the COLMO package are deliberately NOT imported at module scope: they
# are needed only to BUILD a solve, and importing the retargeting package alone costs
# ~1.4 s wall / 3.0 s CPU (it pulls in torch and mink). A cached run never touches any of
# them, so they are imported inside build_bundle() instead. The robot path tables live in
# params.py, which imports nothing but pathlib, so they are loaded from the file directly
# rather than through the package __init__ that would drag torch back in.
def _robot_tables():
    """(ROBOT_XML_DICT, ROBOT_BASE_DICT, IK_CONFIG_DICT), cheaply."""
    params_py = REPO_ROOT / "general_motion_retargeting" / "params.py"
    try:
        spec = importlib.util.spec_from_file_location("_colmo_params", params_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:                        # params grew package-relative imports
        import general_motion_retargeting as mod
    return mod.ROBOT_XML_DICT, mod.ROBOT_BASE_DICT, mod.IK_CONFIG_DICT


ROBOT_XML_DICT, ROBOT_BASE_DICT, IK_CONFIG_DICT = _robot_tables()

ROBOT_PREFIX = "robot_"
ROBOT2_PREFIX = "cmp_"          # second robot (compare mode: --compare_pkl)
BONE_PREFIX = "smplx_"          # mocap body per SMPL-X joint: smplx_<joint_name>
SKIN_NAME = "smplx_body"
SKIN_MATERIAL = "smplx_body_mat"

# Body color: #98A3AA (152, 163, 170), a blue-gray that separates from the default
# 184-gray backdrop without going so dark that the surface shading is lost.
DEFAULT_SKIN_RGBA = (113 / 255, 151 / 255, 171 / 255, 1.0)
# DEFAULT_SKIN_RGBA = (0.25, 0.50, 1.00, 1.0)


# ---------------------------------------------------------------------------
# SMPL-X -> MuJoCo skin asset
#
# A MuJoCo skin stores rest-pose vertices plus, per bone, (bindpos, bindquat) and the
# list of vertices it influences with weights. At render time it evaluates
#     v' = sum_b w_b * ( R_b * (v_rest - bindpos_b) + pos_b )
# with (pos_b, R_b) the bone body's current world pose. SMPL's LBS is
#     v' = sum_b w_b * ( G_b * G_b_rest^-1 ) * v_rest ,   G_b_rest = [I | J_b],
# i.e. exactly the same expression with bindpos_b = J_b (rest joint location) and
# bindquat_b = identity. So driving each bone body with the SMPL-X joint's *global*
# position and rotation reproduces SMPL-X skinning exactly (pose blend shapes aside).
# ---------------------------------------------------------------------------
def smplx_rest_shape(body_model, betas_raw):
    """Rest pose (zero pose, betas applied) -> (vertices, joints), the skin's bind pose.

    ``betas_raw`` is sliced/padded to the body model's ``num_betas`` the same way
    ``utils.smpl.load_smplx_file`` does, so the bind shape matches the posed motion.
    """
    import torch

    num_betas = int(getattr(body_model, "num_betas", body_model.shapedirs.shape[-1]))
    betas = np.asarray(betas_raw, dtype=np.float32)
    if betas.ndim > 1:
        betas = betas[0]
    betas = betas.reshape(-1)
    if betas.size > num_betas:
        betas = betas[:num_betas]
    elif betas.size < num_betas:
        betas = np.concatenate([betas, np.zeros(num_betas - betas.size, np.float32)])

    with torch.no_grad():
        out = body_model(betas=torch.from_numpy(betas).float().view(1, -1))
    num_joints = len(body_model.parents)
    verts = out.vertices[0].detach().numpy().astype(np.float64)
    joints = out.joints[0].detach().numpy().astype(np.float64)[:num_joints]
    return verts, joints


def build_bone_influences(body_model, weight_eps=1e-4):
    """Per-bone (vertex ids, weights) from the SMPL-X LBS weight matrix.

    Weights below ``weight_eps`` are dropped (SMPL-X is ~4 bones/vertex above that,
    vs. 55 dense columns), which keeps the skin asset small without a visible change --
    MuJoCo renormalizes the remaining weights per vertex.
    """
    W = body_model.lbs_weights.detach().numpy().astype(np.float64)
    vertid, vertweight = [], []
    for b in range(W.shape[1]):
        idx = np.where(W[:, b] > weight_eps)[0]
        if idx.size == 0:
            # A bone with no meaningful influence (can happen for eye/jaw joints on some
            # model versions) still needs a non-empty list; bind it to its best vertex.
            idx = np.array([int(np.argmax(W[:, b]))])
        vertid.append(idx.astype(np.int32).tolist())
        vertweight.append(np.maximum(W[idx, b], 1e-6).tolist())
    return vertid, vertweight


def build_scene_model(robot_xml, bone_names, rest_verts, rest_joints, faces,
                      vertid, vertweight, skin_rgba, skin_inflate=0.0,
                      skin_emission=0.25, skin_specular=0.15, skin_shininess=0.3,
                      robot2_xml=None):
    """Compose one model: the robot (name-prefixed) + the SMPL-X skin and its bones.

    NOTE on the blank-parent + attach shape: loading the robot spec as the PARENT and
    adding the human into it would be tidier (no prefixing, and it would stop MuJoCo
    logging an option-merge conflict to MUJOCO_LOG.TXT on every run). It is deliberately
    NOT done, because attach also drops the robot XML's <visual>/<statistic> blocks, and
    the shipped scenes set things there -- the G1 declares `rgba haze` and a 0.8-metre
    scene `extent` -- that visibly change the render: a hazy, close-curved horizon and a
    brighter headlight. Keeping attach keeps the figures pixel-identical; the dropped
    offscreen-buffer / AA / shadow settings are restored by hand in main().
    """
    spec = mj.MjSpec()

    # The shipped robot scenes are lit dimly (G1: one directional light at diffuse 0.5
    # plus a 0.6 headlight), so a pure-white albedo still resolves to mid-gray -- the
    # robot's own white parts land near RGB 73/255. Emission adds a light-independent
    # term, which is what actually makes the body read as WHITE in a figure while the
    # diffuse term keeps the surface shading that gives the mesh its shape.
    spec.add_material(name=SKIN_MATERIAL, rgba=list(skin_rgba),
                      emission=float(skin_emission), specular=float(skin_specular),
                      shininess=float(skin_shininess))

    # Attach the robot scene at the origin -- it also brings the floor and lights along.
    # Its free joint fully overrides any frame offset, so the side-by-side layout is
    # applied to the root position at runtime instead of here.
    robot = mj.MjSpec.from_file(str(robot_xml))
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
    spec.attach(robot, prefix=ROBOT_PREFIX, frame=frame)

    # Compare mode: a second robot (e.g. a saved GMR motion), name-prefixed so its
    # bodies/joints/geoms never collide with the first robot's. Its floor/lights are
    # dropped (the first robot's are kept) to avoid duplicate planes fighting for z.
    if robot2_xml is not None:
        robot2 = mj.MjSpec.from_file(str(robot2_xml))
        # strip the 2nd robot's floor plane(s) so only robot-1's floor remains
        for geom in list(robot2.worldbody.geoms):
            if geom.type == mj.mjtGeom.mjGEOM_PLANE:
                robot2.delete(geom)
        frame2 = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
        spec.attach(robot2, prefix=ROBOT2_PREFIX, frame=frame2)

    # One mocap body per SMPL-X joint. Mocap bodies carry no dofs and no geoms: they
    # exist purely so `data.mocap_pos/mocap_quat` can drive the skin bones directly,
    # which sidesteps building (and inverting) an articulated SMPL-X kinematic tree.
    for name, j in zip(bone_names, rest_joints):
        spec.worldbody.add_body(name=name, mocap=True, pos=j)

    identity_quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(bone_names), 1))
    spec.add_skin(
        name=SKIN_NAME,
        vert=rest_verts.flatten().tolist(),
        face=faces.astype(np.int32).flatten().tolist(),
        bodyname=list(bone_names),
        bindpos=rest_joints.flatten().tolist(),
        bindquat=identity_quat.flatten().tolist(),
        vertid=vertid,
        vertweight=vertweight,
        material=SKIN_MATERIAL,
        rgba=list(skin_rgba),
        inflate=float(skin_inflate),
    )
    return spec.compile()


# ---------------------------------------------------------------------------
# Human scaling (mirrors COLMO.scale_human_data)
# ---------------------------------------------------------------------------
def build_effective_scale(all_names, parents, base_scale):
    """Per-joint scale filled in for joints the IK config does NOT scale: each inherits
    its nearest scaled ancestor's factor (SMPL-X joint order guarantees parents precede
    children), so the skinned body stretches as one piece instead of tearing at
    unscaled intermediate joints. Same helper as vis_colmo_with_smplx.py."""
    eff = {}
    for i, name in enumerate(all_names):
        if name in base_scale:
            eff[name] = np.asarray(base_scale[name], dtype=float)
        elif parents[i] >= 0:
            eff[name] = eff[all_names[parents[i]]]
        else:
            eff[name] = np.ones(3)
    return eff


# ---------------------------------------------------------------------------
# Robot motion from a saved pkl (optional alternative to live retargeting)
# ---------------------------------------------------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Reads pkls written by either numpy 1.x (``numpy.core``) or 2.x (``numpy._core``).
    Copied from vis_compare_retargeting.py -- remapping happens only during class
    resolution, never in ``sys.modules`` (which would corrupt numpy and segfault)."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                return super().find_class("numpy.core" + module[len("numpy._core"):], name)
            if module.startswith("numpy.core"):
                return super().find_class("numpy._core" + module[len("numpy.core"):], name)
            raise


def _load_robot_pkl(path):
    with open(path, "rb") as f:
        data = _NumpyCompatUnpickler(f).load()
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    return root_pos, root_rot, dof_pos


# ---------------------------------------------------------------------------
# Motion bundle + on-disk cache.
#
# Everything the renderer needs is packed into one flat dict of arrays: the skin asset,
# the per-frame SMPL-X joint poses, the retargeter's scale parameters, and the solved
# robot qpos. Building it is the whole cost of this script -- the IK alone is ~17 ms per
# frame, and the SMPL-X forward pass another ~1 s -- while everything downstream
# (camera, colors, --scale_mode, layout, snapshots) is applied at draw time.
#
# Since a figure is made by re-running with tweaked framing, caching the bundle turns
# every run after the first into pure playback: no torch, no smplx, no COLMO, no IK.
# The cache key covers every input that can change the solve, so edits to the IK config,
# the collision config, the robot XML or the motion file all invalidate it on their own.
# ---------------------------------------------------------------------------
CACHE_VERSION = 1


def _stamp(path):
    """Identity of a file for cache keying: name + size + mtime (no hashing of content)."""
    p = pathlib.Path(path)
    if not p.exists():
        return f"{p.name}:absent"
    st = p.stat()
    return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"


def bundle_cache_path(args, smplx_path):
    robot_xml = pathlib.Path(ROBOT_XML_DICT[args.robot])
    parts = [
        str(CACHE_VERSION), _stamp(smplx_path), args.robot,
        str(args.motion_fps), str(args.collision_mode), str(args.weight_eps),
        _stamp(IK_CONFIG_DICT["smplx"][args.robot]),
        _stamp(robot_xml), _stamp(robot_xml.parent / "collision_cfg.yaml"),
    ]
    key = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return pathlib.Path(args.cache_dir) / f"{smplx_path.stem}_{args.robot}_{key}.npz"


def build_bundle(args, smplx_path, solve_to):
    """Load SMPL-X, build the skin asset, and solve the IK for frames [0, solve_to)."""
    import torch
    from general_motion_retargeting import GeneralMotionRetargeting as COLMO
    from general_motion_retargeting.utils.smpl import (
        load_smplx_file, get_smplx_data_offline_fast, JOINT_NAMES)

    if args.torch_threads > 0:
        # torch fans the SMPL-X forward pass across every core by default, which costs
        # ~5x the CPU for no wall-clock gain on a single clip (measured 0.90 s / 4.6
        # cores vs 0.82 s / 1.0 core). Must be set before the model runs.
        torch.set_num_threads(args.torch_threads)

    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        str(smplx_path), pathlib.Path(args.smplx_folder))
    frames, src_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.motion_fps)
    joint_names = list(JOINT_NAMES[: len(body_model.parents)])
    n_frames = len(frames)
    print(f"[green]SMPL-X[/green]: {smplx_path.name} -- {n_frames} frames @ "
          f"{src_fps:.1f} fps, height {human_height:.2f} m, {len(joint_names)} joints")

    retargeter = COLMO(src_human="smplx", tgt_robot=args.robot,
                       actual_human_height=human_height,
                       collision_mode=args.collision_mode)
    retargeter.adjust_hips_scale_for_motion(frames)

    rest_verts, rest_joints = smplx_rest_shape(body_model, smplx_data["betas"])
    vertid, vertweight = build_bone_influences(body_model, weight_eps=args.weight_eps)

    # IK is solved from frame 0 regardless of --start_frame: the solver is sequential
    # (warm start, plus the frame velocity/acceleration limits), so starting cold in the
    # middle of a clip gives a different -- and worse -- result than playing into it.
    solve_to = int(np.clip(solve_to, 0, n_frames))
    root_pos = np.zeros((solve_to, 3))
    root_rot = np.zeros((solve_to, 4))
    dof_pos = None
    for i in tqdm(range(solve_to), desc="retarget (cached after this)"):
        q = retargeter.retarget(frames[i], frame_idx=i)
        if dof_pos is None:
            dof_pos = np.zeros((solve_to, len(q) - 7))
        root_pos[i], root_rot[i], dof_pos[i] = q[:3], q[3:7], q[7:]
    if dof_pos is None:
        dof_pos = np.zeros((0, 0))

    scale_names = list(retargeter.human_scale_table.keys())
    scale_vals = np.array([np.broadcast_to(
        np.asarray(retargeter.human_scale_table[k], dtype=float).reshape(-1), (3,))
        for k in scale_names]) if scale_names else np.zeros((0, 3))
    root_xy = retargeter._root_xy_traj
    return dict(
        n_frames=np.int64(n_frames), n_solved=np.int64(solve_to),
        fps=np.float64(src_fps), human_height=np.float64(human_height),
        ground_offset=np.float64(retargeter.ground_offset),
        joint_names=np.array(joint_names), parents=np.asarray(body_model.parents, np.int64),
        human_root_name=np.array(retargeter.human_root_name),
        joint_pos=np.array([[frames[t][n][0] for n in joint_names]
                            for t in range(n_frames)], dtype=np.float64),
        joint_quat=np.array([[frames[t][n][1] for n in joint_names]
                             for t in range(n_frames)], dtype=np.float64),
        scale_names=np.array(scale_names) if scale_names else np.array([""])[:0],
        scale_vals=scale_vals,
        root_xy_traj=np.zeros((0, 2)) if root_xy is None else np.asarray(root_xy, float),
        robot_root_pos=root_pos, robot_root_rot=root_rot, robot_dof=dof_pos,
        skin_vert=rest_verts.astype(np.float32), skin_face=np.asarray(body_model.faces, np.int32),
        rest_joints=rest_joints,
        vertid_flat=np.concatenate([np.asarray(v, np.int32) for v in vertid]),
        vertid_off=np.cumsum([0] + [len(v) for v in vertid]).astype(np.int64),
        vertweight_flat=np.concatenate([np.asarray(w, np.float64) for w in vertweight]),
    )


# ---------------------------------------------------------------------------
# Backdrop.
#
# The visible background is the skybox above the horizon and the floor plane below it, so
# a flat backdrop means coloring every skybox texel AND hiding the floor. The skybox is
# not lit, so its texel value reaches the framebuffer verbatim -- the rendered background
# is EXACTLY the requested RGB. Texture data is uploaded to the GPU when the render
# context is built, which happens after this runs, so no mjr_uploadTexture is needed.
# ---------------------------------------------------------------------------
def parse_background(text):
    """'scene' -> None. '#RRGGBB' | 'R,G,B' (0-255) | 'white' -> uint8 (3,)."""
    s = str(text).strip()
    if s.lower() == "scene":
        return None
    if s.lower() == "white":
        return np.array([255, 255, 255], dtype=np.uint8)
    if s.startswith("#"):
        h = s[1:]
        if len(h) != 6:
            raise SystemExit(f"--background '{s}': hex must be #RRGGBB.")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    parts = s.replace(" ", "").split(",")
    if len(parts) != 3:
        raise SystemExit(f"--background '{s}': expected 'scene', '#RRGGBB' or 'R,G,B'.")
    try:
        vals = [int(round(float(p))) for p in parts]
    except ValueError:
        raise SystemExit(f"--background '{s}': non-numeric component.")
    if not all(0 <= v <= 255 for v in vals):
        raise SystemExit(f"--background '{s}': components must be 0-255.")
    return np.array(vals, dtype=np.uint8)


def skybox_texids(model):
    return [t for t in range(model.ntex)
            if model.tex_type[t] == mj.mjtTexture.mjTEXTURE_SKYBOX]


def set_skybox(model, rgb):
    """Flood every skybox texture with `rgb`. The shipped scenes declare two skyboxes and
    only one wins, so all of them are set rather than guessing which."""
    for t in skybox_texids(model):
        adr = int(model.tex_adr[t])
        nch = int(model.tex_nchannel[t])
        n = int(model.tex_height[t]) * int(model.tex_width[t]) * nch
        texel = np.resize(np.asarray(rgb, dtype=np.uint8), nch)
        if nch == 4:
            texel[3] = 255
        model.tex_data[adr:adr + n] = np.tile(texel, n // nch)


def bundle_ragged(bundle):
    """Unpack the flattened per-bone influence lists back into MuJoCo's list-of-lists."""
    off, ids, w = bundle["vertid_off"], bundle["vertid_flat"], bundle["vertweight_flat"]
    vertid = [ids[off[b]:off[b + 1]].tolist() for b in range(len(off) - 1)]
    vertweight = [w[off[b]:off[b + 1]].tolist() for b in range(len(off) - 1)]
    return vertid, vertweight


def resolve_bundle(args, smplx_path, solve_to):
    """Return the motion bundle, reusing the on-disk cache when it covers this request."""
    path = None if args.no_cache else bundle_cache_path(args, smplx_path)
    if path is not None and path.exists():
        try:
            cached = {k: v for k, v in np.load(path, allow_pickle=False).items()}
            n_frames = int(cached["n_frames"])
            need = min(solve_to, n_frames)
            if int(cached["n_solved"]) >= need:
                print(f"[green]Cache hit[/green]: {path.name} "
                      f"({n_frames} frames, {int(cached['n_solved'])} solved) -- "
                      f"skipping SMPL-X, COLMO and IK.")
                return cached
            print(f"[yellow]Cache covers only {int(cached['n_solved'])} solved frames, "
                  f"need {need} -- re-solving.[/yellow]")
        except Exception as exc:                       # corrupt / stale-format file
            print(f"[yellow]Ignoring unreadable cache {path.name}: {exc}[/yellow]")

    bundle = build_bundle(args, smplx_path, solve_to)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **bundle)
            print(f"[cyan]Cached solve -> {path}[/cyan]")
        except OSError as exc:
            print(f"[yellow]Could not write cache: {exc}[/yellow]")
    return bundle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the scaled SMPL-X body mesh and the retargeted robot in one "
                    "MuJoCo scene (paper figures / videos).")

    # --- input ---------------------------------------------------------------
    parser.add_argument("--smplx_file", required=True, type=str,
                        help="SMPL-X .npz motion file (AMASS/OMOMO format).")
    parser.add_argument("--smplx_folder", type=str,
                        default=str(REPO_ROOT / "assets" / "body_models"),
                        help="Folder holding the SMPL-X body models.")
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        choices=list(ROBOT_XML_DICT.keys()))
    parser.add_argument("--robot_motion_path", type=str, default=None,
                        help="Replay a saved robot motion pkl instead of solving IK on "
                             "the fly. Must be the SAME motion and robot; the human side "
                             "is scaled with the IK config either way.")
    parser.add_argument("--compare_pkl", type=str, default=None,
                        help="Compare mode: add a SECOND robot driven by this saved "
                             "motion pkl (e.g. a GMR result). Forces a 3-column layout: "
                             "human | center robot (COLMO / --robot_motion_path) | this "
                             "pkl on the right.")
    parser.add_argument("--compare_robot", type=str, default=None,
                        choices=list(ROBOT_XML_DICT.keys()),
                        help="Robot model for --compare_pkl (default: same as --robot).")
    parser.add_argument("--motion_fps", type=int, default=30,
                        help="Target fps the SMPL-X clip is resampled to.")
    parser.add_argument("--collision_mode", choices=["cbf", "issf", "off"], default=None,
                        help="COLMO collision-avoidance mode (unset -> YAML default). "
                             "Ignored with --robot_motion_path.")

    # --- cost control --------------------------------------------------------
    parser.add_argument("--no_cache", action="store_true",
                        help="Always re-solve instead of reusing the cached solve. The "
                             "cache key already covers the motion file, robot, fps, IK "
                             "config, collision config and robot XML, so this is only "
                             "needed to force a rebuild.")
    parser.add_argument("--cache_dir", type=str,
                        default=str(REPO_ROOT / ".cache" / "vis_smplx_body_with_robot"))
    parser.add_argument("--torch_threads", type=int, default=1,
                        help="Threads torch may use for the SMPL-X forward pass. The "
                             "library default (one per core) burns ~5x the CPU for no "
                             "wall-clock gain on a single clip, so this defaults to 1. "
                             "0 leaves torch's default alone.")

    # --- what the human looks like ------------------------------------------
    parser.add_argument("--scale_mode", choices=["original", "uniform", "scaled"],
                        default="original",
                        help="original (default): the SMPL-X body at its true size and "
                             "proportions. uniform: the same body shrunk isotropically to "
                             "the robot's stature (undistorted, see --uniform_scale). "
                             "scaled: the IK config's per-joint human_scale_table, i.e. "
                             "literally what the retargeter tracks -- faithful, but it "
                             "squashes the arms into the torso (shoulder 0.55 vs torso 0.9).")
    parser.add_argument("--uniform_scale", type=float, default=None,
                        help="Factor for --scale_mode uniform. Default: the IK config's "
                             "vertical root scale, i.e. how much shorter the retarget "
                             "reference human is than the real one.")
    parser.add_argument("--skin_rgba", type=float, nargs=4, default=None,
                        metavar=("R", "G", "B", "A"),
                        help=f"Human body color (default {DEFAULT_SKIN_RGBA} opaque, or "
                             f"alpha 0.45 in --layout overlay).")
    parser.add_argument("--skin_inflate", type=float, default=0.0,
                        help="Offset the skin outward along its normals (m). A small "
                             "value (e.g. 0.002) removes z-fighting in overlay mode.")
    parser.add_argument("--skin_emission", type=float, default=0.35,
                        help="Light-independent brightness added to the body. The robot "
                             "scenes are lit dimly, so at 0 the surface renders at only "
                             "~45%% of --skin_rgba and the body comes out far darker than "
                             "the color asked for. The default is tuned so the body's "
                             "MIDTONE matches --skin_rgba exactly, with the usual shading "
                             "above and below it. Drop to 0 for the physically-lit look.")
    parser.add_argument("--robot_alpha", type=float, default=None,
                        help="Override the alpha of every robot visual geom (e.g. 0.5).")
    parser.add_argument("--show_collision", action="store_true",
                        help="Also draw the robot's collision primitives (capsules/boxes).")
    parser.add_argument("--weight_eps", type=float, default=1e-4,
                        help="Drop LBS weights below this when building the skin.")
    parser.add_argument("--no_ground_offset", action="store_true",
                        help="Do not drop the human by COLMO's ground-calibration offset. "
                             "By default the human is lowered with the robot's IK targets "
                             "so both stand on the same floor. The offset is measured "
                             "during the first IK solve, so it is unavailable (and this "
                             "flag is implied) with --robot_motion_path.")

    # --- layout --------------------------------------------------------------
    parser.add_argument("--layout", choices=["sequence", "side", "overlay"],
                        default="sequence",
                        help="sequence (default): play the whole clip as the human, then "
                             "replay it as the robot -- one at a time, both on the same "
                             "spot so the two passes line up exactly. side: both at once "
                             "in adjacent lanes. overlay: both at once in the same lane, "
                             "human semi-transparent.")
    parser.add_argument("--axis", choices=["x", "y"], default="y",
                        help="World axis the two characters are lined up along.")
    parser.add_argument("--spacing", type=float, default=1.2,
                        help="Lateral gap (m) between the human and the robot.")
    parser.add_argument("--human_offset", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Explicit world offset for the human, overriding --layout.")
    parser.add_argument("--face_mode", choices=["lock", "frame0", "off"], default="lock",
                        help="How the characters are turned toward the camera. lock "
                             "(default): each one's base yaw is pinned to the front EVERY "
                             "frame, so they never turn away and any base-yaw difference "
                             "between them disappears -- pairs naturally with --root_mode "
                             "lock. frame0: aligned once at frame 0, then free to turn "
                             "with the motion. off: keep the clip's original heading.")
    parser.add_argument("--no_face_forward", action="store_true",
                        help="Alias for --face_mode off.")
    parser.add_argument("--face_yaw", type=float, default=None,
                        help="World yaw (deg) the characters' fronts should point at "
                             "(default: at the camera). Add 180 if they face away.")
    parser.add_argument("--root_mode", choices=["lock", "recenter", "absolute"],
                        default="recenter",
                        help="How each character's horizontal root is placed. lock: pinned "
                             "to its lane every frame (in-place, best for pose comparison); "
                             "recenter (default): recentered on its lane at t=0, then free "
                             "to travel; absolute: raw world translation plus the lane offset.")

    # --- playback ------------------------------------------------------------
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open the interactive viewer (needed over ssh; pair "
                             "with MUJOCO_GL=egl and --record_video / --snapshot_frames).")
    parser.add_argument("--no_rate_limit", action="store_true",
                        help="Do not cap the live viewer at the motion fps.")

    # --- camera --------------------------------------------------------------
    parser.add_argument("--cam_distance", type=float, default=4.0)
    parser.add_argument("--cam_azimuth", type=float, default=None,
                        help="Default: look down the axis the characters are NOT lined "
                             "up along, so both are visible side by side.")
    parser.add_argument("--cam_elevation", type=float, default=-12.0)
    parser.add_argument("--no_match_scale", action="store_true",
                        help="In --layout sequence, do NOT rescale the camera per pass. "
                             "By default the camera distance and reference height are "
                             "scaled by each character's standing height, so the human "
                             "and the (shorter) robot fill the frame identically; pass "
                             "this to keep --cam_distance as-is for both and let the real "
                             "size difference show. No effect in side/overlay, where both "
                             "share one camera.")
    parser.add_argument("--no_follow_camera", action="store_true")
    parser.add_argument("--cam_height", type=float, default=0.85,
                        help="Height (m) of the camera's reference point. Used as-is with "
                             "--no_follow_camera; otherwise only as the frame-0 value "
                             "before the camera locks onto the characters.")
    parser.add_argument("--lookat_shift", type=float, default=0.0,
                        help="Slide the camera reference point sideways in the image "
                             "plane: positive = left, negative = right (meters).")

    # --- output --------------------------------------------------------------
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Default: videos/<smplx stem>_<robot>_body.mp4")
    parser.add_argument("--video_quality", type=int, default=8)
    parser.add_argument("--snapshot_frames", type=int, nargs="+", default=None,
                        metavar="I", help="Frame indices to save as PNG stills.")
    parser.add_argument("--snapshot_all", action="store_true",
                        help="Save EVERY rendered frame as a PNG still into "
                             "--snapshot_dir (honors --start_frame/--end_frame).")
    parser.add_argument("--snapshot_dir", type=str, default="figures",
                        help="Where --snapshot_frames / --snapshot_all PNGs are written.")
    parser.add_argument("--width", type=int, default=1600,
                        help="Render width for video and snapshots.")
    parser.add_argument("--height", type=int, default=900,
                        help="Render height for video and snapshots.")
    parser.add_argument("--msaa", type=int, default=8,
                        help="Anti-aliasing samples (MuJoCo offsamples). Try 8 or 16.")
    parser.add_argument("--background", type=str, default="184,184,184",
                        help="Backdrop color as '#RRGGBB' or 'R,G,B' (0-255), or the word "
                             "'scene' to keep the robot XML's own sky and ground. A color "
                             "forces a flat skybox and hides the floor, so only the "
                             "character is drawn -- which also removes the ground shadow, "
                             "as there is no longer a floor to receive it. "
                             "Default: 184,184,184 (gray).")
    parser.add_argument("--no_shadow", action="store_true",
                        help="Disable shadows in the rendered video/snapshots.")
    parser.add_argument("--shadowsize", type=int, default=8192,
                        help="Shadow map resolution for the render. Raise it further if "
                             "the floor still shows shadow speckles.")

    return parser.parse_args()


def main():
    args = parse_args()
    smplx_path = pathlib.Path(args.smplx_file)

    # A headless GL backend (MUJOCO_GL=egl/osmesa) has no window system, so the live
    # passive viewer cannot open -- trying to create its context fails ("Failed to make
    # the EGL context current") and segfaults. Auto-switch to offscreen rendering.
    gl_backend = os.environ.get("MUJOCO_GL", "").lower()
    if gl_backend in ("egl", "osmesa") and not args.headless:
        args.headless = True
        print(f"[yellow]MUJOCO_GL={gl_backend}: no on-screen window available -> "
              f"forcing --headless (offscreen render).[/yellow]")

    need_render = args.record_video or args.snapshot_frames or args.snapshot_all
    if args.headless and not need_render:
        raise SystemExit("--headless with neither --record_video nor --snapshot_frames "
                         "would render nothing.")

    # --- motion bundle (cached solve) ---------------------------------------
    # --robot_motion_path supplies the robot pose itself, so no IK is needed; the human
    # side still has to be built. solve_to == 0 asks for a bundle with no IK in it.
    solve_to = 0 if args.robot_motion_path else (
        args.end_frame if args.end_frame is not None else np.iinfo(np.int32).max)
    bundle = resolve_bundle(args, smplx_path, solve_to)

    joint_names = [str(n) for n in bundle["joint_names"]]
    parents = [int(p) for p in bundle["parents"]]
    h_root = str(bundle["human_root_name"])
    src_fps = float(bundle["fps"])
    human_height = float(bundle["human_height"])
    joint_pos_all = bundle["joint_pos"]                    # (T, 55, 3)
    joint_quat_all = bundle["joint_quat"]                  # (T, 55, 4), wxyz
    human_scale_table = {str(k): np.asarray(v, float)
                         for k, v in zip(bundle["scale_names"], bundle["scale_vals"])}
    root_xy_traj = bundle["root_xy_traj"]
    if root_xy_traj.shape[0] == 0:
        root_xy_traj = None

    # `eff_scale` scales joint POSITIONS about the root; `body_scale` additionally scales
    # the bind mesh itself. Only the latter changes the body's proportions, which is why
    # "uniform" looks like a normal (just smaller) person while "scaled" does not: the
    # per-joint table moves the shoulders 45% closer to the pelvis without shrinking the
    # rib cage or the upper arm, so the arm mesh telescopes into the chest.
    body_scale = 1.0
    if args.scale_mode == "scaled":
        eff_scale = build_effective_scale(joint_names, parents, human_scale_table)
    elif args.scale_mode == "uniform":
        if args.uniform_scale is not None:
            body_scale = float(args.uniform_scale)
        else:
            # Vertical component of the root entry (per-axis [x, y, z] or a scalar),
            # already multiplied by the actual/assumed height ratio by COLMO.__init__.
            body_scale = float(human_scale_table[h_root].reshape(-1)[-1])
        eff_scale = {n: np.full(3, body_scale) for n in joint_names}
        print(f"[cyan]Uniform body scale[/cyan]: {body_scale:.3f} "
              f"({human_height:.2f} m -> {human_height * body_scale:.2f} m)")
    else:                                                     # original
        eff_scale = {n: np.ones(3) for n in joint_names}

    # The base-speed-capped root xy lives in the per-joint SCALED frame, so it must not
    # leak into the modes that draw the human at its own (un)scaled size.
    if args.scale_mode != "scaled":
        root_xy_traj = None

    # --- robot motion source -------------------------------------------------
    if args.robot_motion_path is not None:
        robot_pkl = _load_robot_pkl(args.robot_motion_path)
        print(f"[green]Robot motion[/green]: {args.robot_motion_path} "
              f"({len(robot_pkl[0])} frames, {robot_pkl[2].shape[1]} dof)")
    else:
        robot_pkl = (bundle["robot_root_pos"], bundle["robot_root_rot"],
                     bundle["robot_dof"])

    # --- compare mode: a second robot from a saved pkl (right column) ---------
    compare = args.compare_pkl is not None
    robot2_pkl = None
    robot2_name = args.compare_robot or args.robot
    if compare:
        robot2_pkl = _load_robot_pkl(args.compare_pkl)
        print(f"[green]Compare robot[/green]: {args.compare_pkl} "
              f"({len(robot2_pkl[0])} frames, {robot2_pkl[2].shape[1]} dof, "
              f"model '{robot2_name}')")

    # --- SMPL-X skin asset ---------------------------------------------------
    rest_verts = bundle["skin_vert"].astype(np.float64)
    rest_joints = bundle["rest_joints"]
    if body_scale != 1.0:
        # Shrink the bind mesh AND its bind joints by the same factor. Combined with
        # eff_scale == body_scale on the joint positions, LBS then yields
        #   v' = R (s*v - s*J) + s*t = s * ( R (v - J) + t ),
        # an exact uniform scaling of the posed body about the world origin -- so the
        # proportions are untouched.
        rest_verts = rest_verts * body_scale
        rest_joints = rest_joints * body_scale
    vertid, vertweight = bundle_ragged(bundle)
    bone_names = [BONE_PREFIX + n for n in joint_names]

    skin_rgba = args.skin_rgba
    if skin_rgba is None:
        skin_rgba = list(DEFAULT_SKIN_RGBA)
        if args.layout == "overlay":
            skin_rgba[3] = 0.45

    model = build_scene_model(
        ROBOT_XML_DICT[args.robot], bone_names, rest_verts, rest_joints,
        bundle["skin_face"], vertid, vertweight, skin_rgba, args.skin_inflate,
        skin_emission=args.skin_emission,
        robot2_xml=ROBOT_XML_DICT[robot2_name] if compare else None)
    data = mj.MjData(model)
    print(f"[cyan]Scene[/cyan]: skin {model.nskinvert} verts / {model.nskinface} faces "
          f"over {model.nskinbone} bones, robot '{args.robot}'")

    # mj_kinematics is ~37x cheaper than mj_forward but does NOT fill light_xpos /
    # light_xdir, and those are what give the floor its directional shading and the
    # characters their shadow -- without them the scene renders flat. Every shipped robot
    # scene declares its lights on the worldbody, so their frames are constant: one
    # mj_forward here establishes them and the per-frame mj_kinematics leaves them alone.
    # A light parented to a moving body would need the full solve every frame.
    lights_static = bool(np.all(model.light_bodyid[:model.nlight] == 0))
    mj.mj_forward(model, data)

    # One pass per character in sequence mode, otherwise a single pass with everything
    # shown at once. In compare mode sequence runs three takes (human -> COLMO -> GMR);
    # any other layout puts all three in a side-by-side row.
    if compare:
        passes = ["human", "robot", "robot2"] if args.layout == "sequence" else ["both"]
    else:
        passes = ["human", "robot"] if args.layout == "sequence" else ["both"]

    # Flat backdrop: color the skybox and hide the floor (see parse_background above).
    background_rgb = parse_background(args.background)
    if background_rgb is not None:
        set_skybox(model, background_rgb)
        for gid in range(model.ngeom):
            if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
                model.geom_rgba[gid, 3] = 0.0

    # Hide the robot's collision primitives. Group ids are NOT portable here -- G1 puts
    # collision shapes in group 2 while H1 and Booster T1 put their visual meshes there --
    # but across every shipped robot the collision shapes are exactly the geoms with
    # conaffinity != 0, and the visual meshes exactly those with conaffinity == 0. The
    # floor plane is excluded by hand (its conaffinity varies from 0 to 63 by model).
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_conaffinity[gid] != 0 and not args.show_collision:
            model.geom_rgba[gid, 3] = 0.0
        elif args.robot_alpha is not None:
            model.geom_rgba[gid, 3] = args.robot_alpha

    # Index maps: the robot's free-joint block in qpos, and each bone's mocap slot.
    # The skin bones are mocap bodies (no dofs), so every joint in the composed model
    # belongs to the robot -- found structurally rather than by name, because some robot
    # XMLs (e.g. unitree_h1) leave the floating-base joint unnamed.
    def robot_slots(prefix, robot_name):
        """(free-joint qpos address, dof count, base body id) for one attached robot,
        found by BODY-name prefix so it works even when the free joint is unnamed."""
        fj = [jid for jid in range(model.njnt)
              if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE
              and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.jnt_bodyid[jid])
                   or "").startswith(prefix)]
        if len(fj) != 1:
            raise SystemExit(f"Expected exactly one free joint for '{prefix}', "
                             f"found {len(fj)}.")
        qa = int(model.jnt_qposadr[fj[0]])
        nd = sum(1 for jid in range(model.njnt)
                 if model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE
                 and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
                      or "").startswith(prefix))
        bid = model.body(prefix + ROBOT_BASE_DICT[robot_name]).id
        return qa, nd, bid

    qadr, robot_ndof, robot_base_bid = robot_slots(ROBOT_PREFIX, args.robot)
    if robot_pkl is not None and robot_pkl[2].shape[1] != robot_ndof:
        raise SystemExit(f"pkl has {robot_pkl[2].shape[1]} dof but robot '{args.robot}' "
                         f"expects {robot_ndof} -- the pkl must match --robot.")
    if compare:
        qadr2, robot2_ndof, robot2_base_bid = robot_slots(ROBOT2_PREFIX, robot2_name)
        if robot2_pkl[2].shape[1] != robot2_ndof:
            raise SystemExit(f"--compare_pkl has {robot2_pkl[2].shape[1]} dof but robot "
                             f"'{robot2_name}' expects {robot2_ndof}.")
    mocap_ids = np.array([model.body_mocapid[model.body(n).id] for n in bone_names])

    # Per-frame posing is pure numpy on these: a (55, 3) scale row-vector and the joint
    # index of the root. The old path walked a 55-entry dict of tuples every frame.
    scale_arr = np.array([eff_scale[n] for n in joint_names])
    root_idx = joint_names.index(h_root)

    # Showing one character at a time. The human is the model's only skin, so mjVIS_SKIN
    # toggles it; the robot is every non-plane geom (the skin bones are geomless mocap
    # bodies), so its saved alphas are zeroed to hide it. The floor plane is excluded from
    # both, so the ground and its shadows stay put across passes. Both channels are picked
    # up by the offscreen renderer directly and by the passive viewer through sync().
    def gids_for(prefix):
        return np.array(
            [g for g in range(model.ngeom)
             if model.geom_type[g] != mj.mjtGeom.mjGEOM_PLANE
             and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
                  or "").startswith(prefix)], dtype=int)

    robot1_gids = gids_for(ROBOT_PREFIX)
    robot2_gids = gids_for(ROBOT2_PREFIX) if compare else np.array([], dtype=int)
    robot_gids = (np.concatenate([robot1_gids, robot2_gids]) if compare
                  else robot1_gids)
    robot1_alpha0 = model.geom_rgba[robot1_gids, 3].copy()
    robot2_alpha0 = model.geom_rgba[robot2_gids, 3].copy() if compare else None
    robot_alpha0 = model.geom_rgba[robot_gids, 3].copy()

    # --- matching the two characters' on-screen size -------------------------
    # A G1 is 1.32 m against a 1.85 m human, so shown one after the other at a fixed
    # camera the robot looks much smaller. Scaling BOTH the camera distance and the
    # look-at height by each character's standing height makes the two passes
    # geometrically similar renders: same framing, same apparent size.
    #
    # Robot: visual mesh extent with every joint at zero and the base upright, i.e. its
    # canonical standing height. Measured on a scratch MjData so `data` is untouched, and
    # from real mesh vertices -- geom_rbound (a bounding SPHERE) overshoots badly, giving
    # 1.48 m for the G1 instead of 1.32 m.
    def robot_standing_height():
        scratch = mj.MjData(model)
        scratch.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_kinematics(model, scratch)
        lo, hi = np.inf, -np.inf
        # Only geoms that actually get drawn: the collision primitives were alpha-zeroed
        # above and stick out well past the meshes (they put the G1 at 1.43 m, not 1.32).
        for g in robot1_gids[robot1_alpha0 > 0]:
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                w = (model.mesh_vert[adr:adr + num] @ scratch.geom_xmat[g].reshape(3, 3).T
                     + scratch.geom_xpos[g])
                zz = w[:, 2]
            else:
                r = model.geom_rbound[g]
                zz = scratch.geom_xpos[g, 2] + np.array([-r, r])
            lo, hi = min(lo, zz.min()), max(hi, zz.max())
        return float(hi - lo)

    # Human: the bind mesh's height. SMPL-X's canonical frame is y-up (a fixed property of
    # the body model, not of the motion), and rest_verts already carries --scale_mode
    # uniform's body_scale. The extra factor picks up --scale_mode scaled, where the whole
    # vertical chain (root z, spine3, hip, knee, ankle) shrinks by the same table entry.
    h_robot = robot_standing_height()
    h_human = float(rest_verts[:, 1].max() - rest_verts[:, 1].min())
    h_human *= float(eff_scale[h_root][2]) / body_scale
    match_scale = (args.layout == "sequence") and not args.no_match_scale
    pass_ratio = {"human": 1.0, "robot": h_robot / h_human,
                  "robot2": h_robot / h_human, "both": 1.0}
    if match_scale:
        print(f"[cyan]Scale match[/cyan]: human {h_human:.2f} m, robot {h_robot:.2f} m "
              f"-> robot pass rendered at {pass_ratio['robot']:.3f}x camera distance")
    else:
        pass_ratio = {k: 1.0 for k in pass_ratio}
    cur_ratio = [1.0]

    # In side-by-side compare (layout != sequence) three lanes must fit, so pull the
    # camera back to span them. Sequential compare shows one at a time, so it keeps the
    # per-character framing. Either respects a larger user --cam_distance.
    side_by_side = compare and args.layout != "sequence"
    cam_distance_base = max(args.cam_distance, 2.0 * args.spacing + 4.0) if side_by_side \
        else args.cam_distance

    def show_pass(name):
        # Per-robot visibility: pass "robot" = COLMO only, "robot2" = GMR only,
        # "both" = everything (side-by-side), "human" = skin only.
        model.geom_rgba[robot1_gids, 3] = (robot1_alpha0 if name in ("robot", "both")
                                           else 0.0)
        if compare:
            model.geom_rgba[robot2_gids, 3] = (robot2_alpha0 if name in ("robot2", "both")
                                               else 0.0)
        opt.flags[mj.mjtVisFlag.mjVIS_SKIN] = 1 if name in ("human", "both") else 0
        cur_ratio[0] = pass_ratio[name]
        cam.distance = cam_distance_base * cur_ratio[0]
        cam.lookat[2] = args.cam_height * cur_ratio[0]

    # --- layout --------------------------------------------------------------
    # "sequence" shows one character at a time, so both sit on the same spot (zero
    # offset): the two passes then occupy identical screen space and can be compared by
    # flipping between them.
    axis_idx = 0 if args.axis == "x" else 1
    human_offset = np.zeros(3)
    robot_offset = np.zeros(3)
    robot2_offset = np.zeros(3)
    if side_by_side:
        # Three lanes: human (left) | center robot (COLMO) | compare robot (GMR, right).
        human_offset[axis_idx] = -args.spacing
        robot_offset[axis_idx] = 0.0
        robot2_offset[axis_idx] = +args.spacing
    elif args.layout == "side":
        human_offset[axis_idx] = -0.5 * args.spacing
        robot_offset[axis_idx] = +0.5 * args.spacing
    if args.human_offset is not None:
        human_offset = np.asarray(args.human_offset, dtype=np.float64)

    def horizontal_shift(offset, root0_xy, cur_xy):
        """Map a character's horizontal root to its on-screen lane. z is never touched,
        so feet keep contact with the floor."""
        if args.root_mode == "lock":
            return np.array([offset[0] - cur_xy[0], offset[1] - cur_xy[1], 0.0])
        if args.root_mode == "recenter":
            return np.array([offset[0] - root0_xy[0], offset[1] - root0_xy[1], 0.0])
        return np.array([offset[0], offset[1], 0.0])

    # --- frame range ---------------------------------------------------------
    n_total = min(len(joint_pos_all), len(robot_pkl[0]))
    if compare:
        n_total = min(n_total, len(robot2_pkl[0]))
    i0 = max(0, args.start_frame)
    i1 = n_total if args.end_frame is None else min(n_total, args.end_frame)
    if i0 >= i1:
        raise SystemExit(f"Empty frame range [{i0}, {i1}).")
    indices = list(range(i0, i1))

    # --- viewer / camera -----------------------------------------------------
    cam_azimuth = args.cam_azimuth if args.cam_azimuth is not None \
        else (180.0 if args.axis == "y" else 90.0)
    az = np.deg2rad(cam_azimuth)
    fwd = np.array([-np.cos(az), -np.sin(az), 0.0])
    # Sideways slide in the image plane, plus the eye-level height the reference point
    # sits at (the floor-level default of a raw MjvCamera crops both heads).
    lookat_offset = np.cross(fwd, [0.0, 0.0, 1.0]) * args.lookat_shift
    lookat_offset[2] = args.cam_height

    # --- facing ---------------------------------------------------------------
    # Each character is turned about a VERTICAL line so its base yaw points at the camera.
    # Human and robot get their OWN correction rather than a shared one: the robot base
    # tracks the human pelvis only up to the IK config's rot_offset, which on this data
    # leaves a steady ~3 deg heading difference -- small, but plainly visible as the two
    # figures not quite facing the same way. Correcting each independently removes it.
    #
    # The human heading is read off the HIP LINE rather than the pelvis quaternion so it
    # does not depend on SMPL-X's root frame convention: with z up, forward = up x
    # (right - left). The robot heading is its base body's +x, which is forward on every
    # shipped robot (pelvis / Waist). The camera looks ALONG (cos az, sin az), so a front
    # pointing at it is yaw az + 180.
    ROBOT_FORWARD_AXIS = np.array([1.0, 0.0, 0.0])
    face_mode = "off" if args.no_face_forward else args.face_mode
    target_yaw = np.deg2rad(args.face_yaw if args.face_yaw is not None
                            else cam_azimuth + 180.0)

    dyaw_h = np.zeros(len(joint_pos_all))
    dyaw_r = np.zeros(len(robot_pkl[1]))
    dyaw_r2 = np.zeros(len(robot2_pkl[1])) if compare else None
    if face_mode != "off":
        try:
            li, ri = joint_names.index("left_hip"), joint_names.index("right_hip")
        except ValueError:
            li = None
            print("[yellow]facing: no hip joints in the motion, the human keeps its "
                  "original heading.[/yellow]")
        if li is not None:
            fwd_h = np.cross([0.0, 0.0, 1.0], joint_pos_all[:, ri] - joint_pos_all[:, li])
            dyaw_h = target_yaw - np.arctan2(fwd_h[:, 1], fwd_h[:, 0])
        fwd_r = R.from_quat(robot_pkl[1], scalar_first=True).apply(ROBOT_FORWARD_AXIS)
        dyaw_r = target_yaw - np.arctan2(fwd_r[:, 1], fwd_r[:, 0])
        if compare:
            fwd_r2 = R.from_quat(robot2_pkl[1], scalar_first=True).apply(ROBOT_FORWARD_AXIS)
            dyaw_r2 = target_yaw - np.arctan2(fwd_r2[:, 1], fwd_r2[:, 0])
        if face_mode == "frame0":
            # One fixed correction per character, so the motion's own turning survives.
            dyaw_h[:] = dyaw_h[min(i0, len(dyaw_h) - 1)]
            dyaw_r[:] = dyaw_r[min(i0, len(dyaw_r) - 1)]
            if compare:
                dyaw_r2[:] = dyaw_r2[min(i0, len(dyaw_r2) - 1)]
        gap = np.rad2deg((dyaw_h[:n_total] - dyaw_r[:n_total] + np.pi) % (2 * np.pi) - np.pi)
        print(f"[cyan]Facing[/cyan]: {face_mode} -> base yaw at "
              f"{np.rad2deg(target_yaw) % 360:.1f} deg; human/robot heading difference "
              f"{gap.mean():+.1f} deg (max |{np.abs(gap).max():.1f}|) corrected out")

    facing = face_mode != "off"

    def yaw_about(points, pivot_xy, dyaw):
        """Rotate world points about the vertical line through `pivot_xy` (z untouched)."""
        points = np.atleast_2d(points)
        if not facing:
            return points
        pivot = np.array([pivot_xy[0], pivot_xy[1], 0.0])
        return R.from_euler("z", dyaw).apply(points - pivot) + pivot

    paused = [False]

    def key_callback(keycode):
        if keycode == 32:  # SPACE
            paused[0] = not paused[0]
            print(f"[{'PAUSED' if paused[0] else 'RESUMED'}] (SPACE toggles)")

    viewer = None
    if not args.headless:
        viewer = mjv.launch_passive(model=model, data=data, show_left_ui=False,
                                    show_right_ui=False, key_callback=key_callback)
        cam, opt = viewer.cam, viewer.opt
    else:
        cam, opt = mj.MjvCamera(), mj.MjvOption()
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)

    opt.flags[mj.mjtVisFlag.mjVIS_SKIN] = 1                 # (default on; be explicit)
    opt.label = mj.mjtLabel.mjLABEL_NONE
    # Sites are not geoms, so hiding the robot by zeroing geom alpha leaves them drawn --
    # the shipped scenes declare IMU sites, which showed up as small gray spheres floating
    # around the human's pelvis during the human pass. Nothing here ever wants them.
    opt.sitegroup[:] = 0
    cam.distance = cam_distance_base
    cam.azimuth = cam_azimuth
    cam.elevation = args.cam_elevation
    cam.lookat[:] = lookat_offset

    # --- offscreen renderer (video + snapshots) ------------------------------
    renderer = None
    mp4_writer = None
    if need_render:
        import imageio
        # MjSpec.attach drops the child XML's <visual> block, so the composed model keeps
        # MuJoCo's 640x480 offscreen buffer and default AA. Grow both before the renderer
        # builds its GL context, or mj.Renderer refuses the requested resolution.
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
        model.vis.quality.offsamples = max(int(model.vis.quality.offsamples), args.msaa)
        # Some robot scenes (unitree_h1, booster_t1) light the floor with a big
        # shadow-casting directional light whose default shadow map is too coarse for a
        # 10k-vertex skin: it speckles the floor with acne. A finer map removes it.
        model.vis.quality.shadowsize = max(int(model.vis.quality.shadowsize),
                                           args.shadowsize)
        renderer = mj.Renderer(model, height=args.height, width=args.width)
        renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0 if args.no_shadow else 1

    if args.record_video:
        if args.video_path is None:
            args.video_path = f"videos/{smplx_path.stem}_{args.robot}_body.mp4"
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        mp4_writer = imageio.get_writer(args.video_path, fps=int(round(src_fps)),
                                        quality=args.video_quality, macro_block_size=None)
        print(f"[cyan]Recording -> {args.video_path}[/cyan]")

    snapshot_set = set(args.snapshot_frames or [])
    if snapshot_set or args.snapshot_all:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    # The ground offset is a single constant measured during the retargeter's first-frame
    # calibration; with a replayed pkl there was no calibration to read it from.
    ground = 0.0 if (args.no_ground_offset or args.robot_motion_path) \
        else float(bundle["ground_offset"])

    def human_positions(i):
        """Scaled world positions for every SMPL-X joint of frame i -- vectorized.

        Reproduces COLMO.scale_human_data + apply_ground_offset: the root is scaled about
        the world origin (or replaced by the base-speed-capped trajectory), every other
        joint is scaled about the root, and the body drops by `ground`. Note the root row
        needs no special case: (root - root) * s + scaled_root == scaled_root.
        """
        raw = joint_pos_all[i]
        raw_root = raw[root_idx]
        scaled_root = scale_arr[root_idx] * raw_root
        if root_xy_traj is not None:
            scaled_root = scaled_root.copy()
            scaled_root[:2] = root_xy_traj[i % len(root_xy_traj)]
        pos = (raw - raw_root) * scale_arr + scaled_root
        pos[:, 2] -= ground
        return pos

    # Frame-0 anchors for --root_mode recenter. The human's is known up front; the
    # robot's is read from the first frame actually played.
    human_root0_xy = human_positions(i0)[root_idx, :2].copy()
    robot_root0_xy = [None]
    robot2_root0_xy = [None]

    def step(i, shown="both"):
        """Fetch frame i and write both characters into `data`.

        Both are always posed; `shown` only decides what the camera follows (visibility
        is handled once per pass by show_pass, not per frame).
        """
        # 1) Robot pose. In "frame0" the whole trajectory swings about the frame-0 root,
        #    so travel turns with the body; in "lock" each frame pivots on its OWN root,
        #    which leaves the root where it is and only re-aims the body.
        root_pos, root_quat, dof = robot_pkl[0][i], robot_pkl[1][i], robot_pkl[2][i]
        if robot_root0_xy[0] is None:
            robot_root0_xy[0] = np.asarray(root_pos[:2], dtype=np.float64).copy()
        pivot_r = robot_root0_xy[0] if face_mode == "frame0" else root_pos[:2]
        root_pos = yaw_about(root_pos, pivot_r, dyaw_r[i])[0]
        if facing:
            root_quat = (R.from_euler("z", dyaw_r[i])
                         * R.from_quat(root_quat, scalar_first=True)).as_quat(scalar_first=True)
        shift = horizontal_shift(robot_offset, robot_root0_xy[0], root_pos[:2])
        data.qpos[qadr:qadr + 3] = root_pos + shift
        data.qpos[qadr + 3:qadr + 7] = root_quat
        data.qpos[qadr + 7:qadr + 7 + robot_ndof] = dof

        # 1b) Second (compare) robot, driven exactly like the first from its own pkl.
        if compare:
            rp2, rq2, dof2 = robot2_pkl[0][i], robot2_pkl[1][i], robot2_pkl[2][i]
            if robot2_root0_xy[0] is None:
                robot2_root0_xy[0] = np.asarray(rp2[:2], dtype=np.float64).copy()
            pivot_r2 = robot2_root0_xy[0] if face_mode == "frame0" else rp2[:2]
            rp2 = yaw_about(rp2, pivot_r2, dyaw_r2[i])[0]
            if facing:
                rq2 = (R.from_euler("z", dyaw_r2[i])
                       * R.from_quat(rq2, scalar_first=True)).as_quat(scalar_first=True)
            shift2 = horizontal_shift(robot2_offset, robot2_root0_xy[0], rp2[:2])
            data.qpos[qadr2:qadr2 + 3] = rp2 + shift2
            data.qpos[qadr2 + 3:qadr2 + 7] = rq2
            data.qpos[qadr2 + 7:qadr2 + 7 + robot2_ndof] = dof2

        # 2) Human body: scaled joint poses -> skin bones.
        pos = human_positions(i)
        pivot_h = human_root0_xy if face_mode == "frame0" else pos[root_idx, :2].copy()
        pos = yaw_about(pos, pivot_h, dyaw_h[i])
        quat = joint_quat_all[i]
        if facing:
            quat = (R.from_euler("z", dyaw_h[i])
                    * R.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True)
        h_shift = horizontal_shift(human_offset, human_root0_xy, pos[root_idx, :2])
        data.mocap_pos[mocap_ids] = pos + h_shift
        data.mocap_quat[mocap_ids] = quat

        # Kinematics only. mj_forward would additionally run collision detection and the
        # whole constraint/acceleration pipeline, none of which the renderer reads --
        # measured 0.225 ms vs 0.006 ms per frame on this scene.
        if lights_static:
            mj.mj_kinematics(model, data)
        else:
            mj.mj_forward(model, data)

        if not args.no_follow_camera:
            # Track whoever is on screen horizontally, but hold the reference height
            # fixed so the camera does not bob with the pelvis (that reads as jitter).
            human_root_w = pos[root_idx] + h_shift
            robot_root_w = data.xpos[robot_base_bid]
            if shown == "human":
                target = human_root_w
            elif shown == "robot":
                target = robot_root_w
            elif shown == "robot2":
                target = data.xpos[robot2_base_bid]
            elif compare:                     # "both" side-by-side: center the 3-lane row
                target = (human_root_w + robot_root_w
                          + data.xpos[robot2_base_bid]) / 3.0
            else:                             # "both" single robot
                target = 0.5 * (human_root_w + robot_root_w)
            cam.lookat[0] = target[0] + lookat_offset[0]
            cam.lookat[1] = target[1] + lookat_offset[1]
            cam.lookat[2] = args.cam_height * cur_ratio[0]

    def render():
        renderer.update_scene(data, camera=cam, scene_option=opt)
        return renderer.render()

    rate_limiter = None
    if viewer is not None and not args.no_rate_limit:
        rate_limiter = RateLimiter(frequency=src_fps, warn=False)

    # Flat (pass, frame) timeline: one entry per rendered frame. In sequence mode that is
    # the whole clip as the human followed by the whole clip as the robot, which is also
    # exactly what gets appended to the video -- human take first, robot take second.
    timeline = [(p, i) for p in passes for i in indices]
    if len(passes) > 1:
        print(f"[cyan]Sequence[/cyan]: {' -> '.join(passes)}, "
              f"{len(indices)} frames each ({len(timeline)} total)")

    pbar = tqdm(total=len(timeline), desc=" -> ".join(passes))
    k = 0
    cur_pass = [None]
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if paused[0]:
                if viewer is not None:
                    viewer.sync()
                    time.sleep(0.01)
                continue

            shown, i = timeline[k]
            if shown != cur_pass[0]:
                show_pass(shown)
                cur_pass[0] = shown
            step(i, shown)

            if viewer is not None:
                viewer.sync()
            if mp4_writer is not None:
                mp4_writer.append_data(render())
            if args.snapshot_all or i in snapshot_set:
                tag = f"_{shown}" if len(passes) > 1 else ""
                out = os.path.join(args.snapshot_dir,
                                   f"{smplx_path.stem}_{args.robot}{tag}_{i:05d}.png")
                imageio.imwrite(out, render())
                print(f"[cyan]snapshot[/cyan] {out}")

            # Cap to real time only for live viewing; when only recording, render as
            # fast as possible (the mp4 fps is fixed).
            if rate_limiter is not None and not need_render:
                rate_limiter.sleep()

            pbar.update(1)
            k += 1
            if k >= len(timeline):
                if args.loop and viewer is not None and mp4_writer is None:
                    k = 0
                    pbar.reset()
                else:
                    break
    finally:
        pbar.close()
        if mp4_writer is not None:
            mp4_writer.close()
            print(f"[cyan]Video saved to {args.video_path}[/cyan]")
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
            time.sleep(0.3)


if __name__ == "__main__":
    main()
