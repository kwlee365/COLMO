"""Solve COLMO retargeting for a robot and show the SMPL-X human skeleton in the same
MuJoCo viewer.

SMPL-X counterpart of vis_colmo_with_bvh.py -- same markers, colors, labels, keys and
flags. Differences come only from the input format:
  * the skeleton hierarchy comes from the SMPL-X body model (``body_model.parents`` +
    ``JOINT_NAMES``) instead of the BVH header;
  * SMPL-X carries 55 joints (22 body + jaw + 2 eyes + 30 hand joints), so
    ``--draw_joints`` defaults to the 22 body joints -- the set that corresponds to a
    LAFAN1 BVH skeleton. Pass ``--draw_joints all`` to include fingers/face;
  * there is no FootMod synthetic bone (that is a LAFAN1 loader construct); the SMPL-X
    config drives left_ankle_roll_link from the real ``left_ankle`` joint.
"""

import argparse
import pathlib
import time

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    GeneralMotionRetargeting as COLMO,
    ROBOT_XML_DICT,
    ROBOT_BASE_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
    JOINT_NAMES,
)

# SMPL-X joints 0..21 are the body tree (pelvis .. right_wrist); 22 is the jaw, 23/24 the
# eyes and 25.. the 30 hand joints. Drawing the hands puts ~30 extra labelled markers and
# coordinate frames on the screen, which is unreadable at the default axes size.
NUM_BODY_JOINTS = 22


def draw_sphere(viewer, pos, radius=0.025, rgba=(1.0, 0.3, 0.3, 1.0), label=None):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    # Always set the label (empty if None) so that a re-used geom slot from
    # a previous frame doesn't leak its stale label string.
    geom.label = label if label is not None else ""
    viewer.user_scn.ngeom += 1


def draw_bone(viewer, from_pos, to_pos, radius=0.012, rgba=(0.9, 0.9, 0.2, 1.0)):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
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
    viewer.user_scn.ngeom += 1


def draw_frame_axes(viewer, pos, quat_wxyz, size=0.08):
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba_list[i],
        )
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos,
            to=pos + size * mat[:, i],
        )
        geom.label = ""
        viewer.user_scn.ngeom += 1


def load_smplx_motion(smplx_file, smplx_folder, tgt_fps=30, draw_joints="body"):
    """Load an SMPL-X .npz and return everything the viewer needs.

    Returns (frames, height, joints, parents, edges, fps) where `joints` is the ordered
    list of joint names to DRAW, `parents` maps each drawn joint to its parent index
    within the SMPL-X tree, and `edges` are (child, parent) name pairs restricted to the
    drawn set -- the analogue of the BVH bone list / hierarchy.
    """
    smplx_data, body_model, smplx_output, height = load_smplx_file(
        smplx_file, pathlib.Path(smplx_folder))
    frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    all_names = JOINT_NAMES[: len(body_model.parents)]
    parents = [int(p) for p in body_model.parents]
    n = NUM_BODY_JOINTS if draw_joints == "body" else len(all_names)
    joints = [j for j in all_names[:n] if j in frames[0]]
    keep = set(joints)
    edges = [(all_names[i], all_names[parents[i]])
             for i in range(len(all_names))
             if parents[i] >= 0 and all_names[i] in keep and all_names[parents[i]] in keep]
    return frames, height, joints, all_names, parents, edges, fps


def build_effective_scale(all_names, parents, base_scale):
    """Per-joint scale mirroring COLMO.scale_human_data, but filled in for the joints the
    IK config does NOT scale: each inherits its nearest scaled ancestor's factor (SMPL-X
    joint order guarantees parents precede children), so the drawn skeleton moves as one
    piece instead of tearing apart at unscaled intermediate joints."""
    eff = {}
    for i, name in enumerate(all_names):
        if name in base_scale:
            eff[name] = base_scale[name]
        elif parents[i] >= 0:
            eff[name] = eff[all_names[parents[i]]]
        else:
            eff[name] = 1.0
    return eff


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Solve COLMO retargeting and show the SMPL-X human skeleton in the same MuJoCo viewer."
    )
    parser.add_argument("--smplx_file", required=True, type=str, help="SMPL-X .npz motion file.")
    parser.add_argument("--smplx_folder", type=str, default=str(HERE / ".." / "assets" / "body_models"),
                        help="Folder holding the SMPL-X body models.")
    parser.add_argument("--robot", default="unitree_g1",
                        choices=["unitree_g1", "unitree_h1", "kapex", "unitree_go2", "booster_t1"])
    parser.add_argument("--motion_fps", type=int, default=30,
                        help="Target fps the SMPL-X clip is resampled to (also the playback "
                             "rate cap; the loader's achieved fps is used for the cap when it "
                             "differs, e.g. a source rate that is not an integer multiple).")
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=True,
                        help="Cap playback at motion_fps (on by default).")
    parser.add_argument("--no_rate_limit", action="store_true", default=False,
                        help="Disable the FPS cap.")

    # SMPL-X skeleton drawing options
    parser.add_argument("--draw_joints", choices=["body", "all"], default="body",
                        help="Which SMPL-X joints to draw: 'body' = the 22 body joints "
                             "(pelvis..wrists, the LAFAN1-equivalent set, default); "
                             "'all' = all 55 including jaw, eyes and the 30 hand joints.")
    parser.add_argument("--human_offset", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="World offset (m) applied to the SMPL-X skeleton so it sits next to the robot.")
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="Overlay the SMPL-X skeleton on the robot by aligning "
                             "the (scaled) SMPL-X pelvis with the robot base each "
                             "frame. Ignores --human_offset.")
    parser.add_argument("--hide_axes", action="store_true", default=False,
                        help="Hide the per-joint coordinate frames on the SMPL-X skeleton.")
    parser.add_argument("--axes_size", type=float, default=0.2,
                        help="Length of the per-joint coordinate frame axes (meters).")
    parser.add_argument("--no_follow_camera", action="store_true", default=False,
                        help="Disable camera following the robot base.")
    parser.add_argument("--show_robot", action="store_true", default=False,
                        help="Render the robot mesh too (off by default — only keyframe axes are shown).")
    parser.add_argument("--show_collision", action="store_true", default=False,
                        help="Render the collision-geom group (group 2). Useful for "
                             "checking IK targets against the robot's collision "
                             "shapes (capsules/boxes/spheres declared via "
                             "`<default class=\"cls\">` in the MJCF).")
    parser.add_argument("--no_labels", action="store_true", default=False,
                        help="Hide text labels on robot keyframe-body markers "
                             "and SMPL-X IK-keypoint markers. Use this when "
                             "labels overlap and clutter the view.")
    parser.add_argument("--collision_mode", choices=["cbf", "issf", "off"],
                        default=None,
                        help="Collision avoidance mode. cbf: hard QP inequality "
                             "(mink.CollisionAvoidanceLimit); issf: robustified hard CBF "
                             "(ISSfCollisionAvoidanceLimit); off. Unset -> YAML "
                             "parameters.collision_mode or 'issf'.")
    args = parser.parse_args()

    # --- Load SMPL-X motion -------------------------------------------------------------
    frames, actual_human_height, joints, all_names, parents, edges, src_fps = load_smplx_motion(
        args.smplx_file, args.smplx_folder, tgt_fps=args.motion_fps, draw_joints=args.draw_joints)
    print(f"Loaded {len(frames)} SMPL-X frames @ {src_fps:.1f} fps "
          f"(human_height={actual_human_height:.2f} m, drawing {len(joints)}/{len(all_names)} joints)")

    # --- Initialize retargeter and robot MuJoCo model -----------------------------------
    retargeter = COLMO(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        collision_mode=args.collision_mode,
    )
    # Per-motion base horizontal-speed cap (collision_cfg max_base_horizontal_speed): shrink the
    # root xy scale for this clip only if its base would exceed the cap. No-op if unset. The
    # human overlay reads the retargeter's capped trajectory, so both stay consistent.
    retargeter.adjust_hips_scale_for_motion(frames)

    xml_path = ROBOT_XML_DICT[args.robot]
    robot_base = ROBOT_BASE_DICT[args.robot]
    cam_distance = VIEWER_CAM_DISTANCE_DICT[args.robot]

    # Robot bodies that are IK match targets for the SMPL-X joints — only these get
    # coordinate frames drawn on the robot.
    robot_keyframe_bodies = sorted(set(retargeter.ik_match_table1.keys())
                                   | set(retargeter.ik_match_table2.keys()))
    print(f"Robot keyframe bodies ({len(robot_keyframe_bodies)}): {robot_keyframe_bodies}")

    # IK keypoint SMPL-X joints (= entry[0] in every ik_match_table row). Only
    # these get labels on the SMPL-X skeleton side, to avoid cluttering the
    # viewer with every intermediate joint.
    ik_keypoints = set()
    for table in (retargeter.ik_match_table1, retargeter.ik_match_table2):
        for entry in table.values():
            ik_keypoints.add(entry[0])
    print(f"IK keypoints ({len(ik_keypoints)}): {sorted(ik_keypoints)}")

    h_root_name = retargeter.human_root_name
    base_scale = dict(retargeter.human_scale_table)
    effective_scale = build_effective_scale(all_names, parents, base_scale)

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_step(model, data)

    if not args.show_robot:
        for gid in range(model.ngeom):
            if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
                continue
            if args.show_collision and model.geom_group[gid] == 2:
                model.geom_rgba[gid] = [0.2, 0.8, 1.0, 0.35]
            else:
                model.geom_rgba[gid, 3] = 0.0

    # Suppress labels on MODEL geoms (robot mesh + collision capsules) so the
    # viewer's mjLABEL_GEOM mode only renders the IK-target marker labels we
    # explicitly attach to user_scn geoms. Redirect every name_geomadr entry
    # to a null byte in `model.names` → mj_id2name returns "" → no label.
    names_arr = np.frombuffer(model.names, dtype=np.uint8)
    null_pos = int(np.where(names_arr == 0)[0][0]) if (names_arr == 0).any() else 0
    for gid in range(model.ngeom):
        model.name_geomadr[gid] = null_pos

    # Spacebar pauses/resumes playback. Mutable container so the closure can
    # toggle it from inside MuJoCo's GLFW callback thread.
    paused = [False]

    def _key_callback(keycode):
        if keycode == 32:  # GLFW_KEY_SPACE
            paused[0] = not paused[0]
            print(f"[{'PAUSED' if paused[0] else 'RESUMED'}] (press SPACE to toggle)")

    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False,
                                key_callback=_key_callback)
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    # Enable display of group-2 (collision) geoms only when --show_collision is set.
    viewer.opt.geomgroup[2] = 1 if args.show_collision else 0
    # Keep model-geom labels OFF (mjLABEL_NONE). MuJoCo would otherwise
    # fall back to "geom <id>" auto-labels for unnamed geoms, cluttering the
    # robot. User_scn geom labels are rendered through a separate mechanism
    # and stay visible regardless of `opt.label` — so our IK-target markers
    # (set via `geom.label = ...` in draw_sphere) keep showing.
    viewer.opt.label = mj.mjtLabel.mjLABEL_NONE
    viewer.cam.distance = cam_distance
    viewer.cam.elevation = -10
    # Initial lookat at the robot base. With --no_follow_camera the camera
    # stays anchored here (user can still orbit/pan/zoom with the mouse);
    # without it, this just sets a sensible starting view that will then be
    # overridden every frame by the follow-camera update below.
    viewer.cam.lookat[:] = data.xpos[model.body(robot_base).id]

    human_offset = np.array(args.human_offset, dtype=np.float64)
    rate_limited = args.rate_limit and not args.no_rate_limit
    # Pace on the fps the loader ACHIEVED, not the requested one: get_smplx_data_offline_fast
    # resamples by an integer frame_skip, so the two differ when the source rate is not an
    # integer multiple of --motion_fps, and pacing on the request would drift.
    rate_limiter = RateLimiter(frequency=src_fps, warn=False) if rate_limited else None

    pbar = tqdm(total=len(frames), desc="COLMO + SMPL-X")
    i = 0

    try:
        while viewer.is_running():
            frame = frames[i]

            # 1) Retarget human → robot qpos and push to MuJoCo state.
            if not paused[0]:
                qpos = retargeter.retarget(frame, frame_idx=i)
                data.qpos[:3] = qpos[:3]
                data.qpos[3:7] = qpos[3:7]
                data.qpos[7:] = qpos[7:]
                mj.mj_forward(model, data)

            # 2) Camera follow on the robot base
            if not args.no_follow_camera:
                viewer.cam.lookat = data.xpos[model.body(robot_base).id]

            # 3) Overlay the SMPL-X human skeleton, offset sideways so it does not overlap
            viewer.user_scn.ngeom = 0

            # 3a) Robot keyframe-body coordinate frames + name labels
            # (only IK match targets). A tiny marker sphere at each link
            # origin carries the body name as a geom label so each IK target
            # link is visually identified, mirroring the keypoint labels on
            # the SMPL-X skeleton below.
            for body_name in robot_keyframe_bodies:
                bid = model.body(body_name).id
                if not args.hide_axes:
                    draw_frame_axes(viewer, data.xpos[bid], data.xquat[bid],
                                    size=args.axes_size)
                # Robot IK target markers: blue (matches the "G1 is blue"
                # color convention used by vis_colmo_with_bvh).
                draw_sphere(viewer, data.xpos[bid], radius=0.022,
                            rgba=(0.2, 0.4, 1.0, 1.0),
                            label=None if args.no_labels else body_name)

            # Apply JSON's human_scale_table the same way COLMO.scale_human_data
            # does, but with `effective_scale` so intermediate (non-keypoint)
            # joints also move correctly with their nearest scaled ancestor —
            # otherwise the visualized skeleton would have torn joints.
            raw_root_pos = np.asarray(frame[h_root_name][0])
            scaled_root_pos = effective_scale[h_root_name] * raw_root_pos
            # With the "per_frame" base-speed cap the retargeter's horizontal root comes from
            # a pre-integrated saturated trajectory, not a scalar multiply -- take it from the
            # retargeter so the skeleton travels exactly with the robot. None -> "clip" mode
            # (the scale table itself is already shrunk) or no cap, and the above stands.
            _capped_xy = retargeter.capped_root_xy(i)
            if _capped_xy is not None:
                scaled_root_pos = np.asarray(scaled_root_pos, dtype=float).copy()
                scaled_root_pos[:2] = _capped_xy

            # In --overlay mode, recompute human_offset each frame so the
            # scaled SMPL-X root coincides with the robot base body's current
            # world position. This makes the skeleton draw ON TOP of the
            # robot rather than next to it.
            if args.overlay:
                robot_base_pos = data.xpos[model.body(robot_base).id]
                human_offset = robot_base_pos - scaled_root_pos

            def _scaled_pos(joint_name):
                if joint_name == h_root_name:
                    return scaled_root_pos
                local = (np.asarray(frame[joint_name][0]) - raw_root_pos) \
                    * effective_scale[joint_name]
                return local + scaled_root_pos

            for joint_name in joints:
                pos, quat = frame[joint_name]
                is_keypoint = joint_name in ik_keypoints
                world_pos = _scaled_pos(joint_name) + human_offset
                if joint_name == h_root_name:
                    # Root: a slightly larger orange marker to distinguish
                    # from the regular keypoints.
                    rgba = (1.0, 0.40, 0.20, 1.0)
                    radius = 0.045
                elif is_keypoint:
                    # SMPL-X IK keypoint markers: red.
                    rgba = (1.0, 0.20, 0.20, 1.0)
                    radius = 0.035
                else:
                    # Non-keypoint intermediate joints: small gray.
                    rgba = (0.55, 0.55, 0.55, 1.0)
                    radius = 0.020
                label = joint_name if (is_keypoint and not args.no_labels) else None
                draw_sphere(viewer, world_pos, radius=radius, rgba=rgba,
                            label=label)
                if not args.hide_axes:
                    # Draw the keypoint frame in the ROBOT convention: apply the IK
                    # rot_offset (human joint frame -> robot body frame) so the human
                    # keypoint axes overlap the matching robot key-body axes when
                    # tracking is correct. (Raw SMPL-X axes differ from the robot body
                    # frame by this fixed rot_offset, so any REMAINING divergence is
                    # the actual IK orientation-tracking error.)  Non-keypoint joints
                    # have no rot_offset and are still drawn in their raw SMPL-X frame.
                    draw_quat = quat
                    rot_off = retargeter.rot_offsets1.get(
                        joint_name, retargeter.rot_offsets2.get(joint_name))
                    if rot_off is not None:
                        draw_quat = (R.from_quat(quat, scalar_first=True) * rot_off
                                     ).as_quat(scalar_first=True)
                    draw_frame_axes(viewer, world_pos, draw_quat, size=args.axes_size)

            for child_name, parent_name in edges:
                child_pos = _scaled_pos(child_name) + human_offset
                parent_pos = _scaled_pos(parent_name) + human_offset
                draw_bone(viewer, parent_pos, child_pos)

            # 3c) pos_offset markers (JSON 4th element of each ik_match entry).
            # The actual IK target sent to mink is `scaled_pos + R_target · pos_offset`,
            # where R_target = R_human · R_offset_quat. Draw a magenta sphere at
            # the shifted target on the human side and connect it to the raw
            # keypoint with a thin magenta capsule. Visible only for entries
            # where pos_offset is non-zero (others would just overlap the red
            # keypoint marker).
            for joint_name, pos_off in retargeter.pos_offsets1.items():
                if np.allclose(pos_off, 0.0):
                    continue
                if joint_name not in frame:
                    continue
                raw_quat = np.asarray(frame[joint_name][1])
                R_human = R.from_quat(raw_quat, scalar_first=True)
                R_off = retargeter.rot_offsets1[joint_name]
                R_target = R_human * R_off
                world_offset = R_target.apply(pos_off)
                base_world = _scaled_pos(joint_name) + human_offset
                target_world = base_world + world_offset
                draw_bone(viewer, base_world, target_world,
                          radius=0.006, rgba=(1.0, 0.2, 1.0, 0.9))
                draw_sphere(viewer, target_world, radius=0.028,
                            rgba=(1.0, 0.2, 1.0, 1.0),
                            label=None if args.no_labels else f"{joint_name}+offset")

            viewer.sync()
            if rate_limiter is not None:
                rate_limiter.sleep()

            # When paused, don't advance the frame index or progress bar — the
            # next iteration redraws the same `frames[i]`.
            if paused[0]:
                continue
            pbar.update(1)

            if args.loop:
                i = (i + 1) % len(frames)
                if i == 0:
                    pbar.reset()
            else:
                i += 1
                if i >= len(frames):
                    break
    finally:
        pbar.close()
        viewer.close()
        time.sleep(0.3)
