import argparse
import time
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    ROBOT_XML_DICT,
    ROBOT_BASE_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve GMR retargeting for G1 and show BVH human skeleton in the same MuJoCo viewer."
    )
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file.")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--robot", default="unitree_g1",
                        choices=["unitree_g1", "unitree_h1"])
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--rate_limit", action="store_true", default=True,
                        help="Cap playback at motion_fps (on by default).")
    parser.add_argument("--no_rate_limit", action="store_true", default=False,
                        help="Disable the FPS cap.")

    # BVH skeleton drawing options
    parser.add_argument("--human_offset", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="World offset (m) applied to the BVH skeleton so it sits next to the robot.")
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="Overlay the BVH skeleton on the robot by aligning "
                             "the (scaled) BVH Hips with the robot base each "
                             "frame. Ignores --human_offset.")
    parser.add_argument("--hide_axes", action="store_true", default=False,
                        help="Hide the per-joint coordinate frames on the BVH skeleton.")
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
                             "and BVH IK-keypoint markers. Use this when "
                             "labels overlap and clutter the view.")
    parser.add_argument("--collision_mode", choices=["cbf", "issf", "off"],
                        default=None,
                        help="Collision avoidance mode. cbf: hard QP inequality "
                             "(mink.CollisionAvoidanceLimit); issf: robustified hard CBF "
                             "(ISSfCollisionAvoidanceLimit); off. Unset -> YAML "
                             "parameters.collision_mode or 'issf'.")
    args = parser.parse_args()

    # --- Load BVH motion ----------------------------------------------------------------
    frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    raw = read_bvh(args.bvh_file)
    bones = list(raw.bones)
    parents = list(raw.parents)
    edges = [(bones[i], bones[p]) for i, p in enumerate(parents)
             if p >= 0 and bones[i] in frames[0]]
    print(f"Loaded {len(frames)} BVH frames "
          f"(human_height={actual_human_height:.2f} m, {len(bones)} bones)")

    # --- Initialize retargeter and robot MuJoCo model -----------------------------------
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        collision_mode=args.collision_mode,
    )

    xml_path = ROBOT_XML_DICT[args.robot]
    robot_base = ROBOT_BASE_DICT[args.robot]
    cam_distance = VIEWER_CAM_DISTANCE_DICT[args.robot]

    # Robot bodies that are IK match targets for the BVH joints — only these get
    # coordinate frames drawn on the robot.
    robot_keyframe_bodies = sorted(set(retargeter.ik_match_table1.keys())
                                   | set(retargeter.ik_match_table2.keys()))
    print(f"Robot keyframe bodies ({len(robot_keyframe_bodies)}): {robot_keyframe_bodies}")

    # IK keypoint BVH bones (= entry[0] in every ik_match_table row). Only
    # these get labels on the BVH skeleton side, to avoid cluttering the
    # viewer with every intermediate bone.
    ik_keypoints = set()
    for table in (retargeter.ik_match_table1, retargeter.ik_match_table2):
        for entry in table.values():
            ik_keypoints.add(entry[0])
    print(f"IK keypoints ({len(ik_keypoints)}): {sorted(ik_keypoints)}")

    h_root_name = retargeter.human_root_name
    base_scale = dict(retargeter.human_scale_table)
    effective_scale = {}
    for idx, b in enumerate(bones):
        if b in base_scale:
            effective_scale[b] = base_scale[b]
        elif parents[idx] >= 0:
            effective_scale[b] = effective_scale[bones[parents[idx]]]
        else:
            effective_scale[b] = 1.0

    for side in ("Left", "Right"):
        mod = f"{side}FootMod"
        if mod not in effective_scale:
            effective_scale[mod] = base_scale.get(
                mod, effective_scale.get(f"{side}Foot", 1.0))

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
    rate_limiter = RateLimiter(frequency=args.motion_fps, warn=False) if rate_limited else None

    pbar = tqdm(total=len(frames), desc="GMR + BVH")
    i = 0

    try:
        while viewer.is_running():
            frame = frames[i]

            # 1) Retarget human → robot qpos and push to MuJoCo state.
            if not paused[0]:
                qpos = retargeter.retarget(frame)
                data.qpos[:3] = qpos[:3]
                data.qpos[3:7] = qpos[3:7]
                data.qpos[7:] = qpos[7:]
                mj.mj_forward(model, data)

            # 2) Camera follow on the robot base
            if not args.no_follow_camera:
                viewer.cam.lookat = data.xpos[model.body(robot_base).id]

            # 3) Overlay BVH human skeleton, offset sideways so it does not overlap
            viewer.user_scn.ngeom = 0

            # 3a) Robot keyframe-body coordinate frames + name labels
            # (only IK match targets). A tiny green marker sphere at each
            # link origin carries the body name as a geom label so each IK
            # target link is visually identified, mirroring the yellow
            # keypoint labels on the BVH skeleton below.
            for body_name in robot_keyframe_bodies:
                bid = model.body(body_name).id
                if not args.hide_axes:
                    draw_frame_axes(viewer, data.xpos[bid], data.xquat[bid],
                                    size=args.axes_size)
                # Robot IK target markers: blue (matches the "G1 is blue"
                # color convention requested by the user).
                draw_sphere(viewer, data.xpos[bid], radius=0.022,
                            rgba=(0.2, 0.4, 1.0, 1.0),
                            label=None if args.no_labels else body_name)

            # Apply JSON's human_scale_table the same way GMR.scale_human_data
            # does, but with `effective_scale` so intermediate (non-keypoint)
            # bones also move correctly with their nearest scaled ancestor —
            # otherwise the visualized skeleton would have torn joints.
            raw_root_pos = np.asarray(frame[h_root_name][0])
            scaled_root_pos = effective_scale[h_root_name] * raw_root_pos

            # In --overlay mode, recompute human_offset each frame so the
            # scaled BVH root coincides with the robot base body's current
            # world position. This makes the skeleton draw ON TOP of the
            # robot rather than next to it.
            if args.overlay:
                robot_base_pos = data.xpos[model.body(robot_base).id]
                human_offset = robot_base_pos - scaled_root_pos

            def _scaled_pos(bone_name):
                if bone_name == h_root_name:
                    return scaled_root_pos
                local = (np.asarray(frame[bone_name][0]) - raw_root_pos) \
                    * effective_scale[bone_name]
                return local + scaled_root_pos

            for bone_name, (pos, quat) in frame.items():
                is_keypoint = bone_name in ik_keypoints
                # Mod bones (LeftFootMod / RightFootMod) are synthetic IK
                # keypoints injected by the loader — they ARE in the BVH
                # frame but are NOT in the BVH HIERARCHY. Skip them only
                # when they are NOT IK keypoints; otherwise draw and label.
                if bone_name.endswith("Mod") and not is_keypoint:
                    continue
                world_pos = _scaled_pos(bone_name) + human_offset
                if bone_name == "Hips":
                    # Root: a slightly larger orange marker to distinguish
                    # from the regular keypoints.
                    rgba = (1.0, 0.40, 0.20, 1.0)
                    radius = 0.045
                elif is_keypoint:
                    # BVH IK keypoint markers: red.
                    rgba = (1.0, 0.20, 0.20, 1.0)
                    radius = 0.035
                else:
                    # Non-keypoint intermediate bones: small gray.
                    rgba = (0.55, 0.55, 0.55, 1.0)
                    radius = 0.020
                label = bone_name if (is_keypoint and not args.no_labels) else None
                draw_sphere(viewer, world_pos, radius=radius, rgba=rgba,
                            label=label)
                if not args.hide_axes:
                    draw_frame_axes(viewer, world_pos, quat, size=args.axes_size)

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
            for bone_name, pos_off in retargeter.pos_offsets1.items():
                if np.allclose(pos_off, 0.0):
                    continue
                if bone_name not in frame:
                    continue
                raw_quat = np.asarray(frame[bone_name][1])
                R_human = R.from_quat(raw_quat, scalar_first=True)
                R_off = retargeter.rot_offsets1[bone_name]
                R_target = R_human * R_off
                world_offset = R_target.apply(pos_off)
                base_world = _scaled_pos(bone_name) + human_offset
                target_world = base_world + world_offset
                draw_bone(viewer, base_world, target_world,
                          radius=0.006, rgba=(1.0, 0.2, 1.0, 0.9))
                draw_sphere(viewer, target_world, radius=0.028,
                            rgba=(1.0, 0.2, 1.0, 1.0),
                            label=None if args.no_labels else f"{bone_name}+offset")

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
