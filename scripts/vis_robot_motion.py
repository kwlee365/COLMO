from collision_free_motion_retargeting import RobotMotionViewer, load_robot_motion
from collision_free_motion_retargeting.params import IK_CONFIG_DICT
import argparse
import json
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
from render_style import (FEET_LOOKAT_Z, STUDIO_FLOORS, add_capsule, add_sphere,
                          apply_shadow, apply_studio, downsample, draw_foot_gap,
                          elevation_for_eye_height, floor_height, label_font_scale,
                          parse_frame_spec, shadow_light_params, skybox_texids)

# Keybody-skeleton style, copied from vis_colmo_with_bvh's human skeleton so robot and
# human figures read the same: root orange, IK keybodies red, other joints small gray,
# bones yellow.
ROOT_RGBA, ROOT_RADIUS = (1.0, 0.40, 0.20, 1.0), 0.045
KEYBODY_RGBA, KEYBODY_RADIUS = (1.0, 0.20, 0.20, 1.0), 0.035
JOINT_RGBA, JOINT_RADIUS = (0.55, 0.55, 0.55, 1.0), 0.020
BONE_RGBA, BONE_RADIUS = (0.9, 0.9, 0.2, 1.0), 0.012

# LAFAN1-BVH-shaped skeleton per robot, as (child, parent) node pairs. A node is a robot
# body, or a synthetic point listed in "points" (the mean of those bodies' origins).
# G1 stand-ins for the LAFAN1 bones: pelvis=Hips, hip_roll=UpLeg, knee=Leg,
# ankle_roll=Foot(Mod), toe=Toe, torso_link=Spine2, neck=Neck, head_mocap=Head,
# shoulder_pitch=Shoulder, shoulder_roll=Arm, elbow=ForeArm, wrist_yaw=Hand. The neck
# sits between the shoulder-pitch origins: G1's torso_link origin is at the waist, so
# wiring the shoulders straight to it would splay them out from the waist.
BVH_STYLE_SKELETON = {
    "unitree_g1": {
        "points": {"neck": ("left_shoulder_pitch_link", "right_shoulder_pitch_link")},
        "edges": [
            ("torso_link", "pelvis"), ("neck", "torso_link"), ("head_mocap", "neck"),
            *[(child, parent) for side in ("left", "right") for child, parent in (
                (f"{side}_hip_roll_link", "pelvis"),
                (f"{side}_knee_link", f"{side}_hip_roll_link"),
                (f"{side}_ankle_roll_link", f"{side}_knee_link"),
                (f"{side}_toe_link", f"{side}_ankle_roll_link"),
                (f"{side}_shoulder_pitch_link", "neck"),
                (f"{side}_shoulder_roll_link", f"{side}_shoulder_pitch_link"),
                (f"{side}_elbow_link", f"{side}_shoulder_roll_link"),
                (f"{side}_wrist_yaw_link", f"{side}_elbow_link"),
            )],
        ],
    },
}


def load_keybodies(ik_config_path):
    """Robot keybodies of an IK config: the keys of ik_match_table1/2."""
    with open(ik_config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return set(cfg.get("ik_match_table1", {})) | set(cfg.get("ik_match_table2", {}))


def build_skeleton(model, robot_type, keybodies):
    """Skeleton to draw: ({node: body ids averaged for its position}, [(child, parent)]).

    Robots in BVH_STYLE_SKELETON get the LAFAN1-shaped skeleton; any other robot falls
    back to the keybodies alone, each wired to its nearest keybody ancestor."""
    spec = BVH_STYLE_SKELETON.get(robot_type)
    if spec is not None:
        nodes = {}
        for edge in spec["edges"]:
            for name in edge:
                if name not in nodes:
                    nodes[name] = [model.body(b).id for b in spec["points"].get(name, (name,))]
        return nodes, list(spec["edges"])

    print(f"[warn] no BVH-style skeleton for {robot_type}; drawing its keybodies only")
    nodes = {name: [model.body(name).id] for name in sorted(keybodies)}
    id_to_name = {ids[0]: name for name, ids in nodes.items()}
    edges = []
    for name, ids in nodes.items():
        parent = int(model.body_parentid[ids[0]])
        while parent > 0 and parent not in id_to_name:
            parent = int(model.body_parentid[parent])
        if parent > 0:
            edges.append((name, id_to_name[parent]))
    return nodes, edges


class FootGap:
    """Per-foot clearance: height of the lowest foot-mesh vertex above the floor plane.

    Same definition as the foot-floating metric in scripts/eval_kinematic.py (the mesh
    geoms on each ``*ankle_roll*`` link against the floor geom's height), so the numbers
    drawn here match the evaluation. Positive = floating, negative = penetrating.
    Per foot it draws a stem from the floor up to that vertex and a labeled dot on the floor.
    """

    def __init__(self, model, thresh):
        self.thresh = float(thresh)
        self.floor_z = floor_height(model)
        self.verts = {"L": [], "R": []}  # side -> [(geom id, mesh vertices in the geom frame)]
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
                continue
            body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            if "ankle_roll" not in body:
                continue
            side = "L" if "left" in body else "R" if "right" in body else None
            if side is None:
                continue
            mid = int(model.geom_dataid[gid])
            adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            self.verts[side].append((gid, model.mesh_vert[adr:adr + num].astype(np.float64)))
        if not (self.verts["L"] and self.verts["R"]):
            raise SystemExit("--foot_gap: no *ankle_roll* foot meshes found in this robot model")
        self.gaps = {"L": [], "R": []}

    def lowest_points(self, data):
        pts = {}
        for side, items in self.verts.items():
            best = None
            for gid, v in items:
                world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                p = world[int(np.argmin(world[:, 2]))]
                if best is None or p[2] < best[2]:
                    best = p
            pts[side] = best
        return pts

    def draw(self, scene, data):
        """Record this frame's gaps and, if a scene is given, draw them into it."""
        for side, p in self.lowest_points(data).items():
            self.gaps[side].append(draw_foot_gap(scene, p, self.floor_z, self.thresh))

    def summary(self):
        g = {k: np.asarray(v) for k, v in self.gaps.items()}
        if not len(g["L"]):
            return
        t = self.thresh
        lower = np.minimum(g["L"], g["R"])
        print(f"\nFoot clearance (lowest ankle_roll mesh vertex above the floor; "
              f"threshold {100 * t:.1f} cm, {len(lower)} frames):")
        for side, name in (("L", "left"), ("R", "right")):
            print(f"  {name:<6} mean {100 * g[side].mean():+6.2f} cm | floating "
                  f"{100 * (g[side] > t).mean():5.1f}% | penetrating {100 * (g[side] < -t).mean():5.1f}%")
        print(f"  lower foot mean {100 * lower.mean():+6.2f} cm | both feet off the ground in "
              f"{100 * (lower > t).mean():5.1f}% of frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")

    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str,
                        default="videos/example.mp4")

    # Capture (video / PNG stills). Both render OFFSCREEN ONLY -- no live window --
    # and share the resolution / anti-aliasing / shadow settings below.
    parser.add_argument("--snapshot_frames", type=parse_frame_spec, nargs="+", default=None,
                        help="Save lossless PNG stills of these frames (0-based) to --snapshot_dir. "
                             "Each token is N, A-B or A-B:STEP (e.g. --snapshot_frames 0 60 100-200:20).")
    parser.add_argument("--snapshot_dir", type=str, default="figures")
    parser.add_argument("--video_width", type=int, default=1280,
                        help="Capture width in pixels (e.g. 1920 for FHD, 3840 for 4K).")
    parser.add_argument("--video_height", type=int, default=720,
                        help="Capture height in pixels (e.g. 1080 for FHD, 2160 for 4K).")
    parser.add_argument("--msaa", type=int, default=8,
                        help="Anti-aliasing samples (MuJoCo offsamples). 8 or 16.")
    parser.add_argument("--video_quality", type=int, default=8,
                        help="imageio/ffmpeg quality (0-10; higher = less compression, bigger file).")
    parser.add_argument("--shadow", action="store_true",
                        help="Cast ground shadows (live viewer and capture). The G1 floor ships "
                             "semi-transparent and MuJoCo draws transparent geoms in a pass with no "
                             "shadow map, so this makes the floor opaque and lights the robot from "
                             "the camera's side (--cam_azimuth sets which side).")
    parser.add_argument("--foot_gap", action="store_true",
                        help="Overlay each foot's clearance above the floor (live viewer and "
                             "capture): a stem from the floor to the lowest ankle_roll mesh vertex "
                             "and a dot labeled in cm. Gray = on the ground, amber = floating, "
                             "red = penetrating, split at --foot_gap_thresh. Same definition as "
                             "eval_kinematic's foot floating; prints a per-foot summary at the end.")
    parser.add_argument("--foot_gap_thresh", type=float, default=0.01,
                        help="|clearance| [m] within which a foot counts as on the ground (default 1 cm).")
    parser.add_argument("--shadow_size", type=int, default=8192,
                        help="Shadow map resolution used with --shadow. Try 4096 / 8192 / 16384.")
    parser.add_argument("--shadow_skew", type=float, default=None,
                        help="How far --shadow throws the shadow to the side, as a fraction of the "
                             "straight-back direction (0 = straight away from the camera). "
                             "Default 0.45, or 0.15 with --studio.")
    parser.add_argument("--studio", action="store_true",
                        help="Paper-figure look (implies --shadow): white background with no blue "
                             "horizon band, near-white floor faded into the sky, and a softer, higher "
                             "key light from nearly behind the camera so the robot's limbs throw fewer "
                             "hard-edged shadows onto itself.")
    parser.add_argument("--studio_floor", choices=sorted(STUDIO_FLOORS), default="light",
                        help="Floor tone for --studio: light (near-white, default) or gray "
                             "(contrasts with the white background; use for floor-level shots).")
    parser.add_argument("--supersample", type=int, default=1,
                        help="Capture only: render at N x the output size and downscale (Lanczos). "
                             "2 gives visibly cleaner edges and shadow borders; GPU memory grows with "
                             "N^2 (4K at 2 renders 7680x4320).")
    parser.add_argument("--cam_distance", type=float, default=None,
                        help="Capture camera distance (default: the robot's viewer distance).")
    parser.add_argument("--cam_azimuth", type=float, default=90.0,
                        help="Capture camera azimuth in degrees (90 = MuJoCo viewer default).")
    parser.add_argument("--cam_elevation", type=float, default=-10.0)
    parser.add_argument("--cam_eye_height", type=float, default=None,
                        help="Capture only: put the camera this many meters above the floor "
                             "(overrides --cam_elevation, which is then derived from it). A few "
                             "cm, with --cam_target feet and --studio_floor gray, gives a "
                             "floor-level view where a floating sole shows background under it.")
    parser.add_argument("--cam_target", choices=["base", "feet"], default="base",
                        help="Capture camera look-at point. base (default): the robot base, like "
                             "the live viewer. feet: between the two ankle_roll links at "
                             f"{FEET_LOOKAT_Z} m height -- a ground close-up for judging foot "
                             "floating (pair with e.g. --cam_distance 1.0 --cam_elevation -4).")
    parser.add_argument("--keybody_skeleton", action="store_true",
                        help="Capture only a stick-figure skeleton in vis_colmo_with_bvh's "
                             "human-skeleton style (root orange, IK keybodies red, other joints "
                             "gray, bones yellow), LAFAN1-shaped with neck and head for G1; the "
                             "robot mesh is hidden. Capture only (needs --snapshot_frames or "
                             "--record_video).")
    parser.add_argument("--keybody_src", type=str, default="smplx",
                        choices=sorted(IK_CONFIG_DICT.keys()),
                        help="IK config whose keybodies --keybody_skeleton colors red (pkls made "
                             "by smplx_to_robot.py use smplx).")

    args = parser.parse_args()
    if args.supersample < 1:
        parser.error("--supersample must be >= 1")
    if args.studio:
        args.shadow = True
    shadow_skew, shadow_drop = shadow_light_params(args.studio, args.shadow_skew)

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    # Shadow map coverage: where the root starts and how far it travels in this clip.
    root_xy = np.asarray(motion_root_pos)[:, :2]
    root_travel = float(np.max(np.linalg.norm(root_xy - root_xy[0], axis=1)))

    snapshot_set = sorted({i for spec in (args.snapshot_frames or []) for i in spec})
    if args.keybody_skeleton and not (args.record_video or snapshot_set):
        parser.error("--keybody_skeleton is capture-only; add --snapshot_frames or --record_video")
    if (args.cam_target != "base" or args.cam_eye_height is not None) \
            and not (args.record_video or snapshot_set):
        parser.error("--cam_target / --cam_eye_height are capture-only; "
                     "add --snapshot_frames or --record_video")

    if not (args.record_video or snapshot_set):
        env = RobotMotionViewer(robot_type=robot_type,
                                motion_fps=motion_fps,
                                camera_follow=True)
        # Floor color, light and fog settings are read every frame, so editing the
        # viewer's model after launch still applies.
        if args.studio:
            from collision_free_motion_retargeting import VIEWER_CAM_DISTANCE_DICT
            apply_studio(env.model, VIEWER_CAM_DISTANCE_DICT[robot_type], args.studio_floor)
            # The window uploaded its textures at launch; push the whitened skybox.
            for t in skybox_texids(env.model):
                env.viewer.update_texture(t)
            env.viewer.user_scn.flags[mj.mjtRndFlag.mjRND_FOG] = True  # copied to the main scene
        if args.shadow:
            apply_shadow(env.model, root_xy[0], root_travel, args.cam_azimuth, shadow_skew,
                         shadow_drop, args.shadow_size, robot_type)
        foot = FootGap(env.model, args.foot_gap_thresh) if args.foot_gap else None
        probe = mj.MjData(env.model) if foot is not None else None

        for frame_idx in range(len(motion_root_pos)):
            if foot is not None:
                # Measure this frame's pose on a scratch MjData and draw into user_scn;
                # step() then syncs, so the overlay and the robot show the same frame.
                probe.qpos[:3] = motion_root_pos[frame_idx]
                probe.qpos[3:7] = motion_root_rot[frame_idx]
                probe.qpos[7:] = motion_dof_pos[frame_idx]
                mj.mj_kinematics(env.model, probe)
                with env.viewer.lock():
                    env.viewer.user_scn.ngeom = 0
                    foot.draw(env.viewer.user_scn, probe)
            env.step(motion_root_pos[frame_idx],
                    motion_root_rot[frame_idx],
                    motion_dof_pos[frame_idx],
                    rate_limit=True)
        env.close()
        if foot is not None:
            foot.summary()
    else:
        # Capture renders offscreen only: running the live passive viewer next to an
        # offscreen mj.Renderer makes the two fight over the GL context (the old
        # --record_video path hung on close under Wayland). For a headless GL
        # context, run with MUJOCO_GL=egl.
        import imageio
        from collision_free_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT

        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_type]))
        data = mj.MjData(model)
        # Supersampling renders at N x the output size; the frame is downscaled on save.
        render_w, render_h = args.video_width * args.supersample, args.video_height * args.supersample
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), render_w)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), render_h)
        model.vis.quality.offsamples = max(int(model.vis.quality.offsamples), args.msaa)
        cam_distance = args.cam_distance or VIEWER_CAM_DISTANCE_DICT[robot_type]

        if args.studio:
            apply_studio(model, cam_distance, args.studio_floor)
        if args.shadow:
            apply_shadow(model, root_xy[0], root_travel, args.cam_azimuth, shadow_skew,
                         shadow_drop, args.shadow_size, robot_type)
        foot = FootGap(model, args.foot_gap_thresh) if args.foot_gap else None

        base_name = ROBOT_BASE_DICT[robot_type]
        skel_nodes, skel_edges, keybodies = {}, [], set()
        if args.keybody_skeleton:
            ik_config_path = IK_CONFIG_DICT[args.keybody_src][robot_type]
            keybodies = load_keybodies(ik_config_path)
            skel_nodes, skel_edges = build_skeleton(model, robot_type, keybodies)
            print(f"Keybody skeleton: {len(skel_nodes)} joints "
                  f"({sum(n in keybodies for n in skel_nodes)} keybodies from {ik_config_path.name}), "
                  f"{len(skel_edges)} bones")
            # Hide the robot mesh: geoms with alpha 0 are left out of the scene entirely
            # (so they cast no shadow either); the floor stays. Sites too -- the G1 IMU
            # sites sit inside the mesh and would otherwise show up as stray gray dots.
            for gid in range(model.ngeom):
                if model.geom_type[gid] != mj.mjtGeom.mjGEOM_PLANE:
                    model.geom_rgba[gid, 3] = 0.0
            model.site_rgba[:, 3] = 0.0

        renderer = mj.Renderer(model, height=render_h, width=render_w,
                               font_scale=label_font_scale(render_h))
        renderer.scene.flags[mj.mjtRndFlag.mjRND_FOG] = args.studio

        opt = mj.MjvOption()
        opt.geomgroup[2] = 0  # hide collision geoms, as RobotMotionViewer does
        cam = mj.MjvCamera()
        cam.distance = cam_distance
        cam.azimuth = args.cam_azimuth
        cam.elevation = args.cam_elevation
        base_id = model.body(base_name).id
        feet_ids = [b for b in range(model.nbody)
                    if "ankle_roll" in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or "")]
        if args.cam_target == "feet" and len(feet_ids) != 2:
            raise SystemExit(f"--cam_target feet: expected 2 *ankle_roll* bodies, found {len(feet_ids)}")

        num_frames = len(motion_root_pos)
        late = [i for i in snapshot_set if i >= num_frames]
        if late:
            print(f"[warn] snapshot frame(s) {late} are past the end ({num_frames} frames); skipped")

        writer = None
        if args.record_video:
            os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
            # macro_block_size=None keeps e.g. 1080-pixel heights as-is instead of
            # letting ffmpeg resize them to a multiple of 16.
            writer = imageio.get_writer(args.video_path, fps=motion_fps,
                                        quality=args.video_quality, macro_block_size=None)
        if snapshot_set:
            os.makedirs(args.snapshot_dir, exist_ok=True)
        # With snapshots only, nothing needs rendering past the last requested frame.
        last_frame = num_frames if writer else min(num_frames, snapshot_set[-1] + 1)
        motion_name = os.path.splitext(os.path.basename(robot_motion_path))[0]
        if args.keybody_skeleton:
            motion_name += "_keybody"

        for frame_idx in tqdm(range(last_frame)):
            data.qpos[:3] = motion_root_pos[frame_idx]
            data.qpos[3:7] = motion_root_rot[frame_idx]
            data.qpos[7:] = motion_dof_pos[frame_idx]
            mj.mj_forward(model, data)
            if writer is None and frame_idx not in snapshot_set:
                # Snapshot-only: render just the requested frames (a 4K x2 supersampled
                # render is ~0.3 s), but keep measuring for the --foot_gap summary.
                if foot is not None:
                    foot.draw(None, data)
                continue
            if args.cam_target == "feet":
                cam.lookat[:2] = data.xpos[feet_ids, :2].mean(axis=0)
                cam.lookat[2] = FEET_LOOKAT_Z
            else:
                cam.lookat[:] = data.xpos[base_id]  # follow the base, like the live viewer
            if args.cam_eye_height is not None:
                cam.elevation = elevation_for_eye_height(
                    args.cam_eye_height, cam.lookat, cam.distance, floor_height(model))

            renderer.update_scene(data, camera=cam, scene_option=opt)
            if foot is not None:
                foot.draw(renderer.scene, data)
            if args.keybody_skeleton:
                scene = renderer.scene
                node_pos = {name: data.xpos[ids].mean(axis=0) for name, ids in skel_nodes.items()}
                for child, parent in skel_edges:
                    add_capsule(scene, node_pos[parent], node_pos[child], BONE_RADIUS, BONE_RGBA)
                for name, pos in node_pos.items():
                    if name == base_name:
                        add_sphere(scene, pos, ROOT_RADIUS, ROOT_RGBA)
                    elif name in keybodies:
                        add_sphere(scene, pos, KEYBODY_RADIUS, KEYBODY_RGBA)
                    else:
                        add_sphere(scene, pos, JOINT_RADIUS, JOINT_RGBA)
            img = downsample(renderer.render(), args.video_width, args.video_height)
            if writer is not None:
                writer.append_data(img)
            if frame_idx in snapshot_set:
                out = os.path.join(args.snapshot_dir, f"{motion_name}_{frame_idx:05d}.png")
                imageio.imwrite(out, img)
                print(f"snapshot -> {out}")

        renderer.close()
        if writer is not None:
            writer.close()
            print(f"Video saved to {args.video_path}")
        if foot is not None:
            foot.summary()
