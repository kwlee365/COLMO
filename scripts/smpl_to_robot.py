"""
Retarget a raw SMPL .npz motion (e.g. gns.npz) to a robot.

Unlike scripts/smplx_to_robot.py, this does NOT need an SMPL/SMPL-X body model:
the npz already contains the forward-kinematics result:
    joints_world: (T, 24, 3)   global joint positions  (SMPL, Y-up)
    pose_rotmat:  (T, 24, 3, 3) LOCAL per-joint rotation matrices

SMPL body joints 0..21 share the same names/indices as SMPL-X, so we reuse the
existing `smplx_to_<robot>.json` IK config (src_human="smplx").

Coordinate frame: SMPL data is Y-up; the MuJoCo robot pipeline is Z-up, so we
apply the same Y->Z conversion used by the GVHMR path. Use --no_axis_convert to
disable it if your npz is already Z-up.
"""

import argparse
import pathlib
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from collision_free_motion_retargeting import CollisionFreeMotionRetargeting as COLMO
from collision_free_motion_retargeting import RobotMotionViewer

from rich import print

# SMPL kinematic tree (24 joints). We only need the body joints (0..21).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
NUM_BODY_JOINTS = 22  # pelvis .. right_wrist (shared between SMPL and SMPL-X)


def load_smpl_npz_frames(smpl_file, tgt_fps=30, y_up_to_z_up=True):
    """Build COLMO frames [{joint_name: (pos(3), quat_wxyz(4))}] from a SMPL npz."""
    data = np.load(smpl_file, allow_pickle=True)
    joints = data["joints_world"].astype(np.float64)   # (T, 24, 3) global positions
    rotmat = data["pose_rotmat"].astype(np.float64)     # (T, 24, 3, 3) local rotations

    if "mocap_framerate" in data.files:
        src_fps = float(data["mocap_framerate"])
    elif "fps" in data.files:
        src_fps = float(data["fps"])
    else:
        src_fps = float(tgt_fps)

    T = joints.shape[0]
    nj = min(NUM_BODY_JOINTS, rotmat.shape[1])
    joint_names = JOINT_NAMES[:nj]

    frame_skip = max(1, int(round(src_fps / tgt_fps))) if tgt_fps < src_fps else 1
    idxs = np.arange(0, T, frame_skip)
    aligned_fps = src_fps / frame_skip

    Rc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64) if y_up_to_z_up \
        else np.eye(3)
    rconv = R.from_matrix(Rc)

    # The npz is not floor-referenced (pelvis/feet can sit far below z=0), unlike
    # the LAFAN1 BVH skeleton. The scale table only multiplies positions, so an
    # ungrounded clip stays underground. Shift every frame by one constant so the
    # frame-0 lowest foot rests on z=0; relative vertical motion is preserved.
    P0 = (Rc @ joints[idxs[0]].T).T  # (24, 3) frame-0 joints, Z-up
    foot_ids = [i for i in (7, 8, 10, 11) if i < nj]  # ankles + toes
    floor_z = float(P0[foot_ids, 2].min()) if foot_ids else float(P0[:nj, 2].min())

    frames = []
    for t in idxs:
        G = [None] * nj
        result = {}
        for i in range(nj):
            Li = rotmat[t, i]
            Gi = Li if SMPL_PARENTS[i] == -1 else G[SMPL_PARENTS[i]] @ Li
            G[i] = Gi
            pos = Rc @ joints[t, i] - np.array([0.0, 0.0, floor_z])
            quat = (rconv * R.from_matrix(Gi)).as_quat(scalar_first=True)  # wxyz
            result[joint_names[i]] = (pos, quat)
        frames.append(result)

    b = data["betas"]
    b0 = float(b[0]) if b.ndim == 1 else float(b[0, 0])
    human_height = 1.66 + 0.1 * b0
    return frames, aligned_fps, human_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl_file", type=str, required=True,
                        help="SMPL .npz with joints_world + pose_rotmat (e.g. gns.npz).")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion (.pkl).")
    parser.add_argument("--loop", action="store_true", help="Loop the motion in the viewer.")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--tgt_fps", type=int, default=30)
    parser.add_argument("--no_axis_convert", action="store_true",
                        help="Skip Y-up -> Z-up conversion (use if npz is already Z-up).")
    args = parser.parse_args()

    smpl_frames, aligned_fps, actual_human_height = load_smpl_npz_frames(
        args.smpl_file, tgt_fps=args.tgt_fps, y_up_to_z_up=not args.no_axis_convert
    )
    print(f"Loaded {len(smpl_frames)} frames at {aligned_fps:.2f} fps, "
          f"human height ~{actual_human_height:.3f} m")

    retarget = COLMO(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{pathlib.Path(args.smpl_file).stem}.mp4",
        camera_follow=True,
    )

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    fps_counter = 0
    fps_start_time = time.time()
    i = 0
    while True:
        if args.loop:
            i = (i + 1) % len(smpl_frames)
        else:
            i += 1
            if i >= len(smpl_frames):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 2.0:
            print(f"Actual rendering FPS: {fps_counter / (current_time - fps_start_time):.2f}")
            fps_counter = 0
            fps_start_time = current_time

        qpos = retarget.retarget(smpl_frames[i])

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            contact_points=retarget.get_contact_point_positions(),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=True,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        import pickle
        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    robot_motion_viewer.close()
