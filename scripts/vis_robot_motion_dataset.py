from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
import sys
import subprocess
from tqdm import tqdm

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1


def keyboard_callback(keycode):
    global paused, motion_id, motion_num
    if chr(keycode) == ' ':
        paused = not paused
    if chr(keycode) == '[':
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == ']':
        motion_id = (motion_id + 1) % motion_num


def list_pkl_files(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pkl")])
    return files


def run_single_motion(robot_type: str, pkl_path: str, record_video: bool, video_path: str, rate_limit: bool, video_width: int = 640, video_height: int = 480):
    # Load one pkl
    (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    ) = load_robot_motion(pkl_path)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=True,
        record_video=record_video,
        video_path=video_path,
        video_width=video_width,
        video_height=video_height,
        keyboard_callback=None,  # batch mode
    )

    try:
        num_frames = len(motion_root_pos)
        for i in range(num_frames):
            env.step(
                motion_root_pos[i],
                motion_root_rot[i],
                motion_dof_pos[i],
                rate_limit=rate_limit,
            )
    finally:
        env.close()


def run_batch_subprocess(robot_type: str, robot_motion_folder: str, video_dir: str, rate_limit: bool, video_width: int = 640, video_height: int = 480):
    os.makedirs(video_dir, exist_ok=True)

    motion_files = list_pkl_files(robot_motion_folder)
    if not motion_files:
        raise FileNotFoundError(f"No .pkl motion files found in {robot_motion_folder}")

    print(f"Found {len(motion_files)} motion files in {robot_motion_folder}. Batch recording to {video_dir}")

    # IMPORTANT: each motion in a fresh process -> avoids black mp4 issue
    for idx, motion_file in enumerate(tqdm(motion_files, desc="Recording", unit="file"), 1):
        pkl_path = os.path.join(robot_motion_folder, motion_file)
        base = os.path.splitext(motion_file)[0]
        out_mp4 = os.path.join(video_dir, f"{base}.mp4")

        # Skip already-recorded videos so an interrupted batch can resume
        if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
            print(f"Skip (already exists): {out_mp4}")
            continue

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode", "single",
            "--robot", robot_type,
            "--pkl_path", pkl_path,
            "--record_video",
            "--video_path", out_mp4,
            "--video_width", str(video_width),
            "--video_height", str(video_height),
        ]
        if rate_limit:
            cmd.append("--rate_limit")

        # Optional but often helps offscreen capture stability on Linux+NVIDIA:
        # env = dict(os.environ, MUJOCO_GL="egl")
        # subprocess.run(cmd, check=True, env=env)

        subprocess.run(cmd, check=True)

    print("Batch recording done.")


def interactive_viewer(robot_type: str, robot_motion_folder: str, record_video: bool, video_path: str, video_width: int = 640, video_height: int = 480):
    global motion_num, motion_id, current_motion_id, paused

    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")

    motion_files = list_pkl_files(robot_motion_folder)
    motion_num = len(motion_files)
    if motion_num == 0:
        raise FileNotFoundError(f"No .pkl motion files found in {robot_motion_folder}")

    print(f"Found {motion_num} motion files in {robot_motion_folder}, loading...")

    motion_dataset = []
    last_motion_fps = None
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        (
            motion_data,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            motion_local_body_pos,
            motion_link_body_list,
        ) = load_robot_motion(motion_path)

        last_motion_fps = motion_fps
        motion_dataset.append({
            "motion_file": motion_file,
            "motion_fps": motion_fps,
            "motion_root_pos": motion_root_pos,
            "motion_root_rot": motion_root_rot,
            "motion_dof_pos": motion_dof_pos,
        })

    print("Loading done.")

    # NOTE: interactive mode = single env, single video
    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=last_motion_fps if last_motion_fps is not None else 30,
        camera_follow=True,
        record_video=record_video,
        video_path=video_path,
        video_width=video_width,
        video_height=video_height,
        keyboard_callback=keyboard_callback
    )

    frame_idx = 0
    try:
        while True:
            if current_motion_id != motion_id:
                current_motion_id = motion_id
                frame_idx = 0
                md = motion_dataset[motion_id]
                print(f"Switched to motion {motion_id}: {md['motion_file']}, fps: {md['motion_fps']}, num_frames: {len(md['motion_root_pos'])}")

            md = motion_dataset[motion_id]
            if not paused:
                env.step(
                    md["motion_root_pos"][frame_idx],
                    md["motion_root_rot"][frame_idx],
                    md["motion_dof_pos"][frame_idx],
                    rate_limit=True
                )
                frame_idx += 1
                if frame_idx >= len(md["motion_root_pos"]):
                    frame_idx = 0
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interactive", "batch", "single"], default="interactive")

    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_folder", type=str, default=None)

    # single mode input
    parser.add_argument("--pkl_path", type=str, default=None)

    # recording
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--video_dir", type=str, default="videos")
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--rate_limit", action="store_true")

    args = parser.parse_args()

    if args.mode == "interactive":
        if args.robot_motion_folder is None:
            raise ValueError("--robot_motion_folder is required for interactive mode")
        interactive_viewer(args.robot, args.robot_motion_folder, args.record_video, args.video_path,
                           video_width=args.video_width, video_height=args.video_height)

    elif args.mode == "batch":
        if args.robot_motion_folder is None:
            raise ValueError("--robot_motion_folder is required for batch mode")
        run_batch_subprocess(args.robot, args.robot_motion_folder, args.video_dir, args.rate_limit,
                             video_width=args.video_width, video_height=args.video_height)

    else:  # single
        if args.pkl_path is None:
            raise ValueError("--pkl_path is required for single mode")
        run_single_motion(args.robot, args.pkl_path, args.record_video, args.video_path, args.rate_limit,
                          video_width=args.video_width, video_height=args.video_height)