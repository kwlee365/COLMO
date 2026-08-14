#!/usr/bin/env python3
"""
PICO / XRoboToolkit body tracking -> robot retargeting (TWIST2-style teleoperation).

Requires the XRoboToolkit PC Service to be running and the `xrobotoolkit_sdk` python
bindings to be installed (see the "PICO Streaming to Robot" section of the README).

    python scripts/xrobot_to_robot.py --robot unitree_g1

Note on real-time performance. Measured on unitree_g1 with moving frames on a desktop
CPU, the stock config (issf, max_iter=10) runs ~8.8 ms/frame (~110 Hz) -- already above
any mocap rate, so the defaults usually need no tuning. If a slower machine falls behind,
the levers are `--max_iter` and the soft limits in collision_cfg.yaml, but both are
modest: max_iter 10 -> 3 gained only ~7% here (the IK convergence early-stop usually
fires before the cap), and turning everything off reached ~6.2 ms (~160 Hz). Note
`--collision_mode cbf` is the same constraint class as issf with the robustness margin
removed, so it is not a speedup. Re-measure on your own machine before tuning.
"""

import argparse
import pickle
import os
import signal
import time

import numpy as np
from rich import print

from collision_free_motion_retargeting import CollisionFreeMotionRetargeting as COLMO
from collision_free_motion_retargeting import RobotMotionViewer, XRobotStreamer

# Global flag for graceful shutdown
g_running = True


def signal_handler(signum, frame):
    global g_running
    print(f"\nReceived signal {signum}, shutting down...")
    g_running = False


def main(args):
    global g_running

    qpos_list = []
    if args.save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # ---- Initialize the XR streamer ----
    # Raises ImportError with install instructions if xrobotoolkit_sdk is missing.
    print("[1/3] Initializing XRoboToolkit streamer...")
    streamer = XRobotStreamer()

    # ---- Initialize retargeter ----
    print("[2/3] Initializing retargeter...")
    retargeter = COLMO(
        src_human="xrobot",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        collision_mode=args.collision_mode,
    )
    # Read only inside _solve_ik_stages, so overriding it post-construction is safe -- and
    # keeps the shared collision_cfg.yaml (used by the offline pipeline) untouched.
    if args.max_iter is not None:
        retargeter.max_iter = args.max_iter

    # ---- Initialize viewer ----
    print("[3/3] Initializing viewer...")
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    print("Starting live retargeting... Press Ctrl+C to stop\n")

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    total_frames = 0
    dropped_frames = 0

    try:
        while g_running:
            human_frame = streamer.get_processed_body_data()

            # No body tracking data this tick (headset not tracking the body yet).
            if not human_frame:
                dropped_frames += 1
                time.sleep(0.001)
                continue

            total_frames += 1

            try:
                qpos = retargeter.retarget(human_frame)
            except Exception as e:
                print(f"Retargeting failed: {e}")
                dropped_frames += 1
                continue

            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data if args.show_human else None,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            if args.save_path is not None:
                qpos_list.append(qpos)

            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(
                    f"FPS: {actual_fps:.1f} | "
                    f"Retargeted: {total_frames} | "
                    f"Dropped: {dropped_frames}"
                )
                fps_counter = 0
                fps_start_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        if args.save_path is not None and qpos_list:
            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            # save from wxyz to xyzw
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])

            motion_data = {
                "fps": args.fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved to {args.save_path}")

        print(f"\nTotal retargeted: {total_frames}")
        print(f"Total dropped: {dropped_frames}")

        robot_motion_viewer.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="PICO / XRoboToolkit body tracking to robot retargeting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument("--human_height", type=float, default=None,
                        help="Actual height of human in meters (optional, for better scaling)")
    parser.add_argument("--collision_mode", type=str, default=None,
                        choices=["issf", "cbf", "off"],
                        help="Collision-avoidance mode. Default: read from collision_cfg.yaml. "
                             "Use 'off' for the fastest loop.")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Override IK iterations per frame (collision_cfg.yaml sets 10). "
                             "The strongest real-time lever: frame cost is near-linear in it.")
    parser.add_argument("--fps", type=int, default=60,
                        help="Assumed streaming rate; also the viewer rate limit and saved motion fps.")
    parser.add_argument("--rate_limit", action="store_true", default=False,
                        help="Limit the viewer to --fps instead of running as fast as possible.")
    parser.add_argument("--show_human", action="store_true", default=False,
                        help="Also draw the scaled human keypoints next to the robot.")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/xrobot_live.mp4")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path of the .pkl to write the retargeted robot motion to.")
    args = parser.parse_args()
    main(args)
