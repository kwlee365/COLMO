#!/usr/bin/env python3
"""
Xsens MVN Live Streaming to Robot Retargeting.
Real-time motion capture retargeting following bvh_xsens_to_robot.py format.

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
import os
import signal
import sys
import time

import numpy as np
from rich import print

from collision_free_motion_retargeting import CollisionFreeMotionRetargeting as COLMO
from collision_free_motion_retargeting import RobotMotionViewer
from collision_free_motion_retargeting.utils.xsens_vendor.xsens_to_colmo_adapter import XsensToCOLMO

# Global flag for graceful shutdown
g_running = True


def signal_handler(signum, frame):
    global g_running
    print(f"\nReceived signal {signum}, shutting down...")
    g_running = False


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Xsens MVN Live Streaming to Robot Retargeting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9763,
        help="UDP port for Xsens MVN streaming",
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--human_height",
        type=float,
        default=None,
        help="Actual height of human in meters (optional, for better scaling)",
    )

    parser.add_argument(
        "--collision_mode",
        type=str,
        default=None,
        choices=["issf", "cbf", "off"],
        help="Collision-avoidance mode. Default: read from collision_cfg.yaml. "
             "Use 'off' for the fastest loop.",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Override IK iterations per frame (collision_cfg.yaml sets 10). "
             "The strongest real-time lever: frame cost is near-linear in it.",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/xsens_live.mp4",
    )

    # BooleanOptionalAction, not store_true: with store_true + default=True the flag can
    # only ever set True something already True, so the viewer could never be un-throttled
    # from the CLI. `--no-rate_limit` lets the loop run as fast as the stream allows.
    parser.add_argument(
        "--rate_limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace the viewer to the target fps. Use --no-rate_limit to run unthrottled.",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path of the .pkl to write the retargeted robot motion to. "
             "If not set, no file is saved.",
    )

    args = parser.parse_args()

    target_fps = 60  # Xsens MVN typical streaming rate

    qpos_list = []
    if args.save_path is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save_path))
        os.makedirs(save_dir, exist_ok=True)

    # ---- Initialize Xsens adapter ----
    print("[1/3] Initializing Xsens adapter...")
    xsens = XsensToCOLMO(port=args.port, verbose=True)
    if not xsens.initialize():
        print("Failed to initialize Xsens adapter")
        sys.exit(1)

    # ---- Initialize retargeter ----
    print("[2/3] Initializing retargeter...")
    retargeter = COLMO(
        src_human="xsens_mvn",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        use_velocity_limit=True,
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
        motion_fps=target_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    # ---- Start streaming ----
    xsens.start()
    time.sleep(1.0)  # Wait for stream to stabilize

    print(f"mocap_frame_rate: {target_fps}")
    print("Starting live retargeting... Press Ctrl+C to stop\n")

    # FPS measurement
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    total_frames = 0
    dropped_frames = 0
    last_valid_qpos = None
    last_valid_human_frame = None

    try:
        while g_running:
            human_frame = xsens.get_human_frame()

            if human_frame is None:
                dropped_frames += 1
                # Show last valid pose if available
                if last_valid_qpos is not None:
                    robot_motion_viewer.step(
                        root_pos=last_valid_qpos[:3],
                        root_rot=last_valid_qpos[3:7],
                        dof_pos=last_valid_qpos[7:],
                        human_motion_data=last_valid_human_frame,
                        rate_limit=args.rate_limit,
                    )
                time.sleep(0.001)
                continue

            total_frames += 1

            # Retarget
            try:
                qpos = retargeter.retarget(human_frame)
            except Exception as e:
                print(f"Retargeting failed: {e}")
                dropped_frames += 1
                continue

            last_valid_qpos = qpos.copy()
            last_valid_human_frame = retargeter.scaled_human_data

            # Visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            if args.save_path is not None:
                qpos_list.append(qpos)

            # FPS measurement
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
        # Stop streaming
        xsens.stop()

        # Save trajectory
        if args.save_path is not None and qpos_list:
            import pickle
            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            # save from wxyz to xyzw
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            local_body_pos = None
            body_names = None

            motion_data = {
                "fps": target_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos,
                "link_body_list": body_names,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved to {args.save_path}")

        # Print final stats
        print(f"\nTotal retargeted: {total_frames}")
        print(f"Total dropped: {dropped_frames}")

        robot_motion_viewer.close()
