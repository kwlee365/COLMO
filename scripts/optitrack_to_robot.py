"""OptiTrack (Motive / NatNet) live streaming -> robot retargeting.

Real-time teleoperation entry point. Motive runs on a Windows "server" machine and
streams skeleton data over NatNet; this script runs on the "client" machine (the one
with COLMO installed) and retargets every incoming frame to the robot.

    python scripts/optitrack_to_robot.py --server_ip <server_ip> --client_ip <client_ip> \
        --use_multicast False --robot unitree_g1

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
import threading

from general_motion_retargeting import GeneralMotionRetargeting as COLMO
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack


def str2bool(v):
    """argparse `type=bool` is a trap: bool("False") is True, so `--use_multicast False`
    would silently switch the client INTO multicast -- the opposite of what it reads like,
    and the opposite of the Unicast transmission type the README's Motive screenshot shows.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {v!r}")


def main(args):
    # Check if firewall is disabled on this machine
    print("Make sure to disable firewall on both machines:")
    print("On OptiTrack computer: Disable Windows Firewall")
    print("On this computer: sudo ufw disable")

    client = setup_optitrack(
        server_address=args.server_ip,
        client_address=args.client_ip,
        use_multicast=args.use_multicast,
    )

    if not client:
        print("Failed to setup OptiTrack client")
        exit(1)

    # start a thread to client.run()
    thread = threading.Thread(target=client.run, daemon=True)
    thread.start()

    print(f"OptiTrack client connected: {client.connected()}")
    print("Starting motion retargeting...")

    retarget = COLMO(
        src_human="fbx",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        collision_mode=args.collision_mode,
    )
    # Read only inside _solve_ik_stages, so overriding it post-construction is safe -- and
    # keeps the shared collision_cfg.yaml (used by the offline pipeline) untouched.
    if args.max_iter is not None:
        retarget.max_iter = args.max_iter
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    try:
        while True:
            frame = client.get_frame()
            qpos = retarget.retarget(frame)
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                rate_limit=False,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.200.160")
    parser.add_argument("--client_ip", type=str, default="192.168.200.117")
    parser.add_argument("--use_multicast", type=str2bool, default=False,
                        help="Must match Motive's Streaming > Transmission Type. "
                             "Leave False for Unicast (what the README screenshot shows).")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1"])
    parser.add_argument("--human_height", type=float, default=None,
                        help="Actual height of the performer in meters. Left unset, no height "
                             "scaling is applied (matches the other two teleop scripts).")
    parser.add_argument("--collision_mode", type=str, default=None,
                        choices=["issf", "cbf", "off"],
                        help="Collision-avoidance mode. Default: read from collision_cfg.yaml. "
                             "Use 'off' for the fastest loop.")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Override IK iterations per frame (collision_cfg.yaml sets 10). "
                             "The strongest real-time lever: frame cost is near-linear in it.")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/optitrack_live.mp4")
    args = parser.parse_args()
    main(args)
