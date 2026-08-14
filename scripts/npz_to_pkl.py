"""
Convert a holosoma-generated .npz motion file into the COLMO .pkl format
so it can be visualized with `scripts/vis_robot_motion.py`.

Holosoma npz layout (e.g. holosoma/lafan/*.npz):
    qpos:         (N, 7 + n_dof) float64   # MuJoCo qpos: [pos(3), quat_wxyz(4), dof(n_dof)]
    human_joints: (N, 22, 3) float64
    fps:          () int64
    cost:         () float64

The number of DOF depends on the robot (unitree_g1: 29 -> qpos (N, 36);
kapex: 33 -> qpos (N, 40)). The conversion itself is robot-agnostic slicing;
pass --robot to validate the qpos width against the target robot model.

COLMO pkl layout (see scripts/bvh_to_robot.py and data_loader.py):
    fps:            int
    root_pos:       (N, 3)
    root_rot:       (N, 4)   # xyzw
    dof_pos:        (N, n_dof)
    local_body_pos: None or (N, n_links, 3)
    link_body_list: None or list[str]
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def get_robot_n_dof(robot: str) -> int:
    """Return the number of actuated DOF (qpos width minus the 7-dof free base)
    for a registered robot, by loading its MuJoCo model."""
    import mujoco

    from general_motion_retargeting.params import ROBOT_XML_DICT

    if robot not in ROBOT_XML_DICT:
        raise ValueError(
            f"Unknown robot '{robot}'. Choices: {sorted(ROBOT_XML_DICT.keys())}"
        )
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot].resolve()))
    return model.nq - 7


def convert_one(npz_path: str, pkl_path: str, expected_n_dof: int = None) -> None:
    data = np.load(npz_path, allow_pickle=True)

    qpos = data["qpos"]
    fps = int(data["fps"])

    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError(
            f"Unexpected qpos shape {qpos.shape} in {npz_path}; expected (N, >=7)."
        )

    n_dof = qpos.shape[1] - 7
    if expected_n_dof is not None and n_dof != expected_n_dof:
        raise ValueError(
            f"qpos in {npz_path} has {n_dof} dof (shape {qpos.shape}), but the target "
            f"robot expects {expected_n_dof} dof. Check that --robot matches the npz."
        )

    root_pos = qpos[:, :3].astype(np.float64)
    # MuJoCo stores quaternions as wxyz; COLMO pkl stores them as xyzw.
    root_rot_wxyz = qpos[:, 3:7]
    root_rot = root_rot_wxyz[:, [1, 2, 3, 0]].astype(np.float64)
    dof_pos = qpos[:, 7:].astype(np.float64)

    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"[ok] {npz_path} -> {pkl_path}  (frames={len(root_pos)}, dof={dof_pos.shape[1]}, fps={fps})")


def main():
    parser = argparse.ArgumentParser(description="Convert holosoma .npz motion files to COLMO .pkl format.")
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to a .npz file or a directory containing .npz files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Output path. If --npz_path is a file, this is the output .pkl path "
            "(default: same name with .pkl extension). If --npz_path is a directory, "
            "this is the output directory (default: same directory, .pkl alongside .npz)."
        ),
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help=(
            "Target robot name (e.g. unitree_g1, kapex). If given, the qpos width in "
            "each npz is validated against that robot's expected DOF count."
        ),
    )
    args = parser.parse_args()

    expected_n_dof = get_robot_n_dof(args.robot) if args.robot else None

    src = Path(args.npz_path)
    if src.is_file():
        if src.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got {src}")
        dst = Path(args.save_path) if args.save_path else src.with_suffix(".pkl")
        convert_one(str(src), str(dst), expected_n_dof)
    elif src.is_dir():
        out_dir = Path(args.save_path) if args.save_path else src
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_files = sorted(src.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {src}")
        for f in npz_files:
            convert_one(str(f), str(out_dir / (f.stem + ".pkl")), expected_n_dof)
    else:
        raise FileNotFoundError(f"{src} does not exist")


if __name__ == "__main__":
    main()
