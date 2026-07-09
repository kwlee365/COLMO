import argparse
import pathlib
import os
import time
import pickle
import gc

import numpy as np
import torch
from tqdm import tqdm
from rich import print

from collision_free_motion_retargeting.utils.lafan1 import load_bvh_file as load_lafan1_file
from collision_free_motion_retargeting.kinematics_model import KinematicsModel
from collision_free_motion_retargeting import CollisionFreeMotionRetargeting as COLMO


def collect_bvh_files(src_folder: str):
    bvh_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for fn in filenames:
            if fn.lower().endswith(".bvh"):
                bvh_files.append(os.path.join(dirpath, fn))
    bvh_files.sort()
    return bvh_files


def process_one_bvh(
    bvh_file_path: str,
    tgt_file_path: str,
    robot: str,
    frame_stride: int,
    max_frames: int,
    device: str = "cuda:0",
    collision_mode: str = None,
):
    # Load BVH
    lafan1_data_frames, actual_human_height = load_lafan1_file(bvh_file_path)
    src_fps = 30

    if frame_stride and frame_stride > 1:
        lafan1_data_frames = lafan1_data_frames[::frame_stride]
        src_fps = int(round(src_fps / frame_stride))

    if max_frames and max_frames > 0:
        lafan1_data_frames = lafan1_data_frames[:max_frames]

    num_frames = len(lafan1_data_frames)
    if num_frames == 0:
        raise RuntimeError("No frames after slicing")

    # Init retarget (table mode: JSON human_scale_table, height-ratio scaled).
    retarget = COLMO(
        src_human="bvh_lafan1",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        collision_mode=collision_mode,
    )

    # Retarget per frame
    qpos_list = []
    t0 = time.time()

    frame_pbar = tqdm(
        enumerate(lafan1_data_frames),
        total=num_frames,
        desc=f"Frames ({os.path.basename(bvh_file_path)})",
        unit="frame",
        leave=False,
        mininterval=0.5,  # reduce overhead
    )

    for i, smplx_data in frame_pbar:
        qpos = retarget.retarget(smplx_data)
        qpos_list.append(qpos.copy())

        elapsed = time.time() - t0
        fps_eff = (i + 1) / max(elapsed, 1e-9)
        eta_sec = (num_frames - (i + 1)) / max(fps_eff, 1e-9)
        frame_pbar.set_postfix_str(f"{fps_eff:.2f} it/s, ETA {eta_sec/60:.1f}m")

    qpos_list = np.asarray(qpos_list)

    # FK (batched)
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos_list[:, 7:]

    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos,
        identity_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    body_names = kinematics_model.body_names

    HEIGHT_ADJUST = True
    PERFRAME_ADJUST = False
    if HEIGHT_ADJUST:
        body_pos, _ = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )  # T x N x 3
        ground_offset = 0.0
        if not PERFRAME_ADJUST:
            lowest_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
        else:
            for i in range(root_pos.shape[0]):
                lowest_body_part = torch.min(body_pos[i, :, 2]).item()
                root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

    motion_data = {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "fps": src_fps,
        "link_body_list": body_names,
    }

    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)

    total_elapsed = time.time() - t0

    # aggressive cleanup (helps long batch runs)
    del kinematics_model, local_body_pos, qpos_list
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return num_frames, src_fps, total_elapsed


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()

    parser.add_argument("--bvh_file", type=str, default=None, help="Single BVH file to process.")
    parser.add_argument("--src_folder", type=str, default=None, help="Folder containing BVH files (process all).")

    parser.add_argument("--save_path", type=str, default=None, help="Output pkl path for single-file mode.")
    parser.add_argument("--tgt_folder", type=str, default="results", help="Output root folder for folder mode.")

    parser.add_argument("--robot", default="unitree_g1", type=str)
    parser.add_argument("--override", action="store_true")

    parser.add_argument("--max_files", default=0, type=int)
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)

    # NEW: device selection
    parser.add_argument("--device", default="cuda:0", type=str, help="FK device: cuda:0 or cpu")

    parser.add_argument("--collision_mode", choices=["cbf", "issf", "off"],
                        default=None,
                        help="Collision avoidance mode. cbf: hard QP inequality "
                             "(mink.CollisionAvoidanceLimit); issf: robustified hard CBF "
                             "(ISSfCollisionAvoidanceLimit); off. Unset -> YAML "
                             "parameters.collision_mode or 'issf'.")

    args = parser.parse_args()

    if (args.bvh_file is None) == (args.src_folder is None):
        raise ValueError("Provide exactly one of --bvh_file or --src_folder")

    if args.bvh_file:
        if args.save_path is None:
            raise ValueError("--save_path is required when using --bvh_file")

        if os.path.exists(args.save_path) and not args.override:
            print(f"[yellow]Skip (exists): {args.save_path}[/yellow]")
        else:
            nframes, fps, tel = process_one_bvh(
                args.bvh_file,
                args.save_path,
                args.robot,
                args.frame_stride,
                args.max_frames,
                device=args.device,
                collision_mode=args.collision_mode,
            )
            print(f"[green]Saved[/green]: {args.save_path} | frames={nframes} | fps={fps} | time={tel:.1f}s")

    else:
        bvh_files = collect_bvh_files(args.src_folder)
        if args.max_files and args.max_files > 0:
            bvh_files = bvh_files[:args.max_files]

        if len(bvh_files) == 0:
            raise RuntimeError(f"No .bvh files found under: {args.src_folder}")

        print(f"[bold]Found {len(bvh_files)} BVH files[/bold]")

        for bvh_file_path in tqdm(bvh_files, desc="Files", unit="file"):
            rel_path = os.path.relpath(bvh_file_path, args.src_folder)
            tgt_file_path = os.path.join(args.tgt_folder, os.path.splitext(rel_path)[0] + ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                tqdm.write(f"Skipping (exists): {tgt_file_path}")
                continue

            try:
                nframes, fps, tel = process_one_bvh(
                    bvh_file_path,
                    tgt_file_path,
                    args.robot,
                    args.frame_stride,
                    args.max_frames,
                    device=args.device,
                    collision_mode=args.collision_mode,
                )
                tqdm.write(f"Saved: {tgt_file_path} | frames={nframes} | fps={fps} | time={tel:.1f}s")
            except Exception as e:
                tqdm.write(f"[ERROR] {bvh_file_path}: {e}")

        print("Done. saved to ", args.tgt_folder)