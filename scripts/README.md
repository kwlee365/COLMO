# `scripts/` — Entry Points

Command-line entry points for motion retargeting, visualization, and data conversion.
All retargeting targets the Unitree **G1** / **H1** humanoids and reads runtime params
from `assets/<robot>/collision_cfg.yaml`.

Common flags: `--robot {unitree_g1, unitree_h1}`, `--collision_mode {cbf, issf, off}`,
`--save_path <out.pkl>`, `--record_video --video_path <out.mp4>`.

## Retargeting (input format → robot)

| Script | Role |
|--------|------|
| `bvh_to_robot.py` | Retarget **one** LAFAN1/Nokov BVH file and view it live (+ optional `.pkl` save). |
| `bvh_to_robot_dataset.py` | **Batch** retarget a BVH folder; also runs `KinematicsModel` FK to store `local_body_pos` in each pkl. Supports `--frame_stride`, `--max_frames`, resume/skip. |
| `smplx_to_robot.py` | Retarget one SMPL-X (`.npz`, AMASS) motion and view it (uses the JSON `human_scale_table`). |
| `smplx_to_robot_dataset.py` | Multiprocess **batch** retarget of an AMASS folder (motion filtering, ground alignment, memory watchdog). |
| `smpl_to_robot.py` | Retarget an already-FK'd SMPL `.npz` (joints + local rotmats); reuses the SMPL-X IK config. |
| `gvhmr_to_robot.py` | Convert GVHMR (video-based HMR) `.pt` predictions to SMPL-X data, then retarget + view. |
| `fbx_offline_to_robot.py` | Retarget an offline OptiTrack FBX (pickled frames) + ground alignment. |
| `xsens_bvh_to_robot.py` | Retarget an Xsens (3DSM) BVH motion (needs the `xsens_vendor` subsystem). |

## Teleoperation (live mocap → robot)

Real-time streaming. All three open a MuJoCo window and retarget every incoming frame;
`unitree_g1` only, since that is the only robot with live-source IK configs. Collision
avoidance defaults to `collision_cfg.yaml`; pass `--collision_mode off` if the loop
cannot keep up with the mocap rate.

| Script | Role |
|--------|------|
| `optitrack_to_robot.py` | **OptiTrack / Motive** over NatNet. Needs `--server_ip` (the Motive machine) and `--client_ip` (this machine); firewalls off on both. Source `fbx`. |
| `xrobot_to_robot.py` | **PICO / XRoboToolkit** body tracking (TWIST2-style). Needs `xrobotoolkit-pc-service` running and the `xrobotoolkit_sdk` bindings installed. Source `xrobot`. |
| `xsens_live_streaming.py` | **Xsens MVN** UDP network stream (`--port`, default 9763). Needs the external `xsens_mvn_robot` package. Source `xsens_mvn`. |

## Visualization

| Script | Role |
|--------|------|
| `vis_robot_motion.py` | Play back a single robot-motion `.pkl` (minimal viewer, optional recording). |
| `vis_robot_motion_dataset.py` | 3-mode viewer for a folder of pkls: `interactive` (browse with `[`/`]`), `batch` (record each to mp4 via one subprocess per motion), `single`. |
| `vis_bvh_motion.py` | Render just the **input BVH skeleton** in an empty MuJoCo world (input sanity-check; no retargeting). |
| `vis_colmo_with_bvh.py` | Show the retargeted robot next to / overlaid on the source BVH skeleton; draws IK targets, keypoints, and offsets for debugging the human→robot matching. |
| `vis_robot_urdf.py` | **IsaacGym** utility: load a robot asset and print rigid-body / DOF names (asset inspection; unrelated to the retargeting pipeline, requires `isaacgym`). |

## Data conversion (pre-/post-processing)

| Script | Role |
|--------|------|
| `convert_omomo_to_smplx.py` | Split the raw OMOMO dataset into per-sequence SMPL-X `.pkl` files (hardcoded paths). |
| `smpl_to_smplx.py` | Normalize SMPL `.npz` → SMPL-X format (pad betas, rename keys, split poses). Single-file or folder batch. |
| `npz_to_pkl.py` | Convert a holosoma robot-motion `.npz` (qpos) → COLMO `.pkl` format so it can be replayed. |
| `batch_colmo_pkl_to_csv.py` | Batch-convert retargeted `.pkl` → beyondmimic CSV (`root_pos + root_rot + dof_pos`, down-sampled to 30 fps). |

## Typical workflow

```bash
# 1. retarget a single BVH and preview
python scripts/bvh_to_robot.py --bvh_file <file.bvh> --robot unitree_g1 --collision_mode issf

# 2. batch a whole BVH folder into .pkl
python scripts/bvh_to_robot_dataset.py --src_folder <bvh_dir> --tgt_folder results/out --robot unitree_g1

# 3. review the results (batch-record videos)
python scripts/vis_robot_motion_dataset.py --mode batch --robot unitree_g1 \
    --robot_motion_folder results/out --video_dir videos/out
```
