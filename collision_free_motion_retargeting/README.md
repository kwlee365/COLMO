# `collision_free_motion_retargeting/` — Core Library

Core Python package for retargeting human motion (BVH / SMPL-X / FBX-offline) onto
the Unitree **G1** and **H1** humanoids.

## Pipeline at a glance

```
human motion file
  → format loader (utils/*)  →  per-frame dict {body_name: (position, quaternion)}
  → CollisionFreeMotionRetargeting.retarget()  (mink IK QP)
  → robot qpos  →  viewer / saved .pkl
```

## Core modules

| File | Role |
|------|------|
| `motion_retarget.py` | **Heart of the pipeline.** `CollisionFreeMotionRetargeting.retarget()` scales/offsets each human frame, sets mink `FrameTask` targets, and solves a 2-stage (table1/table2) IK QP to produce robot `qpos`. Includes CBF / ISSf self-collision avoidance (hard QP inequalities), a **soft** per-frame acceleration limit (DAQP `sense=8`), and a foot-contact zero-velocity limit. Reads runtime params from `assets/<robot>/collision_cfg.yaml`. |
| `params.py` | Registry constants: robot MuJoCo XML paths (`ROBOT_XML_DICT`), input-source × robot IK-config JSON map (`IK_CONFIG_DICT`), robot base body names (`ROBOT_BASE_DICT`), viewer camera distances (`VIEWER_CAM_DISTANCE_DICT`). Currently registers **g1** and **h1** only; sources: `smplx`, `bvh_lafan1`, `bvh_nokov`, `bvh_xsens`, `fbx_offline`. |
| `__init__.py` | Public API. Re-exports the constants plus `CollisionFreeMotionRetargeting`, `RobotMotionViewer`, `draw_frame`, `load_robot_motion`, `KinematicsModel`. |
| `robot_motion_viewer.py` | MuJoCo passive-viewer visualization. `RobotMotionViewer.step()` sets robot qpos and renders, overlaying human coordinate frames (`draw_frame`) and foot-contact points; supports camera follow, fps rate-limiting, mp4 recording (imageio), and collision-geom toggling. |
| `data_loader.py` | Reader for saved robot-motion pickles. `load_robot_motion()` unpacks fps / root_pos / root_rot / dof_pos and converts root_rot xyzw → wxyz (MuJoCo scalar-first). Consumed by the playback/vis scripts. |

## Kinematics & math

| File | Role |
|------|------|
| `kinematics_model.py` | Pure-PyTorch (batched, differentiable) forward-kinematics model parsed directly from the robot MuJoCo XML. `forward_kinematics(root_pos, root_rot, dof_pos) → body_pos/body_rot`. **Not** part of the IK; used by the batch dataset scripts to compute `local_body_pos` for the saved pkl. |
| `torch_utils.py` | IsaacGym-derived batched PyTorch quaternion ops (scalar-last / xyzw). Used by `kinematics_model.py`; mainly RL/post-processing helpers, not the core IK. |
| `rot_utils.py` | NumPy/SciPy rotation helpers (scalar-first). **Currently unused** — its only caller was the (now removed) XR teleop module. |

## `utils/` — human-format loaders

| File | Role |
|------|------|
| `utils/lafan1.py` | **LAFAN1 / Nokov BVH loader.** Computes global pose, applies Y-up→Z-up + cm→m, adds `LeftFootMod`/`RightFootMod` foot keys; returns `(frames, human_height)`. |
| `utils/smpl.py` | **SMPL-X / GVHMR loader.** Runs the SMPL-X body model, builds per-joint global (pos, quat) dicts, and offers fps down-sampling (slerp). |
| `utils/xsens.py` | **Xsens BVH loader** (via `xsens_vendor/BVHParser` + `offsets.json`). Note: imports the PyQt6 `CurveEditor`, so it pulls in the Xsens GUI dependency chain. |
| `utils/lafan_vendor/extract.py` | Original-author LAFAN1 BVH parser. `read_bvh` (used by `lafan1.py`). |
| `utils/lafan_vendor/utils.py` | LAFAN1 vendor quaternion/FK math (`quat_fk`, etc.), shared by `lafan1.py` / `xsens.py` / `extract.py`. |
| `utils/__init__.py`, `utils/lafan_vendor/__init__.py` | Empty package initializers. |

## `utils/xsens_vendor/` — Xsens-only tooling

Needed **only if you use Xsens BVH input**; not part of LAFAN1/SMPL-X retargeting.

| File | Role |
|------|------|
| `BVHParser.py` | Xsens BVH parser (HIERARCHY/MOTION, zxy axis remap, MuJoCo XML generation, `bias_edit()` GUI hook). Used by `xsens.py`. |
| `mujoco_xsens_bvh_view.py` | Replays a parsed Xsens BVH as a **human skeleton** in MuJoCo (pre-retarget inspection / recording). |
| `mujoco_retargeting_robot_view.py` | Standalone: replays a retargeted pkl on the robot and exports CSV (hardcoded paths). |
| `mujoco_xml_read.py` | Standalone one-off snippet: prints human-vs-robot body coordinate ratios. |
| `pkls_to_csvs.py` | Standalone: batch-convert retargeted pkls → CSV (multiprocessing). |
| `video_recorder.py` | Generic image-stream → mp4 helper (cv2 + optional ffmpeg re-encode). |
| `rq.py` | Standalone dev snippet: rotvec → quaternion value printer. |
| `bvh_edit/CurveEditor.py` | PyQt6 GUI for editing joint rotation-offset curves + `offsets.json`. |
| `bvh_edit/bspline.py` | B-spline curve-editing prototype (**unused**). |
| `bvh_edit/spine_bias_edit.py` | Thin `BVHParser` subclass hook (**unused**). |
| `xsens_vendor/__init__.py`, `bvh_edit/__init__.py` | Empty package initializers. |
