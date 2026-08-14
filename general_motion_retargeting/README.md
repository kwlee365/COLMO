# `general_motion_retargeting/` — Core Library

Core Python package for retargeting human motion (BVH / SMPL-X / FBX-offline) onto
the Unitree **G1** and **H1** humanoids.

## Pipeline at a glance

```
human motion file
  → format loader (utils/*)  →  per-frame dict {body_name: (position, quaternion)}
  → GeneralMotionRetargeting.retarget()  (mink IK QP)
  → robot qpos  →  viewer / saved .pkl
```

## Core modules

| File | Role |
|------|------|
| `motion_retarget.py` | **Heart of the pipeline.** `GeneralMotionRetargeting.retarget()` scales/offsets each human frame, sets mink `FrameTask` targets, and solves a 2-stage (table1/table2) IK QP to produce robot `qpos`. Includes CBF / ISSf self-collision avoidance (hard QP inequalities), a **soft** per-frame acceleration limit (DAQP `sense=8`), and a foot-contact zero-velocity limit. Reads runtime params from `assets/<robot>/collision_cfg.yaml`. |
| `params.py` | Registry constants: robot MuJoCo XML paths (`ROBOT_XML_DICT`), input-source × robot IK-config JSON map (`IK_CONFIG_DICT`), robot base body names (`ROBOT_BASE_DICT`), viewer camera distances (`VIEWER_CAM_DISTANCE_DICT`). Offline sources: `smplx`, `bvh_lafan1`, `bvh_nokov`, `bvh_xsens`, `fbx_offline`. Real-time teleop sources (g1 only): `fbx` (OptiTrack), `xrobot` (PICO), `xsens_mvn`. |
| `__init__.py` | Public API. Re-exports the constants plus `GeneralMotionRetargeting`, `RobotMotionViewer`, `draw_frame`, `load_robot_motion`, `KinematicsModel`, `human_head_to_robot_neck`, and (if `xrobotoolkit_sdk` is installed) `XRobotStreamer` / `XRobotRecorder`. |
| `robot_motion_viewer.py` | MuJoCo passive-viewer visualization. `RobotMotionViewer.step()` sets robot qpos and renders, overlaying human coordinate frames (`draw_frame`) and foot-contact points; supports camera follow, fps rate-limiting, mp4 recording (imageio), and collision-geom toggling. |
| `data_loader.py` | Reader for saved robot-motion pickles. `load_robot_motion()` unpacks fps / root_pos / root_rot / dof_pos and converts root_rot xyzw → wxyz (MuJoCo scalar-first). Consumed by the playback/vis scripts. |

## Kinematics & math

| File | Role |
|------|------|
| `kinematics_model.py` | Pure-PyTorch (batched, differentiable) forward-kinematics model parsed directly from the robot MuJoCo XML. `forward_kinematics(root_pos, root_rot, dof_pos) → body_pos/body_rot`. **Not** part of the IK; used by the batch dataset scripts to compute `local_body_pos` for the saved pkl. |
| `torch_utils.py` | IsaacGym-derived batched PyTorch quaternion ops (scalar-last / xyzw). Used by `kinematics_model.py`; mainly RL/post-processing helpers, not the core IK. |
| `rot_utils.py` | NumPy/SciPy rotation helpers (scalar-first). `quat_mul_np` is used by `xrobot_utils.py` for the Unity→right-handed coordinate transform. |

## Real-time teleoperation

Live mocap sources. Each streams frames in the same `{body_name: (position, quaternion)}`
format the offline loaders produce, so they feed `retarget()` unchanged.

| File | Role |
|------|------|
| `xrobot_utils.py` | **PICO / XRoboToolkit.** `XRobotStreamer` pulls body-, hand-, controller- and headset-tracking from the `xrobotoolkit_sdk` bindings and converts Unity coordinates to right-handed. `XRobotRecorder` replays a recorded mp4 + tracking-txt pair with the same interface. Import is guarded: without the SDK the classes still import but constructing `XRobotStreamer` raises a clear `ImportError`. Driven by `scripts/xrobot_to_robot.py`. |
| `neck_retarget.py` | `human_head_to_robot_neck()` — head-relative-to-spine rotation → (neck_yaw, neck_pitch) radians, for robots with an actuated neck. |
| `optitrack_vendor/` | Vendored OptiTrack **NatNet** client (`NatNetClient.py` + `DataDescriptions.py` / `MoCapData.py`). `setup_optitrack()` builds a client; `get_frame()` pops one skeleton frame off the receive queue. Driven by `scripts/optitrack_to_robot.py`. |
| `utils/xsens_vendor/xsens_to_colmo_adapter.py` | **Xsens MVN live.** `XsensToCOLMO` wraps the external `xsens_mvn_robot.XsensWrapper` UDP stream, maps Xsens link names to the IK-config body names, and applies yaw normalization. Driven by `scripts/xsens_live_streaming.py`. |

Teleop constructs the retargeter exactly like the offline path, minus
`adjust_hips_scale_for_motion()` (which needs the whole clip up front), so the plain
scalar root scaling applies. The first `retarget()` call still runs the full
`warmup_iters` settle + ground calibration; every call after that is one normal IK solve.
Collision avoidance defaults to the offline `collision_mode` from
`assets/<robot>/collision_cfg.yaml` — all three teleop scripts take `--collision_mode`
to trade it away for loop rate.

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
