# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

This is a Python package for motion retargeting to humanoid robots. Install in development mode:

```bash
conda create -n colmo python=3.10 -y
conda activate colmo
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

## Code Architecture

### Core Components

- **`CollisionFreeMotionRetargeting`** (`collision_free_motion_retargeting/motion_retarget.py`): Main class for motion retargeting using inverse kinematics (IK) solver built on mink/mujoco
- **`KinematicsModel`** (`collision_free_motion_retargeting/kinematics_model.py`): Handles robot kinematics calculations
- **`RobotMotionViewer`** (`collision_free_motion_retargeting/robot_motion_viewer.py`): MuJoCo-based visualization for robot motions
- **Configuration System** (`collision_free_motion_retargeting/params.py`): Simplified robot definitions and IK config mappings - cleaned to focus on core supported robots

### Data Flow

1. **Human Motion Input**: SMPL-X (AMASS/OMOMO) or BVH (LAFAN1) format
2. **Motion Format**: Each frame = dict of (human_body_name, 3D translation + rotation)
3. **Robot Output**: Tuple of (base_translation, base_rotation, joint_positions)
4. **IK Configs**: JSON files in `collision_free_motion_retargeting/ik_configs/` define human-to-robot body mappings

### Supported Robots

Core robot models in `assets/` directory:
- Unitree G1 (`unitree_g1`) - 29 DOF humanoid
- Unitree H1 (`unitree_h1`) - humanoid

Other robots were removed; only G1 and H1 are registered in `params.py`
(`ROBOT_XML_DICT` / `IK_CONFIG_DICT` / `ROBOT_BASE_DICT`) and in the script
`--robot` choices.

## Common Commands

### Single Motion Retargeting
```bash
# SMPL-X to robot
python scripts/smplx_to_robot.py --smplx_file <path> --robot <robot_name> --save_path <output.pkl>

# BVH to robot  
python scripts/bvh_to_robot.py --bvh_file <path> --robot <robot_name> --save_path <output.pkl>
```

### Batch Processing
```bash
# Process datasets
python scripts/smplx_to_robot_dataset.py
python scripts/bvh_to_robot_dataset.py
```

### Visualization
```bash
# Visualize saved robot motion
python scripts/vis_robot_motion.py --robot <robot_name> --robot_motion_path <path.pkl>
```

Add `--record_video --video_path <output.mp4>` to any visualization command to record video.

### Teleoperation (live mocap → robot, `unitree_g1` only)
```bash
python scripts/optitrack_to_robot.py --server_ip <ip> --client_ip <ip> --use_multicast False
python scripts/xrobot_to_robot.py --robot unitree_g1     # PICO / XRoboToolkit
python scripts/xsens_live_streaming.py --port 9763       # Xsens MVN
```

All three accept `--collision_mode {issf, cbf, off}` to trade collision avoidance for
loop rate.

## Key Technical Details

- **IK Solver**: Uses mink library with configurable solver (default: "daqp") and damping (default: 5e-1)
- **Human Height Scaling**: Automatic scaling based on `actual_human_height` parameter vs config assumptions
- **Body Model Dependencies**: Requires SMPL-X body models in `assets/body_models/smplx/`

## File Organization

- `scripts/`: Entry point scripts for different retargeting workflows
- `collision_free_motion_retargeting/`: Core library code
- `assets/`: Robot models (MuJoCo XML) and body models (SMPL-X)
- `collision_free_motion_retargeting/ik_configs/`: JSON configuration files for human-to-robot body mappings:
  - SMPL-X configs: `smplx_to_{g1,h1}.json`
  - BVH configs: `bvh_lafan1_to_g1.json`, `bvh_{nokov,xsens}_to_g1.json`
  - FBX configs: `fbx_offline_to_g1.json` (offline), `fbx_to_g1.json` (OptiTrack live)
  - Teleop configs: `xrobot_to_g1.json` (PICO), `xsens_mvn_to_g1.json` (Xsens MVN live)
- `collision_free_motion_retargeting/optitrack_vendor/`: vendored NatNet client for OptiTrack streaming
- `collision_free_motion_retargeting/xrobot_utils.py`: PICO / XRoboToolkit streamer + recorder
- `general_motion_retargeting/`: **compatibility shim only** — aliases GMR's old package/class
  names onto COLMO so upstream GMR code (TWIST2) imports unchanged. Never put real logic here;
  new code imports `collision_free_motion_retargeting` directly.

## Project Status & Features

**Current State**: Motion retargeting system focused on Unitree G1 and H1.

**Key Capabilities**:
- **Multi-format Input**: SMPL-X (AMASS/OMOMO), BVH (LAFAN1/Nokov/Xsens), FBX (offline)
- **Live Teleop Input**: OptiTrack (NatNet), PICO / XRoboToolkit, Xsens MVN
- **Robot Models**: Unitree G1 (29 DOF) and Unitree H1
- **Robust IK**: Mink-based solver with automatic human height scaling
- **Visualization**: MuJoCo-based viewer with video recording capabilities
- **Batch Processing**: Dataset-level retargeting workflows

**Use Cases**:
- Offline motion retargeting (BVH / SMPL-X / FBX files → robot)
- Real-time whole-body teleoperation (OptiTrack / PICO / Xsens → robot)
- RL policy training data generation
- Motion capture to robot deployment
- Cross-platform humanoid motion transfer