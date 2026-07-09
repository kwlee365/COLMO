# COLMO: Collision-Free Motion Retargeting

  <!-- <a href="https://arxiv.org/abs/2505.02833">
    <img src="https://img.shields.io/badge/paper-arXiv%3A2505.02833-b31b1b.svg" alt="arXiv Paper"/> -->
  </a> <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  <!-- </a> <a href="https://github.com/YanjieZe/GMR/releases">
    <img src="https://img.shields.io/badge/version-0.2.0-blue.svg" alt="Version"/>
  </a> <a href="https://yanjieze.github.io/humanoid-foundation/#GMR">
    <img src="https://img.shields.io/badge/blog-GMR-blue.svg" alt="Blog"/> -->
  </a>

#### Key features of COLMO:
- Generates high-quality kinematic reference motions for humanoid whole-body control and RL-based motion tracking.
- Collision-free humanoid motion retargeting using ISSf-CBF-based self-collision avoidance and ground-penetration prevention.
- Soft acceleration-limit constraints that reduce velocity spikes, abrupt corrective motions, and infeasibility caused by hard acceleration bounds.
- Built on the overall code structure of GMR, including its two-stage differential IK retargeting pipeline, and extended with the proposed collision-free retargeting formulation.

This repo is licensed under the [MIT License](LICENSE).

## Demos
<!-- 
<table>
  <tr>
    <td align="center" width="20%">
      <b>Demo 1</b><br>
      Retargeting LAFAN1 dancing motion to 5 robots.<br>
      <video src="https://github.com/user-attachments/assets/23566fa5-6335-46b9-957b-4b26aed11b9e" width="200" controls></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 2</b><br>
      Galexea R1 Pro robot (view 1).<br>
      <video src="https://github.com/user-attachments/assets/903ed0b0-0ac5-4226-8f82-5a88631e9b7c" width="200" controls></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 3</b><br>
      Galexea R1 Pro robot (view 2).<br>
      <video src="https://github.com/user-attachments/assets/deea0e64-f1c6-41bc-8661-351682006d5d" width="200" controls></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 4</b><br>
      Switching robots by changing one argument.<br>
      <video src="https://github.com/user-attachments/assets/03f10902-c541-40b1-8104-715a5759fd5e" width="200" controls></video>
    </td>
    <td align="center" width="20%">
      <b>Demo 5</b><br>
      HighTorque robot doing a twist dance.<br>
      <video src="https://github.com/user-attachments/assets/1d3e663b-f29e-41b1-8e15-5c0deb6a4a5c" width="200" controls></video>
    </td>
  </tr>

  <tr>
    <td align="center">
      <b>Demo 6</b><br>
      Kuavo robot picking up a box.<br>
      <video src="https://github.com/user-attachments/assets/02fc8f41-c363-484b-a329-4f4e83ed5b80" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 7</b><br>
      Unitree H1 robot doing a ChaCha dance.<br>
      <video src="https://github.com/user-attachments/assets/28ee6f0f-be30-42bb-8543-cf1152d97724" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 8</b><br>
      Booster T1 robot jumping (view 1).<br>
      <video src="https://github.com/user-attachments/assets/2c75a146-e28f-4327-930f-5281bfc2ca9c" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 9</b><br>
      Booster T1 robot jumping (view 2).<br>
      <video src="https://github.com/user-attachments/assets/ff10c7ef-4357-4789-9219-23c6db8dba6d" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 10</b><br>
      Unitree H1-2 robot jumping.<br>
      <video src="https://github.com/user-attachments/assets/2382d8ce-7902-432f-ab45-348a11eeb312" width="200" controls></video>
    </td>
  </tr>

  <tr>
    <td align="center">
      <b>Demo 11</b><br>
      PND Adam Lite robot.<br>
      <video src="https://github.com/user-attachments/assets/a8ef1409-88f1-4393-9cd0-d2b14216d2a4" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 12</b><br>
      Tienkung robot walking.<br>
      <video src="https://github.com/user-attachments/assets/7a775ecc-4254-450c-a3eb-49e843b8e331" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 13</b><br>
      Extracting human pose (GVHMR + COLMO).<br>
      <a href="https://www.bilibili.com/video/BV1Tnpmz9EaE">▶ Watch on Bilibili</a>
    </td>
    <td align="center">
      <b>Demo 14</b><br>
      PAL Robotics’ Talos robot fighting.<br>
      <video src="https://github.com/user-attachments/assets/3ec0bf80-80c1-4181-a623-dc2b072c2ca2" width="200" controls></video>
    </td>
    <td align="center">
      <b>Demo 15</b><br>
      (Optional placeholder if you add a new one later!)<br>
      <i>Coming soon...</i>
    </td>
  </tr>
</table> -->

## Installation

> [!NOTE]
> The code is tested on Ubuntu 22.04/20.04.

First create your conda environment:

```bash
conda create -n colmo python=3.10 -y
conda activate colmo
```

Then, install COLMO:

```bash
pip install -e .
```

After installing SMPLX, change `ext` in `smplx/body_models.py` from `npz` to `pkl` if you are using SMPL-X pkl files.

And to resolve some possible rendering issues:

```bash
conda install -c conda-forge libstdcxx-ng -y
```

## Data Preparation

[[SMPLX](https://github.com/vchoutas/smplx) body model] download SMPL-X body models to `assets/body_models` from [SMPL-X](https://smpl-x.is.tue.mpg.de/) and then structure as follows:
```bash
- assets/body_models/smplx/
-- SMPLX_NEUTRAL.pkl
-- SMPLX_FEMALE.pkl
-- SMPLX_MALE.pkl
```

[[AMASS](https://amass.is.tue.mpg.de/) motion data] download raw SMPL-X data to any folder you want from [AMASS](https://amass.is.tue.mpg.de/). NOTE: Do not download SMPL+H data.

[[OMOMO](https://github.com/lijiaman/omomo_release) motion data] download raw OMOMO data to any folder you want from [this google drive file](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view?usp=sharing). And process the data into the SMPL-X format using `scripts/convert_omomo_to_smplx.py`.

[[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) motion data] download raw LAFAN1 bvh files from [the official repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), i.e., [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip).


## Human/Robot Motion Data Formulation

To better use this library, you can first have an understanding of the human motion data we use and the robot motion data we obtain.

Each frame of **human motion data** is formulated as a dict of (human_body_name, 3d global translation + global rotation). The rotation is usually represented as quaternion (with wxyz order by default, to align with mujoco).

Each frame of **robot motion data** can be understood as a tuple of (robot_base_translation, robot_base_rotation, robot_joint_positions).

## Usage

### Retargeting from SMPL-X (AMASS, OMOMO) to Robot

> [!NOTE]
> NOTE: after install SMPL-X, change `ext` in `smplx/body_models.py` from `npz` to `pkl` if you are using SMPL-X pkl files.

Retarget a single motion:

```bash
python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit
```

By default you should see the visualization of the retargeted robot motion in a mujoco window.
If you want to record video, add `--record_video` and `--video_path <your_video_path,mp4>`.

- `--rate_limit` is used to limit the rate of the retargeted robot motion to keep the same as the human motion. If you want it as fast as possible, remove `--rate_limit`.

Retarget a folder of motions:

```bash
python scripts/smplx_to_robot_dataset.py --src_folder <path_to_dir_of_smplx_data> --tgt_folder <path_to_dir_to_save_robot_data> --robot <robot_name>
```

By default there is no visualization for batch retargeting.

### Retargeting from GVHMR to Robot

First, install GVHMR by following [their official instructions](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md).

And run their demo that can extract human pose from monocular video:

```bash
cd path/to/GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```

Then you should obtain the saved human pose data in `GVHMR/outputs/demo/tennis/hmr4d_results.pt`.

Then, run the command below to retarget the extracted human pose data to your robot:

```bash
python scripts/gvhmr_to_robot.py --gvhmr_pred_file <path_to_hmr4d_results.pt> --robot unitree_g1 --record_video
```



## Retargeting from BVH (LAFAN1, Nokov) to Robot

Retarget a single motion:

```bash
# single motion
python scripts/bvh_to_robot.py --bvh_file <path_to_bvh_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit --format <format>
```

By default you should see the visualization of the retargeted robot motion in a mujoco window. 
- `--rate_limit` is used to limit the rate of the retargeted robot motion to keep the same as the human motion. If you want it as fast as possible, remove `--rate_limit`.
- `--format` is used to specify the format of the BVH data. Supported formats are `lafan1` and `nokov`.


Retarget a folder of motions:

```bash
python scripts/bvh_to_robot_dataset.py --src_folder <path_to_dir_of_bvh_data> --tgt_folder <path_to_dir_to_save_robot_data> --robot <robot_name>
```

By default there is no visualization for batch retargeting.



## Retargeting from Xsens to Robot

### Offline: Xsens BVH to Robot

#### Visualize Xsens BVH Data using MuJoCo

Install PyQt6:
```bash
pip install PyQt6 PyQt6-Qt6 PyQt6-sip
```


```bash
python collision_free_motion_retargeting/utils/xsens_vendor/mujoco_xsens_bvh_view.py \
  --bvh_file <path_to_dir_of_bvh_data> \
  --scale <displacement scaling size> \
  --reset_to_zero
```
like
```bash
python collision_free_motion_retargeting/utils/xsens_vendor/mujoco_xsens_bvh_view.py \
  --scale 0.01 \
  --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
  --reset_to_zero
```

- `--start` is used to specify the initial processing frame. If no input is given, processing will start from the first frame by default.

- `--end` is used to specify the final processing frame. If not input, it will be processed by default to the last frame.

- `--reset_to_zero` is used to reset the displacement and Z-axis rotation to zero.This function, when used in combination with `--start`, will set the data to the initial zero position very well.Because sometimes the first one or two frames of some datasets differ too much from the subsequent data, these data need to be discarded.

- `--scale` is used to set the scaling value of the displacement, which depends on the relationship between the unit used for the displacement in the dataset and the meter.

- ##### Before using it, you must install PyQt6. `pip install PyQt6`
- ##### Upon executing this command, a UI interface will be launched, enabling you to adjust the angle values for each channel in the x, y, and z directions of every joint. After completing the adjustments, click the `"Apply and Preview"` button, which will generate an `offset.json` file locally and perform a MuJoCo visualization playback of the BVH file. When running `xsens_bvh_to_robot.py`, it will read the data from this JSON file. Therefore, you need to execute `mujoco_xsens_bvh_view.py` prior to using `xsens_bvh_to_robot.py` for motion retargeting, to ensure that the `offset.json` file exists locally.

#### Retarget a single motion:
```bash
# single motion
python scripts/xsens_bvh_to_robot.py \
  --bvh_file <path_to_bvh_data> \
  --robot <path_to_robot_data> \
  --save_path <path_to_save_robot_data.pkl> \
  --rate_limit \
  --start <number of the first frame> \
  --scale <displacement scaling size> \
  --reset_to_zero \
  --bvh_format <exported bvh format>
```
like
```bash
python scripts/xsens_bvh_to_robot.py  \
  --robot unitree_h1_2 \
  --scale 0.01 \
  --reset_to_zero \
  --bvh_format 3DSM \
  --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
  --save_path retargeting_data/h1/251021_04_boxing_120Hz_cm_3DsMax.pkl
```
##### By default you should see the visualization of the retargeted robot motion in a mujoco window. 
- `--rate_limit` is used to limit the rate of the retargeted robot motion to keep the same as the human motion. If you want it as fast as possible, remove `--rate_limit`.

- `--start` is used to specify the initial processing frame. If no input is given, processing will start from the first frame by default.

- `--end` is used to specify the final processing frame. If not input, it will be processed by default to the last frame.

- `--reset_to_zero` is used to reset the displacement and Z-axis rotation to zero.This function, when used in combination with `--start`, will set the data to the initial zero position very well.Because sometimes the first one or two frames of some datasets differ too much from the subsequent data, these data need to be discarded.

- `--scale` is used to set the scaling value of the displacement, which depends on the relationship between the unit used for the displacement in the dataset and the meter.

##### ！！！！！！！！！！！！！！！！！！ ATTENTION ！！！！！！！！！！！！！！！！！！！！
- `--bvh_format` is used to set the format of the bvh being parsed. In the Xsens MVN software, BVH files in three formats can be exported. There will be some differences among BVH files in different formats. Here I recommend using the 3D Studio Max format.(In fact, I have not yet completed the parsing of data in other formats.)

- The exported pkl file will represent quaternions in the `wxyz` format. ^ _ ^

---

### Online Streaming (Xsens MVN)

Stream live motion data from **Xsens MVN Software** directly into COLMO for real-time robot retargeting.

#### 1. Install the Xsens MVN UDP Data Parser

The `xsens_mvn_robot_python` library parses the Xsens MVN network datagram (Position + Orientation in quaternion format) into Python-accessible data structures. Install the correct `.whl` file for your Python version.

```bash
# Clone the parser repository
git clone https://github.com/jiminghe/xsens_mvn_robot_python.git
cd xsens_mvn_robot_python

# Install the matching wheel for your Python version
# Example for Python 3.10:
pip install xsens_mvn_robot_python-*-cp310-*.whl
```

> Select the `.whl` file whose filename contains your Python version tag (e.g. `cp310` for Python 3.10, `cp38` for Python 3.8). This library handles UDP socket binding and datagram unpacking automatically.

#### 2. Configure the Xsens MVN Network Streamer

Launch **Xsens MVN Software** on either Windows or Linux. You can stream from a live recording session while wearing the Xsens Link / Awinda suit, or replay a previously recorded `.mvn` file.

| Step | Action |
|---|---|
| 1 | Click **Options → Network Streamer** |
| 2 | In the pop-up window, click **Add** to create a new stream destination |
| 3 | Set the **Host Address** (see table below) |
| 4 | Under Network Streamer Options, tick **Position + Orientation (Quaternion)** only |
| 5 | No other data sources are needed for COLMO retargeting |
| 6 | Click **OK** — confirm green status on the streamer |

**Host Address Reference:**

| Scenario | Host Address Setting |
|---|---|
| MVN on the same Linux machine (MVN Linux) | `127.0.0.1` (localhost) |
| MVN on Windows → streaming to Ubuntu (same LAN) | Ubuntu IP address, e.g. `192.168.1.10` |

> **Important:** When streaming from a Windows PC to an Ubuntu computer, ensure both machines are on the same LAN. Disable Windows Firewall for the MVN application or create an inbound UDP rule on the MVN default port (`9763`).

#### 3. Run the COLMO Live Streaming Script

With the Xsens MVN Network Streamer active and the conda environment loaded, run the live-streaming retargeting script. A MuJoCo window will open showing the retargeted Unitree G1 robot mirroring your movements in real time.

```bash
# Activate the COLMO environment
conda activate colmo

# Run the Xsens live streaming retargeting script
python scripts/xsens_live_streaming.py
```

### Retargeting from FBX (OptiTrack) to Robot

#### Offline FBX Files

Retarget a single motion:

1. Install `fbx_sdk` by following [these instructions](https://github.com/nv-tlabs/ASE/tree/main/ase/poselib#importing-from-fbx) and [these instructions](https://github.com/nv-tlabs/ASE/issues/61#issuecomment-2670315114). You will probably need a new conda environment for this.

2. Activate the conda environment where you installed `fbx_sdk`.
Use the following command to extract motion data from your `.fbx` file:

```bash
cd third_party
python poselib/fbx_importer.py --input <path_to_fbx_file.fbx> --output <path_to_save_motion_data.pkl> --root-joint <root_joint_name> --fps <fps>
```

3. Then, run the command below to retarget the extracted motion data to your robot:

```bash
conda activate colmo
# single motion
python scripts/fbx_offline_to_robot.py --motion_file <path_to_saved_motion_data.pkl> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit
```

By default you should see the visualization of the retargeted robot motion in a mujoco window. 

- `--rate_limit` is used to limit the rate of the retargeted robot motion to keep the same as the human motion. If you want it as fast as possible, remove `--rate_limit`.

#### Online Streaming

We provide the script to use OptiTrack MoCap data for real-time streaming and retargeting.

Usually you will have two computers, one is the server that installed with Motive (Desktop APP for OptiTrack) and the other is the client that installed with COLMO.

Find the server ip (the computer that installed with Motive) and client ip (your computer). Set the streaming as follows:

![OptiTrack Streaming](./assets/optitrack.png)

And then run:

```bash
python scripts/optitrack_to_robot.py --server_ip <server_ip> --client_ip <client_ip> --use_multicast False --robot unitree_g1
```

You should see the visualization of the retargeted robot motion in a mujoco window.

### Visualize saved robot motion

Visualize a single motions:

```bash
python scripts/vis_robot_motion.py --robot <robot_name> --robot_motion_path <path_to_save_robot_data.pkl>
```

If you want to record video, add `--record_video` and `--video_path <your_video_path,mp4>`.

Visualize a folder of motions:

```bash
python scripts/vis_robot_motion_dataset.py --robot <robot_name> --robot_motion_folder <path_to_save_robot_data_folder>
```

After launching the MuJoCo visualization window and clicking on it, you can use the following keyboard controls::
* `[`: play the previous motion
* `]`: play the next motion
* `space`: toggle play/pause

## Citation


## Known Issues

Designing a single config for all different humans is not trivial. We observe some motions might have bad retargeting results. If you observe some bad results, please let us know! We now have a collection of such motions in [TEST_MOTIONS.md](TEST_MOTIONS.md).

## Acknowledgement
This repository is built upon the overall code structure of [GMR](https://github.com/YanjieZe/GeneralMotionRetargeting). We thank the GMR authors for open-sourcing their implementation.
Accordingly, our IK solver is built upon [mink](https://github.com/kevinzakka/mink) and [mujoco](https://github.com/google-deepmind/mujoco). Our visualization is built upon [mujoco](https://github.com/google-deepmind/mujoco). The human motion data we try includes [AMASS](https://amass.is.tue.mpg.de/), [OMOMO](https://github.com/lijiaman/omomo_release), and [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

The original robot models can be found at the following locations:

* [Unitree G1](https://github.com/unitreerobotics/unitree_ros): [Link to file](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
* [Unitree H1](https://github.com/unitreerobotics/unitree_ros): [Link to file](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/h1_description)