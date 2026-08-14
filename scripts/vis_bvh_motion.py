import argparse
import time
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from rich import print
from tqdm import tqdm

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


# Minimal MuJoCo world for skeleton visualization
EMPTY_WORLD_XML = """
<mujoco model="bvh_skeleton">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def draw_sphere(viewer, pos, radius=0.025, rgba=(1.0, 0.3, 0.3, 1.0)):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def draw_bone(viewer, from_pos, to_pos, radius=0.012, rgba=(0.9, 0.9, 0.2, 1.0)):
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=radius,
        from_=np.asarray(from_pos, dtype=np.float64),
        to=np.asarray(to_pos, dtype=np.float64),
    )
    viewer.user_scn.ngeom += 1


def draw_frame_axes(viewer, pos, quat_wxyz, size=0.08):
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba_list[i],
        )
        mj.mjv_connector(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos,
            to=pos + size * mat[:, i],
        )
        viewer.user_scn.ngeom += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BVH human motion in MuJoCo viewer.")
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file to visualize.")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true", default=False, help="Loop the motion.")
    parser.add_argument("--hide_axes", action="store_true", default=False,
                        help="Hide the per-joint coordinate frames (on by default).")
    parser.add_argument("--axes_size", type=float, default=0.2,
                        help="Length of the per-joint coordinate frame axes (meters).")
    parser.add_argument("--no_follow_camera", action="store_true", default=False,
                        help="Disable camera following the root (Hips) joint.")
    args = parser.parse_args()

    # Load BVH frames as dict-per-frame, and grab parent/bone hierarchy for skeleton edges.
    frames, human_height = load_bvh_file(args.bvh_file, format=args.format)
    raw = read_bvh(args.bvh_file)
    bones = list(raw.bones)
    parents = list(raw.parents)

    # Only draw bones for joints that exist in the per-frame dict (i.e. the BVH joints).
    edges = [(bones[i], bones[p]) for i, p in enumerate(parents) if p >= 0 and bones[i] in frames[0]]

    print(f"Loaded {len(frames)} frames from [bold]{args.bvh_file}[/bold] "
          f"({len(bones)} bones, human_height={human_height:.2f} m)")

    # Build a tiny MuJoCo world (just the floor) and launch the passive viewer.
    model = mj.MjModel.from_xml_string(EMPTY_WORLD_XML)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 120

    rate_limiter = RateLimiter(frequency=args.motion_fps, warn=False)
    pbar = tqdm(total=len(frames), desc="Playing BVH")

    i = 0
    try:
        while viewer.is_running():
            frame = frames[i]

            viewer.user_scn.ngeom = 0

            # Joints as spheres
            for bone_name, (pos, quat) in frame.items():
                # Hide the "Mod" helper joints used internally for retargeting
                if bone_name.endswith("Mod"):
                    continue
                rgba = (1.0, 0.4, 0.2, 1.0) if bone_name == "Hips" else (0.2, 0.6, 1.0, 1.0)
                radius = 0.04 if bone_name == "Hips" else 0.025
                draw_sphere(viewer, pos, radius=radius, rgba=rgba)
                if not args.hide_axes:
                    draw_frame_axes(viewer, pos, quat, size=args.axes_size)

            # Skeleton edges (parent -> child)
            for child_name, parent_name in edges:
                child_pos = frame[child_name][0]
                parent_pos = frame[parent_name][0]
                draw_bone(viewer, parent_pos, child_pos)

            if not args.no_follow_camera and "Hips" in frame:
                viewer.cam.lookat = frame["Hips"][0]

            viewer.sync()
            rate_limiter.sleep()
            pbar.update(1)

            if args.loop:
                i = (i + 1) % len(frames)
                if i == 0:
                    pbar.reset()
            else:
                i += 1
                if i >= len(frames):
                    break
    finally:
        pbar.close()
        viewer.close()
        time.sleep(0.3)
