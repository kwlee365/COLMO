"""Capture styling shared by the offscreen-rendering vis scripts.

Imports nothing but numpy / mujoco, so scripts that avoid pulling in the retargeting
package (torch, mink) at startup can use it too. Running ``python scripts/<x>.py`` puts
``scripts/`` on sys.path, so a plain ``from render_style import ...`` works.
"""

import argparse

import mujoco as mj
import numpy as np


def parse_frame_spec(text):
    """argparse type for one --snapshot_frames token: ``N``, ``A-B`` or ``A-B:STEP``.

    ``A-B`` is INCLUSIVE of B, so ``--snapshot_frames 0 60 100-200:20`` means frame 0,
    frame 60, and every 20th frame from 100 through 200 (same spec as vis_compare_robots).
    """
    t = text.strip()
    span, _, step_text = t.partition(":")
    start_text, sep, end_text = span.partition("-")
    try:
        start = int(start_text)
        end = int(end_text) if sep else start
        step = int(step_text) if step_text else 1
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad frame spec {text!r}; use N, A-B or A-B:STEP")
    if start < 0 or end < start or step <= 0:
        raise argparse.ArgumentTypeError(f"bad frame spec {text!r}; use N, A-B or A-B:STEP")
    return list(range(start, end + 1, step))


STUDIO_FLOORS = {
    # name: (floor rgba, floor emission)
    "light": ([0.92, 0.92, 0.91, 1.0], 0.3),   # near-white, the default figure look
    # Mid-gray: separates from the white background at the horizon, so with a camera at
    # floor height (--cam_eye_height) a floating sole shows white background under it.
    "gray": ([0.82, 0.82, 0.82, 1.0], 0.3),
}


def apply_studio(model, cam_distance, floor="light"):
    """Paper-figure look: white background, near-white (or gray) floor, softer light.

    - The dark-blue horizon band is the model's gradient skybox plus its blue haze; both
      go white, and white fog starting well behind the subject fades the infinite floor
      into the sky instead of ending it at a hard line. (Fog is a scene flag: the caller
      turns on mjRND_FOG.)
    - The floor gets emission so it reads near-white while still showing the shadow.
    - Weaker key light + more ambient lowers the contrast of the hard-edged patches the
      robot's own limbs shadow onto it (MuJoCo shadows are hard-edged)."""
    model.vis.rgba.haze[:3] = 1.0
    model.vis.rgba.fog[:] = [1.0, 1.0, 1.0, 1.0]
    extent = float(model.stat.extent)  # fog distances are in units of the model extent
    model.vis.map.fogstart = (cam_distance + 3.0) / extent
    model.vis.map.fogend = (cam_distance + 25.0) / extent
    for t in range(model.ntex):
        if model.tex_type[t] == mj.mjtTexture.mjTEXTURE_SKYBOX:
            adr = int(model.tex_adr[t])
            size = int(model.tex_height[t] * model.tex_width[t] * model.tex_nchannel[t])
            model.tex_data[adr:adr + size] = 255
    floor_rgba, floor_emission = STUDIO_FLOORS[floor]
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[gid] = floor_rgba
            mid = int(model.geom_matid[gid])
            if mid >= 0:
                model.mat_rgba[mid] = floor_rgba
                model.mat_emission[mid] = floor_emission
                model.mat_texid[mid, :] = -1  # drop any checker/grid texture
    if model.nlight > 0:
        model.light_diffuse[0] = [0.4, 0.4, 0.4]
    model.vis.headlight.ambient[:] = [0.25, 0.25, 0.25]


def apply_shadow(model, center_xy, radius, cam_azimuth, skew, drop, shadow_size, name="model"):
    """Make ground shadows visible: opaque floor, directional light from the camera side.

    ``center_xy``/``radius``: the floor area the shadow map must cover (where the subject
    starts and how far it travels). ``skew`` throws the shadow sideways, ``drop`` sets how
    steeply the light comes down (the light direction's -z against a unit horizontal).

    The G1 floor ships semi-transparent and MuJoCo draws transparent geoms in a pass with
    no shadow map, so the floor must be opaque for the light's castshadow to show (same
    as vis_compare_robots)."""
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[gid, 3] = 1.0
    model.vis.quality.shadowsize = max(int(model.vis.quality.shadowsize), int(shadow_size))
    if model.nlight == 0:
        print(f"[warn] {name} has no light; the shadow option only makes the floor opaque")
        return
    # Directional instead of the XML's 45-deg spotlight: the spot cone spreads the shadow
    # map over a wide area (jagged self-shadows on the torso), and narrowing the cone
    # darkens the floor outside it. A directional light lights the floor evenly and packs
    # the shadow map into an orthographic box of half-size shadowclip * extent, sized here
    # to cover a standing figure (~2.4 m) plus how far it travels.
    if hasattr(model, "light_type"):
        model.light_type[0] = mj.mjtLightType.mjLIGHT_DIRECTIONAL
    else:  # MuJoCo < 3.3
        model.light_directional[0] = 1
    model.light_castshadow[0] = 1
    model.vis.map.shadowclip = (2.4 + float(radius)) / float(model.stat.extent)
    # Put the light on the camera's side so the shadow falls away from the viewer
    # instead of toward it.
    az = np.deg2rad(cam_azimuth)
    view = np.array([np.cos(az), np.sin(az), 0.0])   # camera -> scene
    side = np.array([-np.sin(az), np.cos(az), 0.0])  # image-right in world
    light_dir = view + skew * side + np.array([0.0, 0.0, -drop])
    light_dir /= np.linalg.norm(light_dir)
    model.light_dir[0] = light_dir
    model.light_pos[0] = np.array([center_xy[0], center_xy[1], 0.0]) - light_dir * 7.0


def shadow_light_params(studio, skew=None):
    """(skew, drop) for apply_shadow. Studio lights from higher and nearer the camera
    axis, so the robot's limbs throw fewer shadows onto itself."""
    if skew is None:
        skew = 0.15 if studio else 0.45
    return skew, (1.8 if studio else 1.1)


def label_font_scale(render_height):
    """Label font for a render of this height, so text labels keep roughly the size they
    have at 720p (MuJoCo's default 150) at higher resolutions / supersampling."""
    target = 150 * render_height / 720
    scale = min((100, 150, 200, 250, 300), key=lambda s: abs(s - target))
    return getattr(mj.mjtFontScale, f"mjFONTSCALE_{scale}")


def downsample(img, width, height):
    """Lanczos-downscale a supersampled render to the output size (no-op if equal)."""
    if img.shape[1] == width and img.shape[0] == height:
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((width, height), Image.LANCZOS))


# ---------------------------------------------------------------------------
# Scene overlays
# ---------------------------------------------------------------------------
def add_sphere(scene, pos, radius, rgba, label=""):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    geom.label = label
    scene.ngeom += 1


def add_capsule(scene, from_pos, to_pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
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
    scene.ngeom += 1


# Foot-clearance overlay: on the ground (|gap| <= threshold), floating, or penetrating.
# Every marker also carries its value as a text label, so the state never rests on
# color alone.
FOOT_CONTACT_RGBA = (0.62, 0.62, 0.60, 0.95)
FOOT_FLOAT_RGBA = (0.93, 0.63, 0.00, 1.0)
FOOT_PEN_RGBA = (0.89, 0.29, 0.28, 1.0)


def draw_foot_gap(scene, lowest_point, floor_z, thresh):
    """Stem from the floor up to a foot's lowest point plus a floor dot labeled with the
    gap in cm. Returns the gap [m] (positive = floating, negative = penetrating)."""
    p = np.asarray(lowest_point, dtype=np.float64)
    gap = float(p[2]) - float(floor_z)
    if scene is None:
        return gap
    if gap > thresh:
        rgba = FOOT_FLOAT_RGBA
    elif gap < -thresh:
        rgba = FOOT_PEN_RGBA
    else:
        rgba = FOOT_CONTACT_RGBA
    floor_pt = np.array([p[0], p[1], floor_z])
    if abs(gap) > 1e-3:
        add_capsule(scene, floor_pt, p, 0.004, rgba)
    add_sphere(scene, floor_pt, 0.012, rgba, label=f"{100 * gap:+.1f} cm")
    return gap


def floor_height(model):
    """Height of the model's first plane geom (0 if it has none)."""
    planes = [g for g in range(model.ngeom) if model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE]
    return float(model.geom_pos[planes[0], 2]) if planes else 0.0


# Close-up on the feet (--cam_target feet): the camera looks at the point between the
# feet at this height, so floor, soles and a lifted foot all stay in frame.
FEET_LOOKAT_Z = 0.2


def elevation_for_eye_height(eye_height, lookat, distance, floor_z=0.0):
    """MuJoCo camera elevation [deg] that puts the eye `eye_height` above the floor while
    looking at `lookat` from `distance` (positive = looking up)."""
    rise = (float(floor_z) + float(eye_height) - float(lookat[2])) / float(distance)
    return -float(np.rad2deg(np.arcsin(np.clip(rise, -1.0, 1.0))))


def skybox_texids(model):
    return [t for t in range(model.ntex) if model.tex_type[t] == mj.mjtTexture.mjTEXTURE_SKYBOX]
