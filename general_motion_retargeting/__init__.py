"""Backwards-compatibility shim: `general_motion_retargeting` -> COLMO.

COLMO is a fork of GMR with the package renamed to `collision_free_motion_retargeting`
and the retargeter class renamed to `CollisionFreeMotionRetargeting`. Downstream projects
written against upstream GMR -- notably TWIST2's `deploy_real/xrobot_teleop_to_robot_w_hand.py`
-- still do `from general_motion_retargeting import GeneralMotionRetargeting as GMR`, which
breaks against a COLMO checkout.

This package makes those imports work unchanged. It is a thin alias layer, NOT a second
implementation: every symbol is COLMO's, and `general_motion_retargeting.<anything>`
resolves to `collision_free_motion_retargeting.<anything>` (including nested submodules
like `utils.lafan1`). The only real code here is the `GeneralMotionRetargeting` subclass,
which restores the two API details COLMO dropped:

  * `retarget(human_data, offset_to_ground=True)` -- COLMO's signature is
    `retarget(human_data, frame_idx=None)`.
  * `offset_human_data_to_ground()` -- removed from COLMO, restored verbatim below.

New code in this repo should import from `collision_free_motion_retargeting` directly.
"""

import importlib
import importlib.abc
import importlib.util
import sys

import numpy as np

import collision_free_motion_retargeting as _colmo
from collision_free_motion_retargeting import (  # noqa: F401  (re-exported)
    ASSET_ROOT,
    IK_CONFIG_DICT,
    IK_CONFIG_ROOT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    CollisionFreeMotionRetargeting,
    KinematicsModel,
    RobotMotionViewer,
    XRobotRecorder,
    XRobotStreamer,
    draw_frame,
    human_head_to_robot_neck,
    load_robot_motion,
)

_TARGET_PKG = _colmo.__name__


class _AliasLoader(importlib.abc.Loader):
    """Loader that hands back the already-imported COLMO module object."""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self._real_name)

    def exec_module(self, module):
        # The real module was executed by its own import; nothing to do here.
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Map `general_motion_retargeting.X.Y` onto `collision_free_motion_retargeting.X.Y`.

    Registered at the front of sys.meta_path so it wins over the path-based finder that
    would otherwise look for real files inside this (nearly empty) shim directory. Kept
    lazy on purpose: eagerly walking COLMO's subpackages would drag in optional-heavy
    modules such as utils.xsens (PyQt6) on every import.
    """

    _PREFIX = __name__ + "."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._PREFIX):
            return None
        real_name = _TARGET_PKG + "." + fullname[len(self._PREFIX):]
        try:
            importlib.import_module(real_name)
        except ImportError:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real_name))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


# GMR constructor kwargs that COLMO reads from assets/<robot>/collision_cfg.yaml instead.
# Accepted and ignored so old call sites keep working; a note is printed because the YAML
# value silently wins rather than the value the caller passed.
_YAML_OWNED_KWARGS = ("solver", "damping", "lm_damping", "max_iter", "motion_fps")


class GeneralMotionRetargeting(CollisionFreeMotionRetargeting):
    """COLMO's retargeter under GMR's name and GMR's call signature."""

    def __init__(self, *args, **kwargs):
        ignored = [k for k in _YAML_OWNED_KWARGS if k in kwargs]
        for k in ignored:
            kwargs.pop(k)
        if ignored:
            print(
                f"[GMR-compat] ignoring {ignored}: COLMO reads these from "
                f"assets/<robot>/collision_cfg.yaml, not from the constructor."
            )
        super().__init__(*args, **kwargs)
        self._offset_to_ground = False

    def retarget(self, human_data, offset_to_ground=False, frame_idx=None):
        """GMR signature. `offset_to_ground` shifts the human so the lowest foot sits
        `ground_offset` above z=0 before the IK targets are set (GMR behaviour)."""
        self._offset_to_ground = bool(offset_to_ground)
        return super().retarget(human_data, frame_idx=frame_idx)

    def apply_ground_offset(self, human_data):
        # update_targets() calls this at exactly the point GMR applied its to-ground shift,
        # so hooking here reproduces GMR's ordering without duplicating update_targets().
        human_data = super().apply_ground_offset(human_data)
        if self._offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        return human_data

    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground

        Restored verbatim from GMR (removed by COLMO's trim commit).
        """
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data


# GMR's own alias, kept so `from general_motion_retargeting import GMR` also works.
GMR = GeneralMotionRetargeting

__all__ = [
    "ASSET_ROOT",
    "IK_CONFIG_DICT",
    "IK_CONFIG_ROOT",
    "ROBOT_BASE_DICT",
    "ROBOT_XML_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
    "GMR",
    "GeneralMotionRetargeting",
    "CollisionFreeMotionRetargeting",
    "KinematicsModel",
    "RobotMotionViewer",
    "XRobotRecorder",
    "XRobotStreamer",
    "draw_frame",
    "human_head_to_robot_neck",
    "load_robot_motion",
]
