import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "unitree_h1": ASSET_ROOT / "unitree_h1" / "h1.xml",
    "kapex": ASSET_ROOT / "kapex" / "KAPEX0_mockup_wo_hand.xml",
    # Low-poly (~15% faces) visual variant of kapex for fast visualization only;
    # identical kinematics/joints, so result pkls made for "kapex" work as-is.
    "kapex_lite": ASSET_ROOT / "kapex" / "KAPEX0_mockup_wo_hand_lowpoly.xml",
    "unitree_go2": ASSET_ROOT / "unitree_go2" / "go2.xml",
    # Booster T1 (serial-chain variant, 23 actuated DOF), imported from GMR.
    "booster_t1": ASSET_ROOT / "booster_t1" / "T1_serial.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "unitree_h1": IK_CONFIG_ROOT / "smplx_to_h1.json",
        "kapex": IK_CONFIG_ROOT / "smplx_to_kapex.json",
        "kapex_lite": IK_CONFIG_ROOT / "smplx_to_kapex.json",
        "unitree_go2": IK_CONFIG_ROOT / "smplx_to_go2.json",
        "booster_t1": IK_CONFIG_ROOT / "smplx_to_booster_t1.json",
    },
    "bvh_lafan1":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",
        "unitree_h1": IK_CONFIG_ROOT / "bvh_lafan1_to_h1.json",
        "kapex": IK_CONFIG_ROOT / "bvh_lafan1_to_kapex.json",
        "kapex_lite": IK_CONFIG_ROOT / "bvh_lafan1_to_kapex.json",
        "unitree_go2": IK_CONFIG_ROOT / "bvh_lafan1_to_go2.json",
        "booster_t1": IK_CONFIG_ROOT / "bvh_lafan1_to_booster_t1.json",
    },
    "bvh_nokov":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_nokov_to_g1.json",
    },
    "bvh_xsens":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_xsens_to_g1.json",
    },
    "fbx_offline":{
        "unitree_g1": IK_CONFIG_ROOT / "fbx_offline_to_g1.json",
    },

    # real-time teleoperation sources
    # OptiTrack / Motive live streaming (NatNet), see scripts/optitrack_to_robot.py
    "fbx":{
        "unitree_g1": IK_CONFIG_ROOT / "fbx_to_g1.json",
    },
    # PICO / XRoboToolkit body tracking, see scripts/xrobot_to_robot.py
    "xrobot":{
        "unitree_g1": IK_CONFIG_ROOT / "xrobot_to_g1.json",
    },
    # Xsens MVN live streaming, see scripts/xsens_live_streaming.py
    "xsens_mvn":{
        "unitree_g1": IK_CONFIG_ROOT / "xsens_mvn_to_g1.json",
    },
}


ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_h1": "pelvis",
    "kapex": "pelvis",
    "kapex_lite": "pelvis",
    "unitree_go2": "base_link",
    "booster_t1": "Waist",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "unitree_h1": 3.0,
    "kapex": 2.5,
    "kapex_lite": 2.5,
    "unitree_go2": 1.5,
    "booster_t1": 2.0,
}
