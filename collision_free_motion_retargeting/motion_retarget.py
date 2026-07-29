import mink
import mujoco as mj
import numpy as np
import json
import yaml
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT, ASSET_ROOT
from rich import print
from mink.exceptions import TargetNotSet
from mink.constants import dof_width
from mink.solve_ik import _compute_qp_objective, _compute_qp_inequalities
import daqp
from ctypes import c_int

# Supported collision-avoidance modes for CollisionFreeMotionRetargeting:
#   "cbf"  : hard QP inequality CBF (mink.CollisionAvoidanceLimit) added to the IK limits.
#   "issf" : hard QP inequality with the ISSf-CBF robustness margin
#            (ISSfCollisionAvoidanceLimit); a robustified variant of "cbf".
#   "off"  : no active collision avoidance (diagnostics are still logged).
COLLISION_MODES = ("cbf", "issf", "off")


def _require_param(params, key):
    """Fetch a REQUIRED parameter from collision_cfg.yaml `parameters:` (no built-in
    default). Raises KeyError if it is missing so misconfiguration fails loudly."""
    if key not in params:
        raise KeyError(
            f"collision_cfg.yaml 'parameters:' must define '{key}' (no built-in default)."
        )
    return params[key]


# How max_base_horizontal_speed is enforced (parameters.base_speed_cap_mode):
#   "per_frame" : increment-level saturation -- the nominal scale is kept EXACTLY on every
#                 frame under the cap, and only the frames that would exceed it are
#                 compressed (they land exactly on the cap). Default.
#   "clip"      : clip-level scaling -- one scalar shrink applied to the whole motion,
#                 sized off the p99.5 speed peak. Uniform, but one fast burst shrinks the
#                 entire clip. Kept for reproducing older runs / visual coherence.
BASE_SPEED_CAP_MODES = ("per_frame", "clip")


def saturate_root_xy(root_xy, nominal_scale_xy, limit, fps):
    """Increment-level (per-frame) saturation of a root horizontal trajectory.

        d_t   = nominal_scale_xy * (p_t - p_{t-1})        # nominally scaled increment
        k_t   = min(1, limit / (fps * ||d_t||))           # scalar => direction preserved
        p'_t  = p'_{t-1} + k_t * d_t,   p'_0 = nominal_scale_xy * p_0

    The resulting base speed is exactly ``min(nominal speed, limit)``: frames below the
    cap are reproduced at full nominal fidelity and only the offending frames are
    compressed. For an isotropic nominal scale this is the s_t = min(s_nom, limit/v_t)
    form; the elementwise product keeps it well-defined if x and y ever differ.

    MUST be integrated on INCREMENTS, not applied to absolute positions: a time-varying
    factor multiplied into an absolute root position teleports the root whenever the
    factor changes.

    Args:
        root_xy: (T, 2) raw human root xy trajectory [m].
        nominal_scale_xy: scalar or (2,) nominal horizontal scale (already height-ratio'd).
        limit: base horizontal speed cap [m/s]; None/<=0 disables saturation.
        fps: frame rate the trajectory is sampled at.

    Returns:
        (traj, info) with traj (T, 2) the saturated trajectory and info a dict of
        diagnostics (raw/nominal/capped peak speeds, saturated-frame fraction, path
        lengths). For T < 2 the trajectory is just the nominally scaled input.
    """
    p = np.asarray(root_xy, dtype=float)
    s_nom = np.broadcast_to(np.asarray(nominal_scale_xy, dtype=float), (2,))
    if p.shape[0] < 2:
        return p * s_nom, {}

    d = np.diff(p, axis=0) * s_nom                    # nominally scaled increments
    v_nom = np.linalg.norm(d, axis=1) * float(fps)    # speed the nominal scale would give
    if limit is not None and float(limit) > 0:
        k = np.minimum(1.0, float(limit) / np.maximum(v_nom, 1e-12))
    else:
        k = np.ones_like(v_nom)

    traj = np.empty_like(p)
    traj[0] = p[0] * s_nom
    traj[1:] = traj[0] + np.cumsum(d * k[:, None], axis=0)

    v_raw = np.linalg.norm(np.diff(p, axis=0), axis=1) * float(fps)
    info = dict(
        v_raw_max=float(v_raw.max()),
        v_raw_p995=float(np.percentile(v_raw, 99.5)),
        v_nom_max=float(v_nom.max()),
        base_peak=float((v_nom * k).max()),
        saturated_frac=float((k < 1.0 - 1e-12).mean()),
        path_len=float(np.linalg.norm(d * k[:, None], axis=1).sum()),
        path_len_nominal=float(np.linalg.norm(d, axis=1).sum()),
    )
    return traj, info


class ISSfCollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
    """
    ISSf-CBF variant of mink's collision avoidance limit.

    mink's stock limit enforces the discrete CBF condition (per active geom pair):

        J_AB q_dot  >=  -alpha * h_AB,      alpha = cbf_gain / dt,   h_AB = d - d_min

    This subclass adds the Input-to-State-Safe (ISSf) robustness margin (eq. 45):

        J_AB q_dot  >=  -alpha * h_AB  +  (1 / eps) * ||J_AB||_2^2

    The extra +(1/eps)||J_AB||^2 term raises the required separation rate near the
    boundary, giving a configuration-adaptive robustness margin: where the distance
    is sensitive to joint motion (large ||J_AB||) the margin grows. Smaller eps
    => larger margin (more conservative); eps -> inf recovers the plain CBF.

    Why a subclass and not `bound_relaxation`: that term is a single per-limit scalar
    fixed at construction, but ||J_AB(q)|| is per-pair AND configuration-dependent,
    so it must be recomputed for every row at every IK iteration.

    In mink's stored form `G v <= h` (where, in BOTH the non-penetrating and
    penetrating branches, `sign*row . v <= h` is equivalent to `h_dot >= -h_upper`),
    the +(1/eps)||J||^2 term lands as a NEGATIVE offset on the upper bound.

    NOTE on sign/norm: this uses the SQUARED L2 norm ||J_AB||^2 (the standard ISSf-CBF
    robustness term, 1/eps * ||L_g h||^2), with the margin tightening the constraint.
    For the first-power variant, drop the `** 2` on `np.linalg.norm(row)` below.
    """

    def __init__(self, *args, issf_epsilon: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.issf_epsilon = float(issf_epsilon)

    def compute_qp_inequalities(self, configuration, dt):
        model = self.model
        data = configuration.data
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, model.nv))
        distmax = self.collision_detection_distance
        min_dist = self.minimum_distance_from_collisions
        cbf_gain = self.gain  # mink's CollisionAvoidanceLimit stores the gain kwarg here
        eps = self.issf_epsilon
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            dist = mj.mj_geomDistance(model, data, geom1_id, geom2_id, distmax, self._fromto)
            if abs(dist - distmax) < 1e-12:
                continue
            row = mink.limits.collision_avoidance_limit.compute_contact_normal_jacobian(
                model, data, geom1_id, geom2_id,
                self._fromto, self._normal, self._jac1, self._jac2,
            )
            # ISSf robustness margin (1/eps)*||J_AB||_2^2, per-pair & per-q.
            # Appears as a negative offset on the upper bound in both branches.
            issf_margin = np.linalg.norm(row) ** 2 / eps
            if dist > min_dist:
                upper_bound[idx] = cbf_gain * (dist - min_dist) - issf_margin + self.bound_relaxation
            else:
                upper_bound[idx] = -issf_margin + self.bound_relaxation
            sign = -1.0 if dist >= 0 else 1.0
            coefficient_matrix[idx] = sign * row
        return mink.Constraint(G=coefficient_matrix, h=upper_bound)


class FrameAccelerationLimit(mink.Limit):
    """Real per-FRAME joint acceleration limit.

    Caps the change in per-frame velocity so that, per actuated joint,

        |v_t - v_{t-1}|  <=  a_max * frame_period,     frame_period = 1 / fps,

    where v_t = (qpos_t - qpos_{t-1}) / frame_period is the effective retargeted joint
    velocity at frame t. Equivalently, on the NET per-frame joint displacement
    Δqpos_t = qpos_t - qpos_{t-1},

        |Δqpos_t - Δqpos_{t-1}|  <=  a_max * frame_period^2.

    Like FrameVelocityLimit, this is a real limit in the MOTION's own time base (T = 1/fps),
    NOT opt.timestep. The IK QP decision variable is the tangent displacement Δq; with
    d = q ⊖ q_frame_start (net displacement so far this frame, latched via set_frame) and
    dq_prev = the previous frame's net displacement Δqpos_{t-1}, this iteration's Δq obeys

        dq_prev - a_max*T^2 <= d + Δq <= dq_prev + a_max*T^2.

    As d approaches the bound the admissible Δq shrinks to zero, capping the frame-to-frame
    velocity change without breaking the inner-loop convergence, and it shares the QP with
    the collision CBF. Free/floating-base DoFs are ignored (actuated joints only).
    """

    def __init__(self, model, accelerations, frame_period):
        """
        Args:
            model: MuJoCo model.
            accelerations: dict joint_name -> max |qddot| [rad/s^2 hinge, m/s^2 slide].
            frame_period: motion frame period [s] = 1 / fps.
        """
        self.model = model
        self.frame_period = float(frame_period)
        limit_list, index_list = [], []
        for joint_name, max_acc in accelerations.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            if jnt_type == mj.mjtJoint.mjJNT_FREE:
                continue                                   # base is not accel-limited here
            vadr = model.jnt_dofadr[jid]
            vdim = dof_width(int(jnt_type))
            index_list.extend(range(vadr, vadr + vdim))
            limit_list.extend([float(max_acc)] * vdim)

        self.indices = np.array(index_list, dtype=int)
        self.limit = np.array(limit_list, dtype=float)
        nb = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if nb > 0 else None

        # Per-frame state, refreshed once per frame via set_frame().
        self.q_frame_start = None              # qpos at the start of the current frame
        self.dq_prev = np.zeros(model.nv)      # previous frame's NET tangent displacement

    def set_frame(self, q_frame_start, dq_prev):
        """Latch the frame-start configuration and the previous frame's net displacement."""
        self.q_frame_start = np.array(q_frame_start, dtype=float)
        self.dq_prev = np.array(dq_prev, dtype=float)

    def compute_qp_inequalities(self, configuration, dt):
        if self.projection_matrix is None or self.q_frame_start is None:
            return mink.Constraint()
        # Net tangent displacement so far this frame: d = q ⊖ q_frame_start.
        d = np.zeros(self.model.nv)
        mj.mj_differentiatePos(self.model, d, 1.0, self.q_frame_start, configuration.q)

        aT2 = self.limit * self.frame_period * self.frame_period   # a_max * T^2
        dq_prev_k = self.dq_prev[self.indices]
        d_lim = d[self.indices]

        # +Δq rows:  Δq <=  dq_prev + a*T^2 - d
        # -Δq rows: -Δq <= -dq_prev + a*T^2 + d
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([dq_prev_k + aT2 - d_lim, -dq_prev_k + aT2 + d_lim])
        return mink.Constraint(G=G, h=h)


class FrameVelocityLimit(mink.Limit):
    """Real per-FRAME joint velocity limit.

    Caps the NET joint displacement across the whole IK solve of one motion frame so that

        |qpos_t - qpos_{t-1}|  <=  v_max * frame_period,      frame_period = 1 / fps,

    i.e. the effective retargeted joint velocity never exceeds ``v_max`` [rad/s]. Unlike a
    per-iteration velocity box (mink.VelocityLimit / VelocityLimitAllDof), which caps each
    QP step by ``v_max * opt.timestep`` -- a solver-internal step clamp, NOT a real velocity
    -- this bounds the net frame displacement in the MOTION's own time base.

    Mirrors AccelerationLimit's frame-based formulation. The QP decision variable is the
    tangent displacement Δq; with d = q ⊖ q_frame_start (net displacement since the frame
    start, latched via set_frame() once per frame), this iteration's Δq is bounded by

        -v_max*T <= d + Δq <= v_max*T,        T = frame_period.

    As d approaches the bound the admissible Δq shrinks to zero, so the limit caps the frame
    velocity without breaking the inner-loop convergence. Free/floating-base DoFs are ignored
    (actuated joints only).
    """

    def __init__(self, model, velocities, frame_period):
        """
        Args:
            model: MuJoCo model.
            velocities: dict joint_name -> v_max [rad/s hinge, m/s slide].
            frame_period: motion frame period [s] = 1 / fps.
        """
        self.model = model
        self.frame_period = float(frame_period)
        limit_list, index_list = [], []
        for joint_name, vmax in velocities.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            if jnt_type == mj.mjtJoint.mjJNT_FREE:
                continue                                   # base is not velocity-limited here
            vadr = model.jnt_dofadr[jid]
            vdim = dof_width(int(jnt_type))
            index_list.extend(range(vadr, vadr + vdim))
            limit_list.extend([float(vmax)] * vdim)
        self.indices = np.array(index_list, dtype=int)
        self.limit = np.array(limit_list, dtype=float)
        nb = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if nb > 0 else None
        self.q_frame_start = None                          # latched once per frame

    def set_frame(self, q_frame_start):
        """Latch the previous frame's final configuration (the anchor the net per-frame
        displacement is measured from)."""
        self.q_frame_start = np.array(q_frame_start, dtype=float)

    def compute_qp_inequalities(self, configuration, dt):
        if self.projection_matrix is None or self.q_frame_start is None:
            return mink.Constraint()
        # Net tangent displacement so far this frame: d = q ⊖ q_frame_start.
        d = np.zeros(self.model.nv)
        mj.mj_differentiatePos(self.model, d, 1.0, self.q_frame_start, configuration.q)
        cap = self.limit * self.frame_period               # v_max * T  (a displacement bound)
        d_lim = d[self.indices]
        # QP variable is Δq; bound the net displacement d + Δq to +/- cap:
        #   +Δq rows:  Δq <=  cap - d
        #   -Δq rows: -Δq <=  cap + d
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([cap - d_lim, cap + d_lim])
        return mink.Constraint(G=G, h=h)


class VelocityLimitAllDof(mink.Limit):
    """Per-iteration IK step clamp: |Δq| <= v_max * dt (dt = opt.timestep) over ALL DoFs,
    including the free/floating base (which mink.VelocityLimit skips). This is NOT a real
    velocity limit -- it clamps each solve_ik iteration's increment to regularize the
    differential-IK step (the old mink.VelocityLimit role). ``v_max_per_dof`` is a length-nv
    array; np.inf entries are left unconstrained. Applied SOFT (DAQP sense=8) in
    _solve_ik_soft, so it yields minimally when it would fight a hard collision wall.
    """

    def __init__(self, model, v_max_per_dof):
        self.model = model
        v = np.asarray(v_max_per_dof, dtype=float)
        self.indices = np.where(np.isfinite(v))[0]
        self.vmax = v[self.indices]
        self.projection_matrix = np.eye(model.nv)[self.indices] if len(self.indices) else None

    def compute_qp_inequalities(self, configuration, dt):
        if self.projection_matrix is None:
            return mink.Constraint()
        h = self.vmax * dt
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        return mink.Constraint(G=G, h=np.hstack([h, h]))


class CollisionFreeMotionRetargeting:
    """Collision-Free Motion Retargeting (COLMO).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        verbose: bool=True,
        use_velocity_limit: bool=None,       # enable FrameVelocityLimit; None -> read from YAML
        use_acceleration_limit: bool=None,   # enable FrameAccelerationLimit; None -> read from YAML
        use_ik_step_limit: bool=None,        # enable IK step clamp; None -> read from YAML
        collision_mode: str = None,
    ) -> None:
        self._warmup_done = False
        self.src_human = src_human
        self.tgt_robot = tgt_robot
        self.actual_human_height = actual_human_height

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # Print DoF names in order
        print("[COLMO] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[COLMO] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        
        print("[COLMO] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot], encoding="utf-8") as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])

        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
        self.height_ratio = float(ratio)   # config scale x this = effective scale used downstream

        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = (np.asarray(ik_config["human_scale_table"][key], dtype=float) * ratio)
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        # Robot foot IK-target bodies that should rest on the ground (e.g.
        # ["left_ankle_roll_link", "right_ankle_roll_link"]). Used only to know which link
        # meshes to measure for the warmup ground calibration (see retarget()); empty ->
        # calibration is skipped.
        self.ground_anchor_bodies = ik_config.get("ground_anchor_bodies", [])
        # Lazily-built cache of (geom_id, local_vertices) for those bodies' meshes, and the
        # floor plane height, used by measure_foot_float.
        self._foot_mesh_cache = None
        self._floor_z = None

        self.verbose = verbose

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        
        # task -> human body mapping for handling duplicate body_name assignments
        self.task_to_human_body1 = {}
        self.task_to_human_body2 = {}

        # HARD inequality limits (collision CBF, foot contact, and the joint-configuration
        # position limit) are appended below. The position-limit object is created here and
        # added to self.ik_limits once the YAML params are loaded. Velocity / acceleration /
        # IK-step limits are all SOFT (self._soft_limits), never hard.
        self.ik_limits = []
        self.config_limit = mink.ConfigurationLimit(self.model)

        # Load robot-specific collision parameters from an external YAML file
        collision_cfg_path = ASSET_ROOT / tgt_robot / "collision_cfg.yaml"
        with open(collision_cfg_path, 'r', encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Override IK solver parameters with values from the configuration
        params = cfg.get('parameters', {})
        # REQUIRED params: must be defined in collision_cfg.yaml or construction fails.
        self.damping = _require_param(params, 'damping')
        self.max_iter = _require_param(params, 'max_iter')
        # Per-joint velocity limits [rad/s] for the FrameVelocityLimit (real per-frame limit
        # |qpos_t - qpos_{t-1}| <= v_max/fps). Enabled by the use_velocity_limit constructor
        # arg; joints NOT listed here are left unconstrained.
        self.frame_velocity_limit_cfg = params.get('frame_velocity_limit_soft', {}) or {}
        # IK solver / loop tuning, externalized to YAML (back-compat defaults kept).
        self.solver = params.get('solver', 'daqp')
        self._warmup_iters = params.get('warmup_iters', 500)
        self.ik_tol = params.get('ik_convergence_tol', 1e-8)   # IK loop early-stop tol
        self.lm_damping = params.get('lm_damping', 1.0)        # per-FrameTask LM damping
        self.motion_fps = params.get('motion_fps', 30)         # foot-contact fps assumption
        # Optional cap [m/s] on the robot base horizontal speed, enforced by
        # adjust_hips_scale_for_motion() before the retarget loop. See BASE_SPEED_CAP_MODES
        # for how base_speed_cap_mode chooses between increment-level saturation (default)
        # and the older clip-level scaling.
        self.max_base_horizontal_speed = params.get('max_base_horizontal_speed', None)
        self.base_speed_cap_mode = str(params.get('base_speed_cap_mode', 'per_frame')).lower()
        if self.base_speed_cap_mode not in BASE_SPEED_CAP_MODES:
            raise ValueError(
                f"collision_cfg.yaml parameters.base_speed_cap_mode must be one of "
                f"{BASE_SPEED_CAP_MODES}, got '{self.base_speed_cap_mode}'."
            )
        # Precomputed saturated root xy trajectory for "per_frame" mode, indexed by the
        # frame counter maintained in retarget(). None -> scale_human_data falls back to
        # the plain scalar root scaling (also the path taken by "clip" mode and by any
        # caller that never runs adjust_hips_scale_for_motion).
        self._root_xy_traj = None
        self._frame_idx = 0        # index of the frame currently being retargeted
        self._auto_frame_idx = 0   # running counter used when retarget() gets no frame_idx
        self.ground_offset = params.get('ground_offset', -0.01)  # per-frame robot ground offset [m]
        # Warmup iteration at which the first-frame ground calibration measures the robot's
        # foot float and folds it into ground_offset (see retarget()). The warmup foot pose
        # converges within a few dozen iters, so this only needs to leave enough remaining
        # warmup iters to re-settle onto the ground; it is clamped to warmup_iters//2.
        self.ground_calib_warmup_iter = params.get('ground_calib_warmup_iter', 200)
        self._ground_calibrated = False
        # Per-joint acceleration limits [rad/s^2] for the FrameAccelerationLimit (real per-
        # frame limit |Δqpos_t - Δqpos_{t-1}| <= a_max*(1/fps)^2). Enabled by the
        # use_acceleration_limit constructor arg; joints NOT listed are left unconstrained.
        self.frame_acceleration_limit_cfg = params.get('frame_acceleration_limit_soft', {}) or {}

        # Collision-avoidance mode switch. Priority: explicit constructor arg >
        # YAML parameters.collision_mode > default "issf". See COLLISION_MODES.
        #   cbf  -> hard QP inequality (mink.CollisionAvoidanceLimit)
        #   issf -> hard QP inequality with the ISSf-CBF robustness margin
        #   off  -> none (diagnostics still logged)
        if collision_mode is None:
            collision_mode = params.get('collision_mode', 'issf')
        collision_mode = str(collision_mode).lower()
        if collision_mode not in COLLISION_MODES:
            raise ValueError(
                f"collision_mode must be one of {COLLISION_MODES}, got {collision_mode!r}"
            )
        self.collision_mode = collision_mode
        self.use_collision_constraint = collision_mode in ("cbf", "issf")
        self.use_issf = collision_mode == "issf"

        # Enable flags for the three IK limits. Priority (same as collision_mode): explicit
        # constructor arg > YAML parameters.<flag> > built-in default.
        def _resolve_flag(arg, key, default):
            return bool(params.get(key, default) if arg is None else arg)
        self.use_velocity_limit = _resolve_flag(use_velocity_limit, 'use_velocity_limit', True)
        self.use_acceleration_limit = _resolve_flag(use_acceleration_limit, 'use_acceleration_limit', True)
        self.use_ik_step_limit = _resolve_flag(use_ik_step_limit, 'use_ik_step_limit', False)
        # ISSf robustness scale (eq. 45): margin = ||J_AB|| / issf_epsilon.
        # Larger epsilon -> milder margin (epsilon -> inf recovers the plain CBF).
        self.issf_epsilon = params.get('issf_epsilon', 50.0)

        # Global defaults for the per-pair collision-limit parameters. Each entry in
        # collision_limits: may still override any of these; otherwise these apply.
        #   cbf_gain        : CBF rate in (0, 1] (hard-constraint path only).
        #   margin      : minimum safety distance d_min in h(q) = d - d_min [m].
        #   detect_dist : distance band within which the limit/barrier activates [m].
        # margin / detect_dist default to None (no global default) so a missing value
        # is reported instead of silently defaulting.
        self.collision_gain = params.get('cbf_gain', 0.001)
        self.collision_margin = params.get('margin', None)
        self.collision_detect_dist = params.get('detect_dist', None)

        # --- Soft-limit configuration -------------------------------------------------
        # The VELOCITY, ACCELERATION and IK-step limits are ALL applied as SOFT DAQP
        # constraints (sense=8, shared rho_soft) -- never hard walls -- so the QP never goes
        # infeasible and they yield MINIMALLY only where they would fight a hard collision /
        # foot-contact constraint. The POSITION (joint-config) limit is always HARD.
        # Magnitude [rad/s] for the per-iteration IK step clamp (use_ik_step_limit): the
        # per-solve-step increment is bounded by |Δq_iter| <= ik_step_limit * opt.timestep.
        self.ik_step_limit = float(params.get('ik_step_limit', 20.0))

        # Resolve actuated joints (name, dof_adr, dof_width) and the free-base DoFs once.
        actuated = []
        for a_id in range(self.model.nu):
            j_id = int(self.model.actuator_trnid[a_id, 0])
            if j_id < 0:
                continue
            j_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)
            if j_name is None:
                continue
            actuated.append((j_name, int(self.model.jnt_dofadr[j_id]),
                             dof_width(int(self.model.jnt_type[j_id]))))
        base_dofs = []
        for j_id in range(self.model.njnt):
            if self.model.jnt_type[j_id] == mj.mjtJoint.mjJNT_FREE:
                adr = int(self.model.jnt_dofadr[j_id])
                base_dofs = list(range(adr, adr + 6))
                break

        actuated_names = {n for n, _, _ in actuated}
        frame_period = 1.0 / float(self.motion_fps)

        # --- Velocity limit (real per-frame, SOFT) ------------------------------------
        # FrameVelocityLimit caps |qpos_t - qpos_{t-1}| <= v_max/fps per joint. Enabled by
        # the use_velocity_limit arg; per-joint v_max come from the frame_velocity_limit_soft
        # config map (joints NOT listed are left unconstrained). Applied SOFT (DAQP sense=8).
        self.frame_velocity_limit = None
        self._vel_limits_map = {}
        if self.use_velocity_limit:
            for n, v in self.frame_velocity_limit_cfg.items():
                if n in actuated_names:
                    self._vel_limits_map[n] = float(v)
                else:
                    print(f"[COLMO][Velocity] WARNING: frame_velocity_limit '{n}' is not "
                          f"an actuated joint, skipping.")
            if self._vel_limits_map:
                self.frame_velocity_limit = FrameVelocityLimit(
                    self.model, self._vel_limits_map, frame_period=frame_period)
            elif verbose:
                print("[COLMO][Velocity] use_velocity_limit=True but frame_velocity_limit is "
                      "empty -> no velocity limit applied.")

        # --- IK step limit (per-iteration Δq clamp; the old mink.VelocityLimit role) ------
        # NOT a real velocity limit: clamps each IK solve iteration's increment,
        # |Δq_iter| <= ik_step_limit * opt.timestep, to regularize the differential-IK step
        # (prevent large single-iteration jumps). Enabled by use_ik_step_limit; magnitude
        # ik_step_limit [rad/s]. Applied SOFT (VelocityLimitAllDof, over actuated + base).
        self.velocity_limit_obj = None
        if self.use_ik_step_limit:
            v_max = np.full(self.model.nv, np.inf)
            for _, dadr, w in actuated:
                v_max[dadr:dadr + w] = self.ik_step_limit
            for d in base_dofs:
                v_max[d] = self.ik_step_limit
            self.velocity_limit_obj = VelocityLimitAllDof(self.model, v_max)  # SOFT

        # --- Acceleration limit (real per-frame, soft only) ---------------------------
        # FrameAccelerationLimit caps |Δqpos_t - Δqpos_{t-1}| <= a_max*(1/fps)^2 per joint,
        # applied SOFT (DAQP sense=8) so the QP never becomes infeasible. Enabled by the
        # use_acceleration_limit arg; per-joint a_max come from the frame_acceleration_limit_soft
        # config map (joints NOT listed are unconstrained). Softness via soft_rho (DAQP
        # rho_soft, shared by all soft rows; smaller -> nearer-hard).
        self.accel_limit = None
        self._accel_limits_map = {}
        self.soft_rho = float(params.get('soft_rho', 1e-6))
        if self.use_acceleration_limit:
            for n, a in self.frame_acceleration_limit_cfg.items():
                if n in actuated_names:
                    self._accel_limits_map[n] = float(a)
                else:
                    print(f"[COLMO][Accel] WARNING: frame_acceleration_limit '{n}' is not "
                          f"an actuated joint, skipping.")
            if self._accel_limits_map:
                self.accel_limit = FrameAccelerationLimit(
                    self.model, self._accel_limits_map, frame_period=frame_period)
            elif verbose:
                print("[COLMO][Accel] use_acceleration_limit=True but frame_acceleration_limit "
                      "is empty -> no acceleration limit applied.")
        # Previous-frame NET tangent displacement consumed by the frame accel limit.
        self._accel_dq_prev = np.zeros(self.model.nv)

        # --- Assemble hard vs soft sets -----------------------------------------------
        # Position limit is ALWAYS hard (like collision / foot-contact). Velocity,
        # acceleration and the IK-step clamp are ALL soft (self._soft_limits, DAQP sense=8).
        self.ik_limits.append(self.config_limit)                     # position: HARD
        self._soft_limits = []
        if self.frame_velocity_limit is not None:                    # real per-frame velocity (SOFT)
            self._soft_limits.append(self.frame_velocity_limit)
        if self.velocity_limit_obj is not None:                      # IK step clamp (SOFT)
            self._soft_limits.append(self.velocity_limit_obj)
        if self.accel_limit is not None:                             # real per-frame accel (SOFT)
            self._soft_limits.append(self.accel_limit)

        if verbose:
            print(f"[COLMO] Final Parameters ->  Damping: {self.damping}, Max Iterations: {self.max_iter}")
            if self.frame_velocity_limit is not None:               # FrameVelocityLimit active
                lo, hi = float(self.frame_velocity_limit.limit.min()), float(self.frame_velocity_limit.limit.max())
                vel_desc = (f"FrameVelocityLimit (real, |Δqpos|<=v/fps, SOFT), v range {lo:g}-{hi:g} "
                            f"rad/s @ fps={self.motion_fps} | {len(self._vel_limits_map)} joints")
            else:                                                   # use_velocity_limit off / no joints
                vel_desc = 'off'
            print(f"[COLMO] Position limit: hard | Velocity limit: {vel_desc} | "
                  f"soft rho_soft={self.soft_rho}")
            step_desc = (f"{self.ik_step_limit} rad/s (SOFT)" if self.use_ik_step_limit else 'off')
            print(f"[COLMO] IK step limit (per-iteration Δq clamp): {step_desc}")
            if self.accel_limit is None:
                accel_status = 'off'
            else:
                lo, hi = float(self.accel_limit.limit.min()), float(self.accel_limit.limit.max())
                accel_status = (f"FrameAccelerationLimit (real, |ΔΔqpos|<=a*(1/fps)^2), a range "
                                f"{lo:g}-{hi:g} rad/s^2 @ fps={self.motion_fps} "
                                f"| {len(self._accel_limits_map)} joints")
            print(f"[COLMO] Acceleration limit (soft): {accel_status}")
            # NOTE: issf_epsilon (and cbf_gain / margin / detect_dist) are resolved PER
            # collision-limit entry below; parameters.issf_epsilon is only the fallback
            # default. The actual per-limit values are logged after the limits are built.
            print(f"[COLMO] Collision mode: {self.collision_mode} "
                  f"(constraint={self.use_collision_constraint})")
        
        # Resolve collision groups and individual geometries
        self.groups = cfg['groups']
        self.all_collision_limits = []
        _limit_summaries = []       # (name, cbf_gain, margin, detect_dist, issf_eps) for logging


        # Build collision pair metadata. Each limit provides geom pairs / margin /
        # detect_dist (h(q) = d - d_min) and, in "cbf"/"issf" mode, is registered as a
        # hard QP inequality in self.ik_limits.
        for limit_cfg in cfg['collision_limits']:
            geom_pairs = []

            for p_a, p_b in limit_cfg['pairs']:
                list_a = self.groups.get(p_a, [p_a] if isinstance(p_a, str) else p_a)
                list_b = self.groups.get(p_b, [p_b] if isinstance(p_b, str) else p_b)
                geom_pairs.append((list_a, list_b))

            # Resolve per-pair parameters, each falling back to the global default
            # from parameters: (see self.collision_*). `cbf_gain` (in (0, 1]) only affects
            # the CBF/issf hard-constraint path. margin (d_min) and detect_dist define
            # h(q) = d - d_min and its activation band. In "issf" mode the ISSf-CBF
            # variant adds the per-pair (1/eps)||J_AB|| robustness margin (eq. 45).
            cbf_gain = float(limit_cfg.get('cbf_gain', self.collision_gain))
            margin = limit_cfg.get('margin', self.collision_margin)
            detect_dist = limit_cfg.get('detect_dist', self.collision_detect_dist)
            issf_eps = float(limit_cfg.get('issf_epsilon', self.issf_epsilon))
            if margin is None or detect_dist is None:
                raise ValueError(
                    f"collision limit {limit_cfg.get('pairs')}: 'margin' and "
                    f"'detect_dist' must be set either per-pair in collision_limits: "
                    f"or globally in parameters:."
                )
            margin = float(margin)
            detect_dist = float(detect_dist)
            _limit_summaries.append(
                (limit_cfg.get('name', '?'), cbf_gain, margin, detect_dist, issf_eps))

            if self.use_issf:
                limit_obj = ISSfCollisionAvoidanceLimit(
                    model=self.model,
                    geom_pairs=geom_pairs,
                    gain=cbf_gain,
                    minimum_distance_from_collisions=margin,
                    collision_detection_distance=detect_dist,
                    issf_epsilon=issf_eps,
                )
            else:
                limit_obj = mink.CollisionAvoidanceLimit(
                    model=self.model,
                    geom_pairs=geom_pairs,
                    gain=cbf_gain,
                    minimum_distance_from_collisions=margin,
                    collision_detection_distance=detect_dist,
                )
            self.all_collision_limits.append(limit_obj)

            # Hard QP inequality (mink's native collision avoidance constraint).
            if self.use_collision_constraint:
                self.ik_limits.append(limit_obj)

        if verbose and _limit_summaries:
            print(f"[COLMO] Per-limit collision params ({len(_limit_summaries)} limits; "
                  f"per-entry values, falling back to parameters: defaults):")
            for name, g, m, dd, eps in _limit_summaries:
                extra = f", issf_epsilon={eps:g}" if self.use_issf else ""
                print(f"[COLMO]   '{name}': cbf_gain={g:g}, margin={m:g}, "
                      f"detect_dist={dd:g}{extra}")

        self.setup_retarget_configuration()

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        
        # targets: tasks that require set_target() every frame (from human_data)
        # solver : tasks actually passed to solve_ik() (targets + regularizers/barriers)
        self.tasks1_targets = []
        self.tasks1_solver = []

        self.tasks2_targets = []
        self.tasks2_solver = [] 
        
        # reset mappings
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.task_to_human_body1 = {}
        self.task_to_human_body2 = {}

        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}


        # ----------------------------
        # Table 1: build FrameTasks
        # ----------------------------
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight == 0 and rot_weight == 0:
                continue

            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=self.lm_damping,
            )
            
            # NOTE: Multiple robot tasks may track the same human body part. 
            # Using the Task object as a key prevents data loss from duplicate body names.
            self.human_body_to_task1[body_name] = task
            self.task_to_human_body1[task] = body_name

            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets1[body_name] = R.from_quat(rot_offset, scalar_first=True)

            self.tasks1_targets.append(task)
            self.tasks1_solver.append(task)

        # ----------------------------
        # Table 2: build FrameTasks
        # ----------------------------
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight == 0 and rot_weight == 0:
                continue

            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=self.lm_damping,
            )

            self.human_body_to_task2[body_name] = task
            self.task_to_human_body2[task] = body_name

            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = R.from_quat(rot_offset, scalar_first=True)

            self.tasks2_targets.append(task)
            self.tasks2_solver.append(task)


    def update_targets(self, human_data):
        # scale/offset human data
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)

        # Apply Table 1 spatial offsets (only for bodies with defined offset entries)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)

        self.scaled_human_data = human_data

        # CORE: Iterate through all tasks in Table 1 and set IK targets using the mapping
        if self.use_ik_match_table1:
            for task in self.tasks1_targets:
                body_name = self.task_to_human_body1[task]
                if body_name not in human_data:
                    # Skip target update and log a warning if the required body data is missing.
                    print(f"[WARN] human_data missing key for tasks1: {body_name}")
                    continue
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for task in self.tasks2_targets:
                body_name = self.task_to_human_body2[task]
                if body_name not in human_data:
                    print(f"[WARN] human_data missing key for tasks2: {body_name}")
                    continue
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    # Utility function to monitor and log active collisions during the IK process
    def log_collision_warning(self, stage_name, num_iter):
        mj.mj_fwdPosition(self.model, self.configuration.data)
        
        fromto = np.zeros(6, dtype=np.float64)

        for limit_obj in self.all_collision_limits:
            current_limit = limit_obj.minimum_distance_from_collisions
            
            for id_a, id_b in limit_obj.geom_id_pairs:
                if id_a == -1 or id_b == -1:
                    continue

                # Compute the shortest distance between two geometries
                dist = mj.mj_geomDistance(self.model, self.configuration.data, id_a, id_b, 8.0, fromto)

                # Log a detailed warning if a penetration (collision) is detected
                if dist <= -0.02:
                    name_a = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, id_a)
                    name_b = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, id_b)
                    
                    p1 = self.configuration.data.geom_xpos[id_a]
                    p2 = self.configuration.data.geom_xpos[id_b]
                    c_dist = np.linalg.norm(p1 - p2)
                    
                    msg = (f"[bold red][COLLISION][/bold red] {stage_name} | Iter:{num_iter} | "
                           f"{name_a} <-> {name_b} | "
                           f"Dist:{dist:.4f} (Limit:{current_limit:.3f}) | CenterDist:{c_dist:.4f}")
                    print(msg)

    def _solve_ik(self, tasks, dt):
        """Velocity from one IK QP solve. Routes through the soft-constraint solver when
        any soft limit (position / velocity / acceleration) is configured, otherwise
        mink's standard hard-only solve."""
        if self._soft_limits:
            return self._solve_ik_soft(tasks, dt)
        return mink.solve_ik(self.configuration, tasks, dt, self.solver,
                             damping=self.damping, limits=self.ik_limits)

    def _solve_ik_soft(self, tasks, dt):
        """Solve the IK QP with the configured soft limits injected as DAQP SOFT rows
        (sense=8, sharing rho_soft). The hard rows (collision CBF / foot contact, plus
        position/velocity when not configured soft — all in self.ik_limits) stay hard, so
        the QP is always feasible and each soft limit is violated only (and minimally)
        where it would otherwise conflict with a hard wall. Returns v = Δq/dt.

        Mirrors mink.build_ik's objective/inequality assembly, then calls daqp directly
        so the per-row constraint `sense` can be set (mink/qpsolvers expose no soft flag).
        """
        cfg = self.configuration
        H, c = _compute_qp_objective(cfg, tasks, self.damping)
        G_hard, h_hard = _compute_qp_inequalities(cfg, self.ik_limits, dt)

        G_list, h_list, sense_list = [], [], []
        if G_hard is not None:
            G_list.append(G_hard)
            h_list.append(h_hard)
            sense_list.append(np.zeros(h_hard.shape[0], dtype=c_int))      # 0 = hard
        for lim in self._soft_limits:
            cons = lim.compute_qp_inequalities(cfg, dt)
            if cons.inactive:
                continue
            G_list.append(cons.G)
            h_list.append(cons.h)
            sense_list.append(np.full(cons.h.shape[0], 8, dtype=c_int))    # 8 = soft

        nv = self.model.nv
        if G_list:
            A = np.ascontiguousarray(np.vstack(G_list))
            bupper = np.ascontiguousarray(np.hstack(h_list))
            blower = np.full(bupper.shape[0], -1e30)
            sense = np.concatenate(sense_list)
        else:
            A = np.zeros((0, nv)); bupper = np.zeros(0); blower = np.zeros(0)
            sense = np.zeros(0, dtype=c_int)

        x, _obj, flag, _info = daqp.solve(
            np.ascontiguousarray(H), np.ascontiguousarray(c),
            A, bupper, blower, sense, rho_soft=self.soft_rho,
        )
        if flag > 0:                       # 1 = optimal, 2 = soft-optimal (both solved)
            return x / dt
        # Soft constraints cannot by themselves make the QP infeasible, so flag<=0 means
        # the HARD limits (collision CBF / foot contact, plus any hard position/velocity)
        # conflict on their own. Skip this iteration's update rather than crashing.
        if self.verbose:
            print(f"[COLMO][SoftIK] DAQP exitflag={flag}: hard limits infeasible, "
                  f"skipping IK update this iteration.")
        return np.zeros(nv)

    def _solve_ik_stages(self, dt, max_iter):
        """Run the stage1 + stage2 IK convergence loops with the given iteration cap."""
        if self.use_ik_match_table1 and len(self.tasks1_solver) > 0:
            num_iter = 0
            curr_error = self.error1()
            while num_iter < max_iter:
                vel1 = self._solve_ik(self.tasks1_solver, dt)
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                if abs(curr_error - next_error) < self.ik_tol:
                    break
                curr_error = next_error
                num_iter += 1
            self.log_collision_warning("Table1", num_iter)
        if self.use_ik_match_table2 and len(self.tasks2_solver) > 0:
            num_iter = 0
            curr_error = self.error2()
            while num_iter < max_iter:
                vel2 = self._solve_ik(self.tasks2_solver, dt)
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                if abs(curr_error - next_error) < self.ik_tol:
                    break
                curr_error = next_error
                num_iter += 1
            self.log_collision_warning("Table2", num_iter)

    def retarget(self, human_data, frame_idx=None):
        """Retarget ONE human frame to a robot qpos.

        frame_idx indexes `human_data` within the motion passed to
        adjust_hips_scale_for_motion(); it selects this frame's entry in the "per_frame"
        base-speed-capped root trajectory. Leave it None for sequential playback and an
        internal counter advances on its own (it is reset by adjust_hips_scale_for_motion);
        pass it explicitly when frames are visited out of order or replayed, e.g. a looping
        or scrubbing viewer. Ignored when no base-speed cap trajectory is active.
        """
        self._frame_idx = self._auto_frame_idx if frame_idx is None else int(frame_idx)
        self._auto_frame_idx = self._frame_idx + 1

        self.update_targets(human_data)
        dt = self.configuration.model.opt.timestep

        if not self._warmup_done:
            # Ground calibration folded into the first-frame warmup: the warmup settles the
            # robot from rest onto frame 0's (still-floating) targets and converges within a
            # few dozen iters. Partway through we MEASURE how far the robot's foot actually
            # floats (FK), fold that into ground_offset, and re-set the targets so the
            # REMAINING warmup iterations settle the foot straight down onto the ground -- no
            # separate calibration pass, no second warmup. Needs a roughly grounded frame 0.
            calib_at = min(self.ground_calib_warmup_iter, self._warmup_iters // 2)
            do_calib = (not self._ground_calibrated) and bool(self.ground_anchor_bodies)
            for k in range(self._warmup_iters):
                if do_calib and k == calib_at:
                    gap = self.measure_foot_float()
                    if gap is not None:
                        self.ground_offset = self.ground_offset + gap
                        self.update_targets(human_data)     # re-target, now grounded
                        if self.verbose:
                            print(f"[COLMO][Ground] Warmup calibration @iter {k}: measured "
                                  f"foot float {gap:+.4f} m -> ground_offset="
                                  f"{self.ground_offset:+.4f} m")
                    self._ground_calibrated = True
                vel = mink.solve_ik(
                    self.configuration,
                    self.tasks1_solver,
                    dt,
                    self.solver,
                    damping=self.damping,
                    limits=None,
                )
                self.configuration.integrate_inplace(vel, dt)
            self._warmup_done = True

        # Frame acceleration limit: latch the frame-start pose and the previous frame's NET
        # displacement so the limit bounds |Δqpos_t - Δqpos_{t-1}| across all the inner IK
        # iterations (see FrameAccelerationLimit).
        if self.accel_limit is not None:
            self._accel_q_start = self.configuration.q.copy()
            self.accel_limit.set_frame(self._accel_q_start, self._accel_dq_prev)

        # Frame velocity limit: latch the previous frame's final pose as the anchor for the
        # net per-frame displacement bound |qpos_t - qpos_{t-1}| <= v_max/fps.
        if self.frame_velocity_limit is not None:
            self.frame_velocity_limit.set_frame(self.configuration.q.copy())

        # Normal IK solve with the configured iteration cap.
        self._solve_ik_stages(dt, self.max_iter)

        mj.mj_fwdPosition(self.model, self.configuration.data)
        mj.mj_forward(self.model, self.configuration.data)

        # Store this frame's NET tangent displacement (q ⊖ q_frame_start) for the next
        # frame's acceleration limit.
        if self.accel_limit is not None:
            dq_now = np.zeros(self.model.nv)
            mj.mj_differentiatePos(self.model, dq_now, 1.0, self._accel_q_start,
                                   self.configuration.q)
            self._accel_dq_prev = dq_now

        qpos = self.configuration.data.qpos.copy()
        return qpos

    def error1(self):
        errs = []
        unset = []
        for task in self.tasks1_targets:
            try:
                errs.append(task.compute_error(self.configuration))
            except TargetNotSet:
                name = getattr(task, "frame_name", None) or getattr(task, "name", None) or task.__class__.__name__
                unset.append(str(name))

        if unset:
            print(f"[WARN] FrameTask target not set (tasks1, skipped): {unset[:10]}{' ...' if len(unset) > 10 else ''}")

        if len(errs) == 0:
            return 0.0
        return np.linalg.norm(np.concatenate(errs))


    def error2(self):
        errs = []
        unset = []
        for task in self.tasks2_targets:
            try:
                errs.append(task.compute_error(self.configuration))
            except TargetNotSet:
                name = getattr(task, "frame_name", None) or getattr(task, "name", None) or task.__class__.__name__
                unset.append(str(name))

        if unset:
            print(f"[WARN] FrameTask target not set (tasks2, skipped): {unset[:10]}{' ...' if len(unset) > 10 else ''}")

        if len(errs) == 0:
            return 0.0
        return np.linalg.norm(np.concatenate(errs))


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        if self._root_xy_traj is not None:
            # "per_frame" base-speed cap: the horizontal root comes from the pre-integrated
            # saturated trajectory (adjust_hips_scale_for_motion) instead of a scalar multiply.
            # z keeps the plain scaling so height / gait bob are untouched. The modulo keeps
            # looping viewers (bvh_to_robot.py --loop) in range.
            scaled_root_pos = np.asarray(scaled_root_pos, dtype=float).copy()
            scaled_root_pos[:2] = self._root_xy_traj[self._frame_idx % len(self._root_xy_traj)]

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global

    def adjust_hips_scale_for_motion(self, frames):
        """Apply the base horizontal-speed cap (max_base_horizontal_speed) to ONE motion.
        Call ONCE with the full human frame list before the retarget loop; it also resets the
        internal frame counter, so it doubles as the "new motion starts here" marker. Only the
        Hips HORIZONTAL motion is ever touched -- z, gait and every limb scale are untouched.

        Two modes (parameters.base_speed_cap_mode, see BASE_SPEED_CAP_MODES):

        "per_frame" (default) -- increment-level saturation. Precomputes the saturated root xy
            trajectory via saturate_root_xy(): every frame under the cap keeps the nominal scale
            EXACTLY, and only the offending frames are compressed, landing exactly on the cap.
            A clip that never exceeds the cap reproduces the uncapped result to roundoff (~1e-14,
            the cumulative sum vs. a direct multiply).

        "clip" -- the older clip-level scaling: one scalar shrink min(nominal, limit/peak_p99.5)
            baked into human_scale_table[root] for the whole motion. Uniform (so a clip keeps a
            single consistent apparent body size) but one fast burst shrinks every frame, and the
            p99.5 peak means the top 0.5% of frames still exceed the cap.

        NOTE (both modes): this bounds the SCALED-REFERENCE base speed. The robot base is
        unconstrained in the IK, so the actual base can still overshoot ~1.5x on the hardest
        frames -- use a lower limit if a hard robot-base cap is required."""
        self._root_xy_traj = None
        self._frame_idx = 0
        self._auto_frame_idx = 0
        if len(frames) < 2:
            return
        root = self.human_root_name
        xy = np.array([np.asarray(frames[t][root][0], float)[:2] for t in range(len(frames))])
        limit = self.max_base_horizontal_speed
        # EFFECTIVE scale = config Hips xy * height_ratio (actual_human_height/assumption), so a
        # config of 0.70 with LAFAN's 1.75m human gives 0.70*0.972 = 0.6806. Both the config
        # value and the ratio are printed below so the number is never surprising.
        rr = self.height_ratio if self.height_ratio else 1.0
        cur = np.array(self.human_scale_table[root], dtype=float)
        if cur.ndim == 0:                                 # scalar config -> per-axis (xy vs z)
            cur = np.array([float(cur)] * 3)
        nom_xy = float(cur[0])
        cfg_nom = nom_xy / rr                             # config Hips xy (pre-ratio nominal)

        if self.base_speed_cap_mode == "per_frame":
            self._root_xy_traj, info = saturate_root_xy(xy, cur[:2], limit, self.motion_fps)
            if limit is None:
                tail = "no cap set"
            elif info["saturated_frac"] > 0:
                shrink = info["path_len"] / max(info["path_len_nominal"], 1e-9)
                tail = (f"saturated {100 * info['saturated_frac']:.1f}% of frames at {limit} m/s "
                        f"-> travel {100 * shrink:.0f}% of nominal")
            else:
                tail = f"under {limit} m/s cap -> fully faithful"
            print(f"[COLMO] Motion max horizontal speed = {info['v_raw_max']:.2f} m/s "
                  f"(p99.5 {info['v_raw_p995']:.2f}) | per-frame cap, base xy scale = "
                  f"{nom_xy:.4f} nominal (config {cfg_nom:.3f} x height_ratio {rr:.3f}) | "
                  f"base peak {info['base_peak']:.2f} m/s  [{tail}]")
            return

        # --- "clip" mode: one scalar shrink for the whole motion -------------------------
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) * self.motion_fps   # raw Hips h-speed
        vmax = float(v.max())
        peak = float(np.percentile(v, 99.5))              # robust peak used for the cap
        capped = False
        if limit is not None and float(limit) > 0 and peak > 1e-9:
            cap_eff = float(limit) / peak                 # max allowed EFFECTIVE xy scale
            cur[0] = min(float(cur[0]), cap_eff)
            cur[1] = min(float(cur[1]), cap_eff)
            self.human_scale_table[root] = cur
            capped = float(cur[0]) < nom_xy - 1e-9
        base_peak = float(cur[0]) * peak                  # scaled-reference base h-speed peak
        if limit is None:
            tail = "no cap set"
        elif capped:
            tail = f"CAPPED (would exceed {limit} m/s) -> base peak {base_peak:.2f} m/s"
        else:
            tail = f"under {limit} m/s cap -> faithful (base peak {base_peak:.2f} m/s)"
        print(f"[COLMO] Motion max horizontal speed = {vmax:.2f} m/s (p99.5 {peak:.2f}) | "
              f"clip cap, base xy scale = {float(cur[0]):.4f}  "
              f"(config {cfg_nom:.3f} x height_ratio {rr:.3f}"
              f"{f' -> capped {cap_eff:.4f}' if capped else ''})  [{tail}]")

    def capped_root_xy(self, frame_idx):
        """Saturated (base-speed-capped) root xy for `frame_idx`, or None when no per-frame
        cap trajectory is active ("clip" mode, no cap, or adjust_hips_scale_for_motion never
        ran). Viewers that draw the human skeleton themselves should use this to place the
        root instead of scaling the raw position, so the overlay travels with the robot."""
        if self._root_xy_traj is None:
            return None
        return self._root_xy_traj[int(frame_idx) % len(self._root_xy_traj)]

    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data

    def _ensure_foot_geom_cache(self):
        """Cache, once, the floor height and each ground-anchor body's mesh geoms
        (geom_id + local vertices) so measure_foot_float can find the real foot's lowest
        world-z. The link MESHES (not the coarse capsule/sphere collision primitives) are
        the robot's real foot surface, matching the ground metric in eval_kinematic."""
        if self._foot_mesh_cache is not None:
            return
        planes = [g for g in range(self.model.ngeom)
                  if self.model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE]
        self._floor_z = float(self.model.geom_pos[planes[0], 2]) if planes else 0.0
        cache = []
        for body_name in self.ground_anchor_bodies:
            try:
                bid = self.model.body(body_name).id
            except KeyError:
                print(f"[COLMO][Ground] WARNING: anchor body '{body_name}' not in model.")
                continue
            for g in range(self.model.ngeom):
                if int(self.model.geom_bodyid[g]) != bid:
                    continue
                if self.model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                    continue
                mid = int(self.model.geom_dataid[g])
                adr = int(self.model.mesh_vertadr[mid])
                num = int(self.model.mesh_vertnum[mid])
                cache.append((g, self.model.mesh_vert[adr:adr + num].astype(np.float64)))
        self._foot_mesh_cache = cache

    def measure_foot_float(self):
        """Height [m] of the robot's real foot meshes above the floor in the CURRENT
        configuration (>=0 = floating, <0 = penetrating). Reads the FK already computed by
        retarget() (mj_fwdPosition/mj_forward), so call it right after a retarget() step.
        Returns None if no ground-anchor foot bodies/meshes are available."""
        self._ensure_foot_geom_cache()
        if not self._foot_mesh_cache:
            return None
        data = self.configuration.data
        lowest = np.inf
        for gid, verts in self._foot_mesh_cache:
            zrow = data.geom_xmat[gid].reshape(3, 3)[2]          # world-z row
            z = float((verts @ zrow + data.geom_xpos[gid][2]).min())
            lowest = min(lowest, z)
        return lowest - self._floor_z
