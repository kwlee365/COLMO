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
                upper_bound[idx] = (cbf_gain * (dist - min_dist) / dt) - issf_margin + self.bound_relaxation
            else:
                upper_bound[idx] = -issf_margin + self.bound_relaxation
            sign = -1.0 if dist >= 0 else 1.0
            coefficient_matrix[idx] = sign * row
        return mink.Constraint(G=coefficient_matrix, h=upper_bound)


class FootContactLimit(mink.Limit):
    """
    Hard foot contact zero-velocity constraint, implemented as QP inequalities.

    Contact detection is based on the human motion position difference per frame:

        delta = ||human_pos_t - human_pos_{t-1}||

    If  delta  <=  threshold * dt  (i.e. human foot speed <= threshold [m/s]):
        Enforce  |J_c(q)[XY] @ v| <= velocity_bound   (hard QP inequality)

    Two-sided bound  |J_c @ v| <= velocity_bound  in G @ v <= h form.
    """
    def __init__(
        self,
        model: mj.MjModel,
        contact_points: list,       # list of (body_id, local_pos_np)
        human_body_names: list,     # list of human body names, one per contact point
        threshold: float = 0.01,    # m/s — contact activates when human foot speed <= threshold
        velocity_bound: float = 0.0,
        fps: float = 30,            # motion frame rate used to convert threshold m/s -> m/frame
        name: str = "foot_contact",
    ):
        self.model = model
        self.contact_points = contact_points
        self.human_body_names = human_body_names
        self.threshold = threshold
        self.velocity_bound = velocity_bound  # max contact-point XY velocity [m/s]
        self.fps = fps
        self.name = name
        self._prev_human_pos = {}              # human_body_name -> np.ndarray(3,)
        self._active_mask = [False] * len(contact_points)  # per-point contact state

    def update_from_human_motion(self, human_data: dict):
        """Update contact active mask from human motion position differences.

        human_data: dict of {human_body_name: (pos, rot)} for the current frame
        """
        threshold_pos = self.threshold / self.fps  # m/s threshold -> meters per frame
        for i, human_name in enumerate(self.human_body_names):
            if human_name not in human_data:
                self._active_mask[i] = False
                continue
            curr_pos = human_data[human_name][0]  # (3,) position
            if human_name in self._prev_human_pos:
                delta = np.linalg.norm(curr_pos - self._prev_human_pos[human_name])
                self._active_mask[i] = delta <= threshold_pos
            else:
                self._active_mask[i] = False  # no previous frame — cannot determine contact
            self._prev_human_pos[human_name] = curr_pos.copy()

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ):
        model = self.model
        data = configuration.data

        mj.mj_fwdPosition(model, data)

        G_rows, h_rows = [], []
        J_pos = np.zeros((3, model.nv))
        J_rot = np.zeros((3, model.nv))

        for i, (body_id, local_pos) in enumerate(self.contact_points):
            if not self._active_mask[i]:
                continue

            body_xpos = data.xpos[body_id]
            body_xmat = data.xmat[body_id].reshape(3, 3)
            world_point = body_xpos + body_xmat @ local_pos

            J_pos[:] = 0.0
            J_rot[:] = 0.0
            mj.mj_jac(model, data, J_pos, J_rot, world_point, body_id)

            J_xy = J_pos[:2]               # XY rows only, (2, nv)
            G_rows.append(J_xy)
            G_rows.append(-J_xy)
            h_rows.append(np.full(2, self.velocity_bound))
            h_rows.append(np.full(2, self.velocity_bound))

        if not G_rows:
            return mink.limits.Constraint()   # inactive (G=None, h=None)

        return mink.limits.Constraint(
            G=np.vstack(G_rows),
            h=np.concatenate(h_rows),
        )


def find_geoms(model, names):
    gids = []
    for n in names:
        gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)
        if gid != -1:
            gids.append(gid)
    return gids


class AccelerationLimit(mink.Limit):
    """Hard QP limit on the per-frame change in joint velocity (acceleration).

    Enforces, per actuated joint,

        |q_dot - q_dot_prev| <= qddot_max * dt

    where q_dot_prev is the previous frame's joint velocity. The IK QP decision
    variable is the tangent displacement Δq = q_dot * dt, and this codebase runs
    several solve_ik iterations per frame (integrating Δq each time). A naive box
    centered at q_dot_prev*dt would force every later iteration to keep moving at
    q_dot_prev (breaking convergence), so the bound is written on the NET tangent
    displacement accumulated since the frame start,

        d = q ⊖ q_frame_start,

    giving, for this iteration's Δq,

        v_prev*dt - a_max*dt^2 <= d + Δq <= v_prev*dt + a_max*dt^2.

    As d approaches the bound the admissible Δq shrinks to zero, so the limit caps
    the frame velocity jump without preventing the inner loop from converging.
    Because it lives inside the same QP as the collision limits, the solver keeps
    collision safety (the CBF) AND the acceleration bound simultaneously.

    The free/floating-base joint IS supported: list it in ``accelerations`` and all
    6 base DoFs (3 translation + 3 rotation) get the given bound. ``mj_differentiatePos``
    handles the quaternion tangent for the rotational DoFs.
    """

    def __init__(self, model, accelerations):
        """
        Args:
            model: MuJoCo model.
            accelerations: dict joint_name -> max |qddot| ([rad]/[s^2] for hinge/ball
                rotation, [m]/[s^2] for slide/base translation). A free joint contributes
                its 6 base DoFs with the given (broadcast) bound.
        """
        self.model = model
        limit_list, index_list = [], []
        for joint_name, max_acc in accelerations.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            vadr = model.jnt_dofadr[jid]
            vdim = 6 if jnt_type == mj.mjtJoint.mjJNT_FREE else dof_width(int(jnt_type))
            max_acc = np.atleast_1d(max_acc)
            index_list.extend(range(vadr, vadr + vdim))
            limit_list.extend(np.broadcast_to(max_acc, (vdim,)).tolist())

        self.indices = np.array(index_list, dtype=int)
        self.limit = np.array(limit_list, dtype=float)
        nb = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if nb > 0 else None

        # Per-frame state, refreshed once per frame via set_frame().
        self.q_frame_start = None              # qpos at the start of the current frame
        self.v_prev = np.zeros(model.nv)       # previous frame's tangent velocity

    def set_frame(self, q_frame_start, v_prev):
        """Latch the frame-start configuration and the previous frame velocity."""
        self.q_frame_start = np.array(q_frame_start, dtype=float)
        self.v_prev = np.array(v_prev, dtype=float)

    def compute_qp_inequalities(self, configuration, dt):
        if self.projection_matrix is None or self.q_frame_start is None:
            return mink.Constraint()
        # Net tangent displacement so far this frame: d = q ⊖ q_frame_start.
        d = np.zeros(self.model.nv)
        mj.mj_differentiatePos(self.model, d, 1.0, self.q_frame_start, configuration.q)

        a_dt2 = self.limit * dt * dt                 # a_max * dt^2 (per limited joint)
        vprev_dt = self.v_prev[self.indices] * dt    # v_prev * dt
        d_lim = d[self.indices]

        # +Δq rows:  Δq <=  v_prev*dt + a*dt^2 - d
        # -Δq rows: -Δq <= -v_prev*dt + a*dt^2 + d
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([vprev_dt + a_dt2 - d_lim, -vprev_dt + a_dt2 + d_lim])
        return mink.Constraint(G=G, h=h)


class VelocityLimitAllDof(mink.Limit):
    """Per-iteration velocity box  |Δq| <= v_max * dt  over ALL DoFs, including the
    free/floating base (which mink.VelocityLimit skips). ``v_max_per_dof`` is a length-nv
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
        use_velocity_limit: bool=True,
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

        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

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
        # added to self.ik_limits once the YAML params are loaded; velocity may be hard or
        # soft (see velocity_limit_soft).
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
        self.vel_limit = _require_param(params, 'velocity_limit')
        # IK solver / loop tuning, externalized to YAML (back-compat defaults kept).
        self.solver = params.get('solver', 'daqp')
        self._warmup_iters = params.get('warmup_iters', 500)
        self.ik_tol = params.get('ik_convergence_tol', 1e-8)   # IK loop early-stop tol
        self.lm_damping = params.get('lm_damping', 1.0)        # per-FrameTask LM damping
        self.motion_fps = params.get('motion_fps', 30)         # foot-contact fps assumption
        self.ground_offset = params.get('ground_offset', -0.01)  # per-frame robot ground offset [m]
        # Per-frame joint-acceleration cap |q_dot - q_dot_prev| <= a_max*dt (soft QP
        # limit, see AccelerationLimit). None/<=0 disables it.
        self.acc_limit = params.get('acceleration_limit', None)
        # Optional PER-JOINT overrides of the global acceleration_limit (joint_name ->
        # max |qddot|, rad/s^2). Any actuated joint NOT listed keeps the global value.
        self.acc_limit_per_joint = params.get('acceleration_limit_per_joint', {}) or {}

        # --- Posture (nullspace) regularization ---------------------------------------
        # joint_name -> cost (low weight) and joint_name -> neutral target angle [rad].
        # A single mink.PostureTask carrying these per-DOF costs is added to the IK task
        # stack in setup_retarget_configuration(). Because its cost is LOW compared with
        # the FrameTask orientation costs, it only acts in the task NULLSPACE (and where a
        # tracking task goes singular / saturates), biasing the listed joints toward their
        # neutral angle. This breaks the reduced-DOF orientation local-minimum trap: e.g.
        # tracking shoulder_roll_link orientation drives only pitch+roll (yaw is a child
        # link), so a backward arm swing saturates shoulder pitch at its limit (+1.249)
        # and the local velocity-level IK cannot climb back out; the posture pull toward
        # neutral supplies the restoring gradient. Empty dict -> disabled.
        self.posture_cost_cfg = params.get('posture_cost', {}) or {}
        self.posture_target_cfg = params.get('posture_target', {}) or {}
        # Optional STATE-DEPENDENT scheduling of the posture cost (see setup_retarget_
        # configuration / _apply_posture_schedule). When enabled, each joint's cost is not
        # constant but ramps from `min_cost` (at its target) up to its posture_cost value
        # (at its upper joint limit, toward which it traps) as a function of the current
        # angle -- a one-sided soft barrier. Disabled -> constant posture_cost.
        self.posture_schedule_cfg = params.get('posture_schedule', {}) or {}

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
        # The VELOCITY and ACCELERATION limits can be applied as SOFT DAQP constraints
        # (sense=8, shared rho_soft) instead of hard walls, so the QP never goes
        # infeasible and they yield MINIMALLY only where they would fight a hard
        # collision / foot-contact constraint. The POSITION (joint-config) limit is
        # always HARD, so the robot never exceeds its joint ranges.
        self.velocity_limit_soft = bool(params.get('velocity_limit_soft', False))

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

        # --- Velocity limit -----------------------------------------------------------
        self.velocity_limit_obj = None
        if use_velocity_limit:
            if self.velocity_limit_soft:
                # SOFT box over all DoFs: the same velocity_limit on the actuated joints
                # AND the free/floating-base 6 DoFs; any other DoF is left unconstrained.
                v_max = np.full(self.model.nv, np.inf)
                for _, dadr, w in actuated:
                    v_max[dadr:dadr + w] = self.vel_limit
                for d in base_dofs:
                    v_max[d] = self.vel_limit
                self.velocity_limit_obj = VelocityLimitAllDof(self.model, v_max)  # SOFT
            else:
                VELOCITY_LIMITS = {n: self.vel_limit for n, _, _ in actuated}
                self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))  # HARD

        # --- Acceleration limit (soft only) -------------------------------------------
        # Per-frame cap |q_dot - q_dot_prev| <= a_max*dt on the ACTUATED joints only,
        # applied as a SOFT DAQP constraint (sense=8) so the QP never becomes infeasible.
        # Enable/disable with the acceleration_limit_soft boolean; acceleration_limit sets
        # the DEFAULT magnitude for every actuated joint, and acceleration_limit_per_joint
        # overrides it for named joints. Softness via acceleration_softness (rho_soft,
        # shared by all soft rows; smaller -> nearer-hard).
        self.acceleration_limit_soft = bool(params.get('acceleration_limit_soft', True))
        self.accel_limit = None
        self.accel_rho_soft = float(params.get('acceleration_softness', 1e-6))
        if self.acceleration_limit_soft and self.acc_limit is not None and float(self.acc_limit) > 0.0:
            ACCEL_LIMITS = {n: float(self.acc_limit) for n, _, _ in actuated}
            # Per-joint overrides: must name an actuated joint; unknown names warn + skip.
            for jname, jacc in self.acc_limit_per_joint.items():
                if jname in ACCEL_LIMITS:
                    ACCEL_LIMITS[jname] = float(jacc)
                else:
                    print(f"[COLMO][Accel] WARNING: acceleration_limit_per_joint '{jname}' "
                          f"is not an actuated joint, skipping.")
            self.accel_limit = AccelerationLimit(self.model, ACCEL_LIMITS)
        # Previous-frame velocity (tangent space) consumed by the accel limit.
        self._accel_v_prev = np.zeros(self.model.nv)

        # --- Assemble hard vs soft sets -----------------------------------------------
        # Position limit is ALWAYS hard (like collision / foot-contact). Velocity may be
        # HARD (self.ik_limits) or SOFT (self._soft_limits, DAQP sense=8); acceleration is
        # soft-only.
        self.ik_limits.append(self.config_limit)                     # position: HARD
        self._soft_limits = []
        if self.velocity_limit_obj is not None:                      # set only when vel soft
            self._soft_limits.append(self.velocity_limit_obj)
        if self.accel_limit is not None:                             # soft-only
            self._soft_limits.append(self.accel_limit)

        if verbose:
            print(f"[COLMO] Final Parameters ->  Damping: {self.damping}, Max Iterations: {self.max_iter}")
            print(f"[COLMO] Position limit: hard | "
                  f"Velocity limit: {self.vel_limit} ({'SOFT' if self.velocity_limit_soft else 'hard'}) | "
                  f"soft rho_soft={self.accel_rho_soft}")
            accel_status = ('off' if self.accel_limit is None else f"{self.acc_limit}"
                            + (f" | per-joint: {self.acc_limit_per_joint}" if self.acc_limit_per_joint else ""))
            print(f"[COLMO] Acceleration limit (soft): {accel_status}")
            print(f"[COLMO] Collision mode: {self.collision_mode} "
                  f"(constraint={self.use_collision_constraint}"
                  f"{f', issf_epsilon={self.issf_epsilon}' if self.use_issf else ''})")
        
        # Resolve collision groups and individual geometries
        self.groups = cfg['groups']
        self.all_collision_limits = []


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

        # Store foot contact config for setup after ik_match_tables are loaded
        self._fc_cfg = cfg.get('foot_contact', {})

        self.setup_retarget_configuration()

        # Set up foot contact limit after ik_match_tables are available (human body names derived from them)
        self.foot_contact_limit = None
        fc_cfg = self._fc_cfg
        if fc_cfg.get('enabled', False):
            fc_threshold      = fc_cfg.get('threshold', 0.01)
            fc_velocity_bound = fc_cfg.get('velocity_bound', 0.0)
            fc_points_cfg     = fc_cfg.get('contact_points', [])

            # Build reverse mapping: robot_body_name -> human_body_name from IK tables
            robot_to_human = {}
            for robot_body, entry in self.ik_match_table1.items():
                robot_to_human[robot_body] = entry[0]
            for robot_body, entry in self.ik_match_table2.items():
                if robot_body not in robot_to_human:
                    robot_to_human[robot_body] = entry[0]

            contact_points = []
            human_body_names = []
            for cp in fc_points_cfg:
                body_name = cp['body_name']
                human_body_name = robot_to_human.get(body_name, '')
                if not human_body_name:
                    print(f"[COLMO][FootContact] WARNING: '{body_name}' not in IK match tables, skipping.")
                    continue
                local_pos = np.array(cp.get('local_pos', [0.0, 0.0, 0.0]), dtype=float)
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
                if body_id == -1:
                    print(f"[COLMO][FootContact] WARNING: robot body '{body_name}' not found in model, skipping.")
                    continue
                contact_points.append((body_id, local_pos))
                human_body_names.append(human_body_name)

            if contact_points:
                self.foot_contact_limit = FootContactLimit(
                    model=self.model,
                    contact_points=contact_points,
                    human_body_names=human_body_names,
                    threshold=fc_threshold,
                    velocity_bound=fc_velocity_bound,
                    fps=self.motion_fps,
                )
                self.ik_limits.append(self.foot_contact_limit)
                if verbose:
                    # contact_points contains resolved (body_id, local_pos); use human_body_names for display
                    robot_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid) for bid, _ in contact_points]
                    pairs = list(zip(robot_names, human_body_names))
                    print(f"[COLMO] FootContactLimit enabled | {pairs} | "
                          f"threshold: {fc_threshold} m/s | human_fps: {self.motion_fps} Hz | "
                          f"threshold_pos: {fc_threshold / self.motion_fps * 1000:.3f} mm/frame | "
                          f"velocity_bound: {fc_velocity_bound} m/s")
            else:
                print("[COLMO][FootContact] WARNING: no valid contact points found, limit disabled.")

        self.floor_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        foot_geoms_cfg = cfg.get('foot_geoms', {})
        left_candidates = foot_geoms_cfg.get('left', [])
        right_candidates = foot_geoms_cfg.get('right', [])

        if not left_candidates or not right_candidates:
            raise ValueError(
                f"'foot_geoms' must be defined in {collision_cfg_path} with non-empty "
                f"'left' and 'right' lists."
            )


        self.left_foot_gids  = find_geoms(self.model, left_candidates)
        self.right_foot_gids = find_geoms(self.model, right_candidates)

        if not self.left_foot_gids:
            raise ValueError(f"No left foot geom found. Tried {left_candidates}")
        if not self.right_foot_gids:
            raise ValueError(f"No right foot geom found. Tried {right_candidates}")

        

        assert self.floor_gid != -1, "geom 'floor' not found"

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

        # ----------------------------
        # Posture (nullspace) regularizer
        # ----------------------------
        # One low-cost mink.PostureTask biases the joints in self.posture_cost_cfg toward
        # a neutral posture (self.posture_target_cfg, default 0 rad = model qpos0). It is
        # appended to BOTH stage solver lists (tasks{1,2}_solver) but NOT the *_targets
        # lists: its target is CONSTANT (set once here) and it is a regularizer, not a
        # per-frame tracking target, so update_targets()/error{1,2}() must skip it.
        # Per-DOF cost is 0 everywhere except the listed joints, so it never perturbs the
        # other DoFs and stays subordinate to the FrameTask tracking (cost << ori cost).
        # posture_cost value is the MAX cost. With scheduling off it is applied constantly;
        # with scheduling on it is only reached at the joint's upper limit (see
        # _apply_posture_schedule). self._posture_sched holds one row per scheduled joint.
        self.posture_task = None
        self._posture_sched = []
        self.posture_schedule_enabled = bool(self.posture_schedule_cfg.get('enabled', False))
        sched_min = float(self.posture_schedule_cfg.get('min_cost', 0.0))
        sched_power = float(self.posture_schedule_cfg.get('power', 1.0))
        # Dead-zone: joint angle at which the cost STARTS rising above min_cost. Below it
        # the cost is min_cost (free tracking); it ramps min->max over [ramp_start, q_hi].
        # Default 0.0 -> ramp begins at the neutral target. Raise it to let the arm swing
        # further back before the barrier engages.
        sched_start = float(self.posture_schedule_cfg.get('ramp_start', 0.0))
        # When true, retarget() prints the live per-frame (pitch, frac, scheduled cost) for
        # each scheduled joint -- a debugging aid to see whether/where the barrier engages.
        self.posture_schedule_debug = bool(self.posture_schedule_cfg.get('debug', False))
        if self.posture_cost_cfg:
            cost = np.zeros(self.model.nv)
            target_q = self.model.qpos0.copy()
            applied = []
            for jname, jcost in self.posture_cost_cfg.items():
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                if jid == -1:
                    print(f"[COLMO][Posture] WARNING: joint '{jname}' not found, skipping.")
                    continue
                dof_adr = int(self.model.jnt_dofadr[jid])
                qpos_adr = int(self.model.jnt_qposadr[jid])
                target = float(self.posture_target_cfg.get(jname, 0.0))
                q_hi = float(self.model.jnt_range[jid][1])          # upper joint limit
                cost_max = float(jcost)
                target_q[qpos_adr] = target
                # Under scheduling start at min_cost (rest pitch ~ target); the constant
                # path uses cost_max directly. _apply_posture_schedule() updates it per solve.
                cost[dof_adr] = sched_min if self.posture_schedule_enabled else cost_max
                self._posture_sched.append((dof_adr, qpos_adr, sched_start, q_hi,
                                            cost_max, sched_min, sched_power, jname))
                applied.append((jname, cost_max, target, q_hi))
            if applied:
                self.posture_task = mink.PostureTask(self.model, cost=cost)
                self.posture_task.set_target(target_q)
                self.tasks1_solver.append(self.posture_task)
                self.tasks2_solver.append(self.posture_task)
                if self.verbose:
                    if self.posture_schedule_enabled:
                        print(f"[COLMO] PostureTask (SCHEDULED) enabled | min={sched_min} "
                              f"power={sched_power} | "
                              f"{[(n, f'max={c}', f'target={t:.2f}', f'q_hi={h:.3f}') for n, c, t, h in applied]}")
                    else:
                        print(f"[COLMO] PostureTask (constant) enabled | "
                              f"{[(n, f'cost={c}', f'target={t:.2f}rad') for n, c, t, _ in applied]}")


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

    def _apply_posture_schedule(self):
        """State-dependent posture cost. For each scheduled joint, ramp its posture cost
        from cost_min (at its target) up to cost_max (at its UPPER joint limit) as a
        function of the CURRENT joint angle:

            frac = clip((q - target) / (q_upper - target), 0, 1)
            cost = cost_min + (cost_max - cost_min) * frac**power

        This is a ONE-SIDED soft barrier: below/at the target the cost is cost_min (~0, so
        tracking is free), and it grows only as the joint approaches the limit toward which
        it traps (positive/backward pitch), where it pulls hard back toward neutral. Called
        every IK iteration so the cost tracks the evolving configuration. No-op when
        scheduling is disabled (the constant cost_max set at construction stays)."""
        if not self.posture_schedule_enabled or self.posture_task is None:
            return
        q = self.configuration.q
        for dof_adr, qpos_adr, ramp_start, q_hi, cost_max, cost_min, power, _jname in self._posture_sched:
            span = q_hi - ramp_start
            if span <= 0.0:
                continue
            frac = (q[qpos_adr] - ramp_start) / span
            frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
            self.posture_task.cost[dof_adr] = cost_min + (cost_max - cost_min) * (frac ** power)

    def _solve_ik(self, tasks, dt):
        """Velocity from one IK QP solve. Routes through the soft-constraint solver when
        any soft limit (position / velocity / acceleration) is configured, otherwise
        mink's standard hard-only solve."""
        self._apply_posture_schedule()
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
            A, bupper, blower, sense, rho_soft=self.accel_rho_soft,
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

    def retarget(self, human_data):
        self.update_targets(human_data)
        dt = self.configuration.model.opt.timestep

        if not self._warmup_done:
            for _ in range(self._warmup_iters):
                self._apply_posture_schedule()   # keep scheduled cost consistent during warmup
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

        # Update foot contact mask once per frame from human motion position differences
        if self.foot_contact_limit is not None:
            self.foot_contact_limit.update_from_human_motion(self.scaled_human_data)

        # Acceleration limit: latch the frame-start pose and the previous frame's
        # velocity so the limit bounds the NET frame velocity jump across all the
        # inner IK iterations (see AccelerationLimit).
        if self.accel_limit is not None:
            self._accel_q_start = self.configuration.q.copy()
            self.accel_limit.set_frame(self._accel_q_start, self._accel_v_prev)

        # Normal IK solve with the configured iteration cap.
        self._solve_ik_stages(dt, self.max_iter)

        mj.mj_fwdPosition(self.model, self.configuration.data)
        mj.mj_forward(self.model, self.configuration.data)

        # Store this frame's net joint velocity (q ⊖ q_frame_start)/dt for the next
        # frame's acceleration limit.
        if self.accel_limit is not None:
            v_now = np.zeros(self.model.nv)
            mj.mj_differentiatePos(self.model, v_now, dt, self._accel_q_start,
                                   self.configuration.q)
            self._accel_v_prev = v_now

        # Debug: print the live scheduled posture cost per frame so it is easy to see
        # whether/where the barrier engages (e.g. pitch stuck high while cost is still ~0
        # means `power` is too steep / `ramp_start` too high). One line per frame.
        if self.posture_schedule_enabled and self.posture_schedule_debug and self.posture_task is not None:
            q = self.configuration.q
            parts = []
            for dof_adr, qpos_adr, ramp_start, q_hi, cost_max, cost_min, power, jname in self._posture_sched:
                span = q_hi - ramp_start
                frac = 0.0 if span <= 0.0 else (q[qpos_adr] - ramp_start) / span
                frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
                parts.append(f"{jname}: q={q[qpos_adr]:+.3f}/{q_hi:.3f} "
                             f"frac={frac:.2f} cost={self.posture_task.cost[dof_adr]:6.2f}")
            print(f"[COLMO][PostureSched] {' | '.join(parts)}")

        qpos = self.configuration.data.qpos.copy()
        return qpos

    def get_contact_point_positions(self) -> list:
        """Return [(world_pos, is_active), ...] for each foot contact point.
        is_active=True  → constraint was enforced in the last IK solve (foot in contact).
        is_active=False → foot moving freely.
        Returns empty list if foot contact is disabled.
        """
        if self.foot_contact_limit is None:
            return []
        data = self.configuration.data
        result = []
        for i, (body_id, local_pos) in enumerate(self.foot_contact_limit.contact_points):
            body_xpos = data.xpos[body_id]
            body_xmat = data.xmat[body_id].reshape(3, 3)
            pos = body_xpos + body_xmat @ local_pos
            result.append((pos, self.foot_contact_limit._active_mask[i]))
        return result

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
