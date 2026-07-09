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

    Floating-base DoFs are ignored (mirrors mink.VelocityLimit).
    """

    def __init__(self, model, accelerations):
        """
        Args:
            model: MuJoCo model.
            accelerations: dict joint_name -> max |qddot| ([rad]/[s^2] for hinge,
                [m]/[s^2] for slide).
        """
        self.model = model
        limit_list, index_list = [], []
        for joint_name, max_acc in accelerations.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            if jnt_type == mj.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Free joint {joint_name} is not supported")
            vadr = model.jnt_dofadr[jid]
            vdim = dof_width(int(jnt_type))
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

        # Initialize IK constraints starting with joint configuration limits
        self.ik_limits = [mink.ConfigurationLimit(self.model)]

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

        if use_velocity_limit:
            VELOCITY_LIMITS = {}
            for a_id in range(self.model.nu):
                j_id = int(self.model.actuator_trnid[a_id, 0])
                if j_id < 0:
                    continue
                j_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)
                if j_name is None:
                    continue
                VELOCITY_LIMITS[j_name] = self.vel_limit

            # Hard velocity bound
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        # Per-frame acceleration bound |q_dot - q_dot_prev| <= a_max*dt on the actuated
        # joints. Applied as a SOFT QP constraint (DAQP sense=8) rather than a hard
        # inequality: it is honoured exactly whenever feasible but yields (minimal
        # violation) when it would otherwise conflict with the hard collision CBF or a
        # joint limit, so the QP never becomes infeasible. NOT added to self.ik_limits;
        # the soft rows are injected in _solve_ik_soft_accel(). Opt-in via
        # parameters.acceleration_limit; softness via parameters.acceleration_softness
        # (DAQP rho_soft: smaller -> nearer-hard / stronger smoothing, larger -> softer).
        self.accel_limit = None
        self.accel_rho_soft = float(params.get('acceleration_softness', 1e-6))
        if self.acc_limit is not None and float(self.acc_limit) > 0.0:
            ACCEL_LIMITS = {}
            for a_id in range(self.model.nu):
                j_id = int(self.model.actuator_trnid[a_id, 0])
                if j_id < 0:
                    continue
                j_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)
                if j_name is None:
                    continue
                ACCEL_LIMITS[j_name] = float(self.acc_limit)
            self.accel_limit = AccelerationLimit(self.model, ACCEL_LIMITS)
        # Previous-frame joint velocity (tangent space) consumed by the accel limit.
        self._accel_v_prev = np.zeros(self.model.nv)

        if verbose:
            print(f"[COLMO] Final Parameters ->  Damping: {self.damping}, Max Iterations: {self.max_iter}")
            print(f"Velocity Limit: {self.vel_limit}")
            print(f"[COLMO] Acceleration Limit (soft): "
                  f"{f'{self.acc_limit} (rho_soft={self.accel_rho_soft})' if self.accel_limit is not None else 'off'}")
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
        """Velocity from one IK QP solve. Routes through the soft-acceleration solver
        when the accel limit is active, otherwise mink's standard hard-only solve."""
        if self.accel_limit is not None:
            return self._solve_ik_soft_accel(tasks, dt)
        return mink.solve_ik(self.configuration, tasks, dt, self.solver,
                             damping=self.damping, limits=self.ik_limits)

    def _solve_ik_soft_accel(self, tasks, dt):
        """Solve the IK QP with the acceleration limit injected as DAQP SOFT
        constraints (sense=8). The hard rows (config / collision CBF / velocity / foot
        limits in self.ik_limits) stay hard; the acceleration box rows are soft, so the
        QP is always feasible and the accel bound is violated only (and minimally) where
        it would otherwise conflict with a hard wall. Returns v = Δq/dt.

        Mirrors mink.build_ik's objective/inequality assembly, then calls daqp directly
        so the per-row constraint `sense` can be set (mink/qpsolvers expose no soft flag).
        """
        cfg = self.configuration
        H, c = _compute_qp_objective(cfg, tasks, self.damping)
        G_hard, h_hard = _compute_qp_inequalities(cfg, self.ik_limits, dt)
        accel_cons = self.accel_limit.compute_qp_inequalities(cfg, dt)

        G_list, h_list, sense_list = [], [], []
        if G_hard is not None:
            G_list.append(G_hard)
            h_list.append(h_hard)
            sense_list.append(np.zeros(h_hard.shape[0], dtype=c_int))      # 0 = hard
        if not accel_cons.inactive:
            G_list.append(accel_cons.G)
            h_list.append(accel_cons.h)
            sense_list.append(np.full(accel_cons.h.shape[0], 8, dtype=c_int))  # 8 = soft

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
        # A soft accel constraint cannot by itself make the QP infeasible, so flag<=0
        # means the HARD limits (config + collision) conflict on their own. Skip this
        # iteration's update rather than crashing with NoSolutionFound.
        if self.verbose:
            print(f"[COLMO][SoftAccel] DAQP exitflag={flag}: hard limits infeasible, "
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
