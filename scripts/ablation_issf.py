#!/usr/bin/env python
"""ISSf-CBF epsilon ablation for COLMO.

Sweeps the ISSf robustness scale ``epsilon`` (and the nominal CBF, epsilon=inf) applied to
BOTH the ground and the self collision limits, retargets a fixed set of LAFAN1 motions, and
reports the collision <-> motion-fidelity trade-off:

  * collision (real-mesh, same machinery as eval_kinematic.py):
      - P_ground / P_self : fraction of frames penetrating the floor / another link (%)
      - D_ground / D_self : mean penetration depth on penetrating frames (cm)
  * fidelity:
      - KpErr : mean keypoint POSITION tracking error over position-tracked IK targets (cm),
                i.e. || robot_body_origin - human_target ||, averaged over keypoints & frames.

Smaller epsilon => larger ISSf margin ( ||J_AB||^2 / eps ) => stronger collision avoidance,
but the robot is pushed further off the human targets (KpErr rises). epsilon=inf ('cbf') is
the plain CBF with no robustness margin. The ablation makes this trade-off explicit.

Usage:
    python scripts/ablation_issf.py                        # defaults: g1, 10 motions, full length
    python scripts/ablation_issf.py --max_frames 600       # cap frames per motion (fast probe)
    python scripts/ablation_issf.py --eps cbf 300 100 50   # custom epsilon list
"""
import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import mujoco as mj

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import eval_kinematic as ek  # noqa: E402  (reuse real-mesh collision machinery)
from collision_free_motion_retargeting import CollisionFreeMotionRetargeting as COLMO  # noqa: E402
from collision_free_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

# 10 collision-relevant + baseline motions: locomotion (walk/run), self-collision-prone
# (fight / dance / multipleActions), and ground-contact (ground / fall / pushAndFall).
DEFAULT_MOTIONS = [
    "walk1_subject1", "run2_subject4",
    "dance2_subject3", "dance1_subject2",
    "fight1_subject2", "multipleActions1_subject1",
    "jumps1_subject1",
    "ground1_subject1", "fallAndGetUp1_subject1", "pushAndFall1_subject1",
]
# inf == nominal CBF (collision_mode='cbf', no ISSf margin).
DEFAULT_EPS = ["cbf", "1000", "300", "100", "50", "10"]


def make_retargeter(tgt_robot, height, eps_token):
    """Fresh COLMO with epsilon applied to BOTH ground + self collision limits."""
    mode = "cbf" if eps_token == "cbf" else "issf"
    rt = COLMO(src_human="bvh_lafan1", tgt_robot=tgt_robot,
               actual_human_height=height, collision_mode=mode, verbose=False)
    if mode == "issf":
        eps = float(eps_token)
        for lim in rt.all_collision_limits:      # body_general + ground_collision, all pairs
            lim.issf_epsilon = eps
    return rt


def kp_target_ids(rt):
    """(robot_body_id, human_body_name) for every position-tracked Table-2 IK target."""
    out = []
    for frame_name, entry in rt.ik_match_table2.items():
        body_name, pos_weight = entry[0], entry[1]
        if not np.any(np.asarray(pos_weight, dtype=float) > 0):
            continue                             # orientation-only target -> not a position kp
        out.append((rt.model.body(frame_name).id, body_name))
    return out


def kp_error_cm(rt, kp):
    """Mean || robot_body_origin - human_target || over position-tracked keypoints (cm)."""
    data = rt.configuration.data
    errs = []
    for bid, hb in kp:
        if hb not in rt.scaled_human_data:
            continue
        tgt = np.asarray(rt.scaled_human_data[hb][0], dtype=float)
        errs.append(np.linalg.norm(data.xpos[bid] - tgt))
    return float(np.mean(errs)) * 100.0 if errs else np.nan


def collision_metrics(qpos_list, meta, ignore_self, gthr, thr):
    """Real-mesh ground/self penetration over a retargeted qpos trajectory."""
    model, data = meta["model"], meta["data"]
    N = len(qpos_list)
    ground = np.zeros(N)
    selfp = np.zeros(N)
    for t, q in enumerate(qpos_list):
        data.qpos[:] = q                         # COLMO qpos == eval qpos layout (pos,quat wxyz,dof)
        mj.mj_kinematics(model, data)
        mj.mj_collision(model, data)
        ground[t] = ek.ground_penetration_depth(meta)
        selfp[t] = ek.self_penetration_depth(meta, ignore_self)
    g = ground > gthr
    s = selfp > thr
    return dict(
        Pg=100.0 * float(g.mean()),
        Dg=100.0 * float(ground[g].mean()) if g.any() else np.nan,
        Ps=100.0 * float(s.mean()),
        Ds=100.0 * float(selfp[s].mean()) if s.any() else np.nan,
    )


def run_one(tgt_robot, frames, height, eps_token, meta, ignore_self, gthr, thr, max_frames):
    rt = make_retargeter(tgt_robot, height, eps_token)
    kp = kp_target_ids(rt)
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]
    qpos_list, kperrs = [], []
    for fr in frames:
        q = rt.retarget(fr)
        assert q.shape[0] == meta["model"].nq, (q.shape[0], meta["model"].nq)
        qpos_list.append(q.copy())
        kperrs.append(kp_error_cm(rt, kp))
    cm = collision_metrics(qpos_list, meta, ignore_self, gthr, thr)
    cm["KpErr"] = float(np.nanmean(kperrs))
    cm["N"] = len(qpos_list)
    return cm


def fmt_ms(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return "n/a".rjust(15)
    return f"{vals.mean():.2f}±{vals.std():.2f}".rjust(15)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--bvh_dir", default=str(HERE.parent / "human_motion" / "lafan1"))
    ap.add_argument("--motions", nargs="+", default=DEFAULT_MOTIONS)
    ap.add_argument("--eps", nargs="+", default=DEFAULT_EPS,
                    help="epsilon list; 'cbf' == inf (nominal CBF, no ISSf margin).")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames/motion (0 = full).")
    ap.add_argument("--pen_thresh", type=float, default=0.01)
    ap.add_argument("--ground_thresh", type=float, default=0.01)
    ap.add_argument("--out", default=str(HERE.parent / "ablation_issf.csv"))
    args = ap.parse_args()

    meta = ek.robot_metadata(args.robot)
    ignore_self = ek.structural_self_pairs(meta) | ek.same_chain_pairs(meta)
    gthr, thr = args.ground_thresh, args.pen_thresh
    print(f"[ablation] robot={args.robot} dof={meta['n_dof']} floor_z={meta['floor_z']:.3f} "
          f"| motions={len(args.motions)} eps={args.eps} max_frames={args.max_frames or 'full'}")

    # Preload each BVH once (frames + measured human height); reused across all epsilon.
    motions = {}
    for m in args.motions:
        p = pathlib.Path(args.bvh_dir) / f"{m}.bvh"
        frames, h = load_bvh_file(str(p), format="lafan1")
        motions[m] = (frames, h)
        print(f"  loaded {m:28} frames={len(frames):5d} height={h:.3f}")

    rows = []  # (eps, motion, cm)
    for eps in args.eps:
        for m in args.motions:
            frames, h = motions[m]
            t0 = time.time()
            cm = run_one(args.robot, frames, h, eps, meta, ignore_self, gthr, thr, args.max_frames)
            rows.append((eps, m, cm))
            print(f"[eps={eps:>5} {m:26}] N={cm['N']:5d} Pg={cm['Pg']:5.1f}% "
                  f"Dg={cm['Dg']:5.2f} Ps={cm['Ps']:5.1f}% Ds={cm['Ds']:5.2f} "
                  f"Kp={cm['KpErr']:5.2f}cm  ({time.time()-t0:.1f}s)")

    # Per-(eps,motion) CSV.
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "motion", "N", "P_ground%", "D_ground_cm",
                    "P_self%", "D_self_cm", "KpErr_cm"])
        for eps, m, cm in rows:
            w.writerow([eps, m, cm["N"], f"{cm['Pg']:.3f}", f"{cm['Dg']:.4f}",
                        f"{cm['Ps']:.3f}", f"{cm['Ds']:.4f}", f"{cm['KpErr']:.4f}"])
    print(f"\n[ablation] per-motion CSV -> {args.out}")

    # Summary: mean +/- std over motions, per epsilon.
    METR = [("P_ground %", "Pg"), ("D_ground cm", "Dg"),
            ("P_self %", "Ps"), ("D_self cm", "Ds"),
            ("KpErr cm", "KpErr")]
    width = 78 + 15 * max(0, len(args.eps) - 4)
    print("\n" + "=" * width)
    print("ISSf-CBF epsilon ablation on COLMO  (mean ± std over "
          f"{len(args.motions)} motions; all metrics LOWER = better except the")
    print("collision <-> fidelity TRADE-OFF: smaller eps -> less penetration but higher KpErr)")
    print("=" * width)
    hdr = f"{'metric':14}" + "".join(f"{('inf(cbf)' if e == 'cbf' else e):>15}" for e in args.eps)
    print(hdr)
    print("-" * width)
    for label, key in METR:
        line = f"{label:14}"
        for e in args.eps:
            line += fmt_ms([cm[key] for (ee, _mm, cm) in rows if ee == e])
        print(line)
    print("=" * width)


if __name__ == "__main__":
    main()
