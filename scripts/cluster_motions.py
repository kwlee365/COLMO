"""Cluster retargeted motions by kinematic similarity, to train ONE policy per group
instead of one per motion.

For each motion pkl it builds a compact kinematic feature vector (root locomotion,
verticality, per-body-group joint activity, range of motion, duration), z-scores the
features, and runs Ward hierarchical clustering (scipy). It prints:
  * the free name-prefix grouping (LAFAN1 action categories),
  * the merge-distance profile (where the "natural" number of clusters is),
  * the data-driven clusters at the requested / suggested k,
and writes motion->features and motion->cluster CSVs.

Usage:
    python scripts/cluster_motions.py --folder results/g1/lafan1/colmo --k 11
    python scripts/cluster_motions.py --folder results/g1/lafan1/colmo   # auto-suggest k
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.transform import Rotation as R

from collision_free_motion_retargeting.params import ROBOT_XML_DICT


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                return super().find_class("numpy.core" + module[len("numpy._core"):], name)
            if module.startswith("numpy.core"):
                return super().find_class("numpy._core" + module[len("numpy.core"):], name)
            raise


def load_pkl(path):
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def dof_group_indices(robot):
    """Split actuated-joint (dof_pos) indices into leg / arm / waist / head by joint name."""
    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot].resolve()))
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
             if m.jnt_type[j] != mj.mjtJoint.mjJNT_FREE]
    grp = {"leg": [], "arm": [], "waist": [], "head": []}
    for i, n in enumerate(names):
        n = n or ""
        if any(k in n for k in ("hip", "knee", "ankle")):
            grp["leg"].append(i)
        elif any(k in n for k in ("shoulder", "elbow", "wrist")):
            grp["arm"].append(i)
        elif "waist" in n:
            grp["waist"].append(i)
        elif "head" in n or "neck" in n:
            grp["head"].append(i)
    return grp, names


FEATURE_NAMES = [
    "speed_mean", "speed_p95", "dist_total", "yaw_rate",       # locomotion / turning
    "z_std", "z_range", "z_mean_rel",                          # verticality (jump/fall/ground)
    "jvel_mean", "jvel_p95",                                   # overall dynamics
    "leg_speed", "arm_speed", "waist_speed",                   # per-body-group activity
    "leg_rom", "arm_rom",                                      # range of motion
]


def extract_features(pkl, grp):
    root = np.asarray(pkl["root_pos"], np.float64)      # (T,3)
    rot = np.asarray(pkl["root_rot"], np.float64)       # (T,4) xyzw
    dof = np.asarray(pkl["dof_pos"], np.float64)        # (T,ndof)
    fps = float(pkl.get("fps", 30))
    dt = 1.0 / fps
    T = len(dof)

    dxy = np.diff(root[:, :2], axis=0)
    speed = np.linalg.norm(dxy, axis=1) / dt            # m/s planar
    z = root[:, 2]
    yaw = R.from_quat(rot).as_euler("xyz")[:, 2]
    yaw_rate = np.abs(np.diff(np.unwrap(yaw))) / dt

    dq = np.diff(dof, axis=0)
    dq = (dq + np.pi) % (2 * np.pi) - np.pi              # shortest-angle wrap
    jvel = np.abs(dq) / dt                               # (T-1, ndof)
    rom = dof.max(0) - dof.min(0)                        # (ndof,)

    def gspeed(idx):
        return jvel[:, idx].mean() if idx else 0.0

    def grom(idx):
        return rom[idx].mean() if idx else 0.0

    return np.array([
        speed.mean(), np.percentile(speed, 95),
        float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        yaw_rate.mean(),
        z.std(), z.max() - z.min(), z.mean() - z.min(),
        jvel.mean(), np.percentile(jvel, 95),
        gspeed(grp["leg"]), gspeed(grp["arm"]), gspeed(grp["waist"]),
        grom(grp["leg"]), grom(grp["arm"]),
    ], dtype=np.float64)


def action_prefix(name):
    """LAFAN1 action category: strip trailing subject and the take number (walk3 -> walk)."""
    base = re.sub(r"_subject\d+$", "", name)
    return re.sub(r"\d+$", "", base)


# --------------------------------------------------------------------------- #
# DTW-based similarity (time-warped trajectory distance).
# --------------------------------------------------------------------------- #
# Key bodies whose 3D positions form the per-frame pose descriptor for DTW.
KEY_BODIES = ["pelvis", "torso_link",
              "left_ankle_roll_link", "right_ankle_roll_link",
              "left_knee_link", "right_knee_link",
              "left_rubber_hand", "right_rubber_hand",
              "left_elbow_link", "right_elbow_link",
              "left_shoulder_roll_link", "right_shoulder_roll_link"]


def dtw_sequences(pkls, robot, n_frames):
    """Per-motion (n_frames, 3*K) pose sequence: each frame is the K key-body 3D positions in
    a CANONICAL root frame -- global xy translation and heading (yaw) removed, but root height
    and lean (pitch/roll) kept (so jumps/falls/crouches stay distinct). This makes the DTW
    compare the actual body choreography, invariant to where/which-way the motion happens."""
    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot].resolve()))
    d = mj.MjData(m)
    bids = []
    for b in KEY_BODIES:
        try:
            bids.append(m.body(b).id)
        except KeyError:
            pass
    seqs = []
    for p in pkls:
        pk = load_pkl(p)
        root = np.asarray(pk["root_pos"], np.float64)
        rot = np.asarray(pk["root_rot"], np.float64)      # xyzw
        dof = np.asarray(pk["dof_pos"], np.float64)
        T = len(dof)
        idx = np.linspace(0, T - 1, n_frames).astype(int)
        S = np.zeros((n_frames, 3 * len(bids)))
        for f, t in enumerate(idx):
            roll, pitch, _yaw = R.from_quat(rot[t]).as_euler("xyz")   # drop yaw
            q_noyaw = R.from_euler("xyz", [roll, pitch, 0.0]).as_quat()  # xyzw
            d.qpos[:3] = (0.0, 0.0, root[t, 2])            # zero xy, keep height
            d.qpos[3:7] = q_noyaw[[3, 0, 1, 2]]            # wxyz
            d.qpos[7:] = dof[t]
            mj.mj_kinematics(m, d)
            S[f] = np.concatenate([d.xpos[b] for b in bids])
        seqs.append(S)
    return seqs


def dtw(A, B):
    """Normalized DTW distance between sequences A (Ta,D) and B (Tb,D) with Euclidean per-frame
    cost. Semi-vectorized DP (prev-row min vectorized; running min scanned). Normalized by the
    grid perimeter (Ta+Tb) so different-length motions stay comparable."""
    C = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))       # (Ta,Tb) frame costs
    Ta, Tb = C.shape
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        prev = D[i - 1]
        pm = np.minimum(prev[1:], prev[:-1])              # min(D[i-1,j], D[i-1,j-1]) for each j
        Ci = C[i - 1]
        row = D[i]
        rj = np.inf                                        # D[i,0] = inf
        for j in range(Tb):
            rj = Ci[j] + (pm[j] if pm[j] < rj else rj)     # + min(prev-row, D[i,j-1])
            row[j + 1] = rj
    return D[Ta, Tb] / (Ta + Tb)


def pairwise_dtw(seqs):
    """Condensed (upper-triangle) DTW distance vector for scipy linkage."""
    n = len(seqs)
    cond = np.zeros(n * (n - 1) // 2)
    k = 0
    total = len(cond)
    for i in range(n):
        for j in range(i + 1, n):
            cond[k] = dtw(seqs[i], seqs[j])
            k += 1
        if (i % 8 == 0) or i == n - 1:
            print(f"    DTW progress: {k}/{total} pairs", end="\r", flush=True)
    print()
    return cond


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="results/g1/lafan1/colmo",
                    help="Folder of motion pkls (one source; motion 'kind' is source-agnostic).")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--method", choices=["feature", "dtw"], default="feature",
                    help="'feature': Ward on 15 kinematic summary features (fast). "
                         "'dtw': time-warped trajectory distance on canonical key-body "
                         "positions (slower, stricter temporal similarity).")
    ap.add_argument("--frames", type=int, default=64,
                    help="Frames each motion is resampled to for --method dtw (default 64).")
    ap.add_argument("--k", type=int, default=None, help="Number of clusters (default: auto-suggest).")
    ap.add_argument("--out", default=None, help="CSV output prefix (default: <folder>/motion_clusters).")
    args = ap.parse_args()

    grp, jnames = dof_group_indices(args.robot)
    pkls = sorted(Path(args.folder).glob("*.pkl"))
    names = [p.stem for p in pkls]
    if not pkls:
        print(f"No pkls in {args.folder}")
        return
    X = np.array([extract_features(load_pkl(p), grp) for p in pkls])   # (N, F)

    # --- name-based grouping (free baseline) ---
    prefixes = [action_prefix(n) for n in names]
    from collections import Counter, defaultdict
    print(f"\n[cluster] {len(names)} motions from {args.folder}")
    print("\n=== Name-based action groups (free) ===")
    byp = defaultdict(list)
    for n, pr in zip(names, prefixes):
        byp[pr].append(n)
    for pr, ms in sorted(byp.items(), key=lambda kv: -len(kv[1])):
        print(f"  {pr:18} ({len(ms)})")
    print(f"  -> {len(byp)} action groups")

    # --- data-driven hierarchical clustering ---
    if args.method == "dtw":
        print(f"\n[cluster] method=dtw: building {args.frames}-frame canonical FK sequences "
              f"({len(pkls)} motions)...")
        seqs = dtw_sequences(pkls, args.robot, args.frames)
        print(f"[cluster] computing {len(pkls) * (len(pkls) - 1) // 2} pairwise DTW distances...")
        cond = pairwise_dtw(seqs)
        # complete linkage on the DTW distance matrix: compact, balanced clusters. (average/
        # single chain -- they peel outliers like floor/get-up motions off as singletons.)
        L = linkage(cond, method="complete")
    else:
        mu, sd = X.mean(0), X.std(0)                    # z-scored features
        sd[sd == 0] = 1.0
        L = linkage((X - mu) / sd, method="ward")

    # merge-distance profile: big jumps in the last merges = natural cut points
    md = L[:, 2]
    print("\n=== Merge-distance profile (last 12 merges; large jump = natural cut) ===")
    N = len(names)
    for i in range(max(0, len(md) - 12), len(md)):
        k_after = N - (i + 1)      # clusters remaining AFTER this merge
        print(f"  merge {i+1:2d}: dist={md[i]:6.2f}  -> {k_after} clusters")
    # suggest k = point of largest relative gap among the last merges
    tail = md[-15:]
    gaps = tail[1:] / np.maximum(tail[:-1], 1e-9)
    j = int(np.argmax(gaps))
    k_suggest = N - (len(md) - (len(md) - 15) - (j + 1))  # clusters after the pre-gap merge
    k_suggest = max(2, min(N - 1, N - (len(md) - (len(md) - len(tail) + j))))
    k = args.k or k_suggest
    print(f"\n[cluster] using k={k}" + ("" if args.k else f" (auto-suggested; override with --k)"))

    labels = fcluster(L, t=k, criterion="maxclust")
    print(f"\n=== Data-driven clusters (k={k}) ===")
    clusters = defaultdict(list)
    for n, pr, c in zip(names, prefixes, labels):
        clusters[c].append((n, pr))
    for c in sorted(clusters):
        ms = clusters[c]
        top = Counter(p for _, p in ms).most_common(2)
        tag = ", ".join(f"{p}×{cnt}" for p, cnt in top)
        print(f"  cluster {c:2d} (n={len(ms):2d}) [{tag}]:")
        for n, pr in sorted(ms):
            print(f"        {n}")

    # --- save CSVs ---
    out = args.out or str(Path(args.folder) / "motion_clusters")
    import csv
    with open(out + "_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion", "action", "cluster"] + FEATURE_NAMES)
        for n, pr, c, row in zip(names, prefixes, labels, X):
            w.writerow([n, pr, int(c)] + [f"{v:.5f}" for v in row])
    with open(out + "_assignments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "motion", "action"])
        for c in sorted(clusters):
            for n, pr in sorted(clusters[c]):
                w.writerow([int(c), n, pr])
    print(f"\n[cluster] wrote {out}_features.csv and {out}_assignments.csv")


if __name__ == "__main__":
    main()
