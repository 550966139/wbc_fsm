#!/usr/bin/env python
"""measure_smplx.py - SMPL-X (GVHMR output) estimation-quality ruler (MC S2).

Input: the npz produced by 224:dsh-workspace/mc_dump_smplx.py from a GVHMR
hmr4d_results.pt (see S2 of docs/motion-calibration.md):

  joints      (N, 55, 3)  world, gravity-aligned (z up)
  names       (55,)       SMPL-X joint names
  body_pose   (N, 63)     SMPL-X body pose params (axis-angle)
  global_orient (N, 3)    root orientation params
  transl      (N, 3)      world translation
  betas, human_height

Measures the estimation-side diseases the calibration layer targets:
  - ground level + per-foot sole clearance (float / penetration)
  - contact schedule (sole near estimated ground)
  - planted-foot skate (ankle XY speed while contact)
  - root/pose jitter (frame-to-frame param steps)
  - stance CoM offset + point-mass ZMP offset vs contact feet (diagnostic;
    the embodiment-correct ZMP gate lives in measure.py, G1 domain)

Anthropometry is scaled by human_height (no per-motion constants).
"""

import argparse
import json

import numpy as np

FPS = 30.0
DT = 1.0 / FPS
G = 9.81
SMOOTH_K = 9
V = 1   # vertical axis: GVHMR gravity-view world is Y-up
H = (0, 2)  # horizontal plane axes
ANKLE_H = 0.050  # ankle joint above sole, fraction of human height
BALL_H = 0.012   # foot (ball) joint above sole, fraction of human height
STANCE_SPEED = 0.35  # m/s ankle speed below which the foot counts as planted
                    # (velocity criterion: robust to the world-root vertical
                    #  wander that makes a fixed ground plane unusable)


def hanning_smooth(a, axis=0):
    ker = np.hanning(2 * SMOOTH_K + 1)
    ker /= ker.sum()
    return np.apply_along_axis(lambda x: np.convolve(x, ker, "same"), axis, a)


def stats_pct(v):
    v = np.asarray(v)
    if len(v) == 0:
        return None
    return {
        "n": int(len(v)),
        "p5": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--label", default=None)
    ap.add_argument("--window", default=None, help="frame window A:B")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    J = d["joints"].astype(np.float64)
    names = [str(x) for x in d["names"]]
    transl = d["transl"].astype(np.float64)
    body_pose = d["body_pose"].astype(np.float64)
    h = float(d["human_height"])
    idx = {n: i for i, n in enumerate(names)}
    n = J.shape[0]

    def side_joints(side):
        return idx[f"{side}_ankle"], idx[f"{side}_foot"]

    # sole height proxy per side (vertical = Y)
    sole = {}
    for s in ("left", "right"):
        ai, fi = side_joints(s)
        sole[s] = np.minimum(J[:, fi, V] - BALL_H * h, J[:, ai, V] - ANKLE_H * h)
    sole_min = np.minimum(sole["left"], sole["right"])

    f0, f1 = (0, n) if not args.window else (int(x) for x in args.window.split(":"))

    # stance via velocity criterion (robust to world-root wander); per-segment
    # ground level exposes the wander itself
    stance = {}
    seg_table = {}
    for s in ("left", "right"):
        ai, fi = side_joints(s)
        vel3 = np.linalg.norm(hanning_smooth(np.gradient(J[:, ai, :], DT, axis=0, edge_order=2), axis=0), axis=1)
        m_ = vel3 < STANCE_SPEED
        segs, i = [], 0
        while i < n:
            if m_[i]:
                j = i
                while j + 1 < n and m_[j + 1]:
                    j += 1
                if j - i + 1 >= 5:
                    segs.append((i, j))
                i = j + 1
            else:
                i += 1
        stance[s] = np.zeros(n, bool)
        seg_table[s] = []
        for a, b in segs:
            stance[s][a:b + 1] = True
            seg_table[s].append({"a": a, "b": b, "ground_z": float(np.median(sole[s][a:b + 1]))})
    all_grounds = [g["ground_z"] for s in ("left", "right") for g in seg_table[s]]
    wander = float(np.percentile(all_grounds, 90) - np.percentile(all_grounds, 10)) if len(all_grounds) >= 2 else 0.0

    rep = {"file": args.npz, "label": args.label or args.npz.split("/")[-1], "frames": n,
           "human_height": h, "segments": seg_table, "ground_wander_m": wander}

    feet = {}
    for s in ("left", "right"):
        ai, fi = side_joints(s)
        m_ = stance[s]
        vxy = np.linalg.norm(np.diff(J[:, ai, :][:, H], axis=0), axis=1) * FPS
        vc = vxy[m_[1:]] if m_.any() else vxy[:0]
        # sole height relative to each stance segment's own ground level
        rel = np.full(n, np.nan)
        for g in seg_table[s]:
            rel[g["a"]:g["b"] + 1] = sole[s][g["a"]:g["b"] + 1] - g["ground_z"]
        relc = rel[~np.isnan(rel)]
        feet[s] = {
            "stance_fraction": float(m_[f0:f1].mean()),
            "segments": len(seg_table[s]),
            "sole_rel_segment_ground": stats_pct(relc),
            "skate_stance": stats_pct(vc),
        }
    rep["feet"] = feet

    # jitter
    rep["jitter"] = {
        "transl_step_m": stats_pct(np.linalg.norm(np.diff(transl, axis=0), axis=1)),
        "body_pose_step_rad": stats_pct(np.abs(np.diff(body_pose, axis=0)).max(axis=1)),
    }

    # CoM proxy (jitter-smoothed) vs stance feet center — static diagnostic.
    # A point-mass ZMP is meaningless here: estimation jitter at 30 fps gives
    # accel noise of the same order as g, so the ZMP proper is measured in the
    # G1 domain (measure.py) where the training-time dynamics model applies.
    core = [idx[k] for k in ("pelvis", "spine1", "spine2", "spine3", "neck", "head",
                             "left_hip", "right_hip", "left_knee", "right_knee",
                             "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
                             "left_elbow", "right_elbow", "left_wrist", "right_wrist") if k in idx]
    com = hanning_smooth(J[:, core, :].mean(axis=1))

    off = {"L": [], "R": [], "LR": []}
    for f in range(f0, f1):
        pts = []
        for s in ("left", "right"):
            if stance[s][f]:
                ai, fi = side_joints(s)
                pts += [J[f, ai, H], J[f, fi, H]]
        if not pts:
            continue
        key = ("L" if stance["left"][f] else "") + ("R" if stance["right"][f] else "")
        off[key].append(float(np.linalg.norm(com[f, H] - np.mean(np.array(pts), axis=0))))
    rep["com_offset_m"] = {k: stats_pct(v) for k, v in off.items() if v}

    print("=" * 78)
    print("%s  (%d frames, h=%.3f m, ground wander p10-p90 %.3f m over %d stance segs)"
          % (rep["label"], n, h, wander, len(all_grounds)))
    for s in ("left", "right"):
        f = rep["feet"][s]
        sc, sk = f["sole_rel_segment_ground"], f["skate_stance"]
        print("  %-5s stance %5.1f%% (%d segs) | sole rel seg-ground p50 %+0.4f min %+0.4f | skate p50 %.3f p90 %.3f p95 %.3f"
              % (s, f["stance_fraction"] * 100, f["segments"],
                 sc["p50"] if sc else float("nan"), sc["min"] if sc else float("nan"),
                 sk["p50"] if sk else float("nan"), sk["p90"] if sk else float("nan"),
                 sk["p95"] if sk else float("nan")))
    j = rep["jitter"]
    print("  jitter: transl p50 %.4f p95 %.4f m/frame | pose p50 %.4f p95 %.4f rad/frame"
          % (j["transl_step_m"]["p50"], j["transl_step_m"]["p95"],
             j["body_pose_step_rad"]["p50"], j["body_pose_step_rad"]["p95"]))
    for k, v in rep["com_offset_m"].items():
        print("  stance-CoM-offset %-2s n=%4d: p50 %.3f p90 %.3f max %.3f m" % (k, v["n"], v["p50"], v["p90"], v["max"]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
