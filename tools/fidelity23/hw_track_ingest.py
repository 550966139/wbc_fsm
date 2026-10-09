#!/usr/bin/env python3
"""hw_track_ingest.py — synthesize a fidelity23 metrics.json from the 1 Hz
[track] snapshots in a hardware controller log (H0 2026-09-30 session: the
900 Hz DDS capture died, the controller's own per-second snapshots survived).

Joint-space metrics only (no IMU quat in [track] lines): per-segment joint
error, per-joint p50/p90 vs the reference chain, kick-window support-ankle
depth, joint amplitudes. World-frame metrics are emitted as NaN and the
workbench shows "—".

Usage: hw_track_ingest.py <wbc_log> <round-index-1-based> <out.metrics.json>
"""
import json
import re
import sys

import numpy as np

REF = np.load("/home/wsk/dsh-workspace/www.unitree.com/wbc_fsm/tools/fidelity23/data/refs/take6_full_r4_50hz.npz")["joint_pos"]
SEGMENTS = [("开场", 0.0, 8.0), ("踢腿", 8.0, 12.0), ("律动", 12.0, 28.0), ("结尾", 28.0, 36.6)]
KW = (400, 620)
ANK_LIMIT = -0.87267


def parse_rounds(path):
    lines = open(path, "rb").read().decode("utf-8", "replace").splitlines()
    rounds, cur = [], None
    for l in lines:
        if "23DoF motion tracking" in l:
            cur = {"frames": [], "worst": []}
            rounds.append(cur)
        m = re.search(r"\[track\] f=(\d+)/1832 worst j(\d+) e([0-9.]+) q=(.*)", l)
        if m and cur is not None:
            cur["frames"].append(int(m.group(1)))
            cur["worst"].append(float(m.group(3)))
            cur.setdefault("q", []).append([float(x) for x in m.group(4).split()][:23])
        if "Damping latched" in l and cur is not None:
            cur["latch"] = l.strip()[:100]
        if "motion complete" in l and cur is not None:
            cur["complete"] = True
    return rounds


def main():
    log, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    r = parse_rounds(log)[idx - 1]
    f = np.array(r["frames"])
    q = np.array(r["q"])
    td = f / 50.0
    err = np.abs(q - REF[f])
    dance = td < 36.0
    jn = ["l_hip_p", "l_hip_r", "l_hip_y", "l_knee", "l_ank_p", "l_ank_r",
          "r_hip_p", "r_hip_r", "r_hip_y", "r_knee", "r_ank_p", "r_ank_r", "waist_yaw",
          "l_shoulder_p", "l_shoulder_r", "l_elbow", "l_wrist", "l_shoulder_yaw",
          "r_shoulder_p", "r_shoulder_r", "r_elbow", "r_wrist", "r_shoulder_yaw"]
    kw = (f >= KW[0]) & (f < KW[1])
    nan = float("nan")
    metrics = {
        "version": "F10_HW", "tag": out.split("/")[-1].replace(".metrics.json", ""),
        "ref_chain": "take6_full_r4_50hz.npz", "t_entry": 0.0, "t_entry_source": "track-frames",
        "date": "2026-09-30", "in_service": False, "hardware": True,
        "latches": 0 if r.get("complete") else 1,
        "note": "H0 hardware session, 1 Hz [track] snapshots (900 Hz DDS capture died); joint-space only. "
                + ("Full dance." if r.get("complete") else "Latched: " + r.get("latch", "?")),
        "summary": {
            "mpjpe_world_p50": nan, "mpjpe_world_p90": nan,
            "fsr_ref_L": nan, "fsr_ref_R": nan,
            "slide_ref_p50_L": nan, "slide_ref_p50_R": nan,
            "float_L": nan, "float_R": nan, "penetration": nan,
            "kick_apex_ratio_L": nan, "kick_apex_cm_L": nan,
            "drift_net": nan, "drift_path": nan,
            "ankle_margin_rad": float(min(q[dance, 4].min(), q[dance, 10].min()) - ANK_LIMIT),
            "joint_err_p50_max": float(np.median(err[dance], axis=0).max()),
            "joint_err_mean": float(err[dance].mean()),
            "kw_ank_p_min": float(min(q[kw, 4].min() if kw.any() else 9, q[kw, 10].min() if kw.any() else 9)),
            "complete": bool(r.get("complete")),
            "hw_latch": r.get("latch"),
        },
        "segments": {},
        "joints": {"names": jn,
                   "err_p50": [float(np.percentile(err[dance, j], 50)) for j in range(23)],
                   "err_p90": [float(np.percentile(err[dance, j], 90)) for j in range(23)],
                   "std_ratio": [float(np.std(q[dance, j]) / max(np.std(REF[f[dance], j]), 1e-6)) for j in range(23)]},
        "series": {"td": td.tolist(),
                   "worst_err": np.array(r["worst"]).tolist(),
                   "ank_p": q[:, 10].tolist(),
                   "ank_p_ref": REF[f, 10].tolist(),
                   "q": None},
        "routing": [],
    }
    for nm, a, b in SEGMENTS:
        m = (td >= a) & (td < b) & dance
        metrics["segments"][nm] = {
            "mpjpe_p50": nan, "mpjpe_p90": nan, "fsr_L": nan, "fsr_R": nan,
            "joint_err_p50": float(np.percentile(err[m].max(axis=1), 50)) if m.any() else nan,
            "success": bool(r.get("complete") and m.any() and np.percentile(err[m].max(axis=1), 50) < 0.5)}
    json.dump(metrics, open(out, "w"), ensure_ascii=False, indent=1)
    s = metrics["summary"]
    print("%s: frames %d-%d complete=%s joint_err_mean %.3f kw_ank_p_min %.3f ankle_margin %.3f" % (
        metrics["tag"], f.min(), f.max(), r.get("complete", False), s["joint_err_mean"],
        s["kw_ank_p_min"], s["ankle_margin_rad"]))


if __name__ == "__main__":
    main()
