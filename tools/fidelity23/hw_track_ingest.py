#!/usr/bin/env python3
"""hw_track_ingest.py — synthesize a fidelity23 metrics.json from the 1 Hz
[track(:NAME)] snapshots in a hardware controller log (H0 2026-09-30: the
900 Hz DDS capture died, the controller's own per-second snapshots survived).

Joint-space metrics only (no IMU quat in [track] lines): per-segment joint
error, per-joint p50/p90 vs the reference chain, kick-window support-ankle
depth, joint amplitudes. World-frame metrics are emitted as NaN and the
workbench shows "—".

The reference chain / segments / kick window are selected by the logged
motion name (take2* -> take2_full_r4 chain incl. turn windows; anything else,
including the legacy single-motion "[track] f=k/1832" lines, -> take6_full_r4).

Usage: hw_track_ingest.py <wbc_log> <round-index-1-based|0=list> <out.metrics.json>
"""
import json
import os
import re
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "refs")
JN = ["l_hip_p", "l_hip_r", "l_hip_y", "l_knee", "l_ank_p", "l_ank_r",
      "r_hip_p", "r_hip_r", "r_hip_y", "r_knee", "r_ank_p", "r_ank_r", "waist_yaw",
      "l_shoulder_p", "l_shoulder_r", "l_elbow", "l_wrist", "l_shoulder_yaw",
      "r_shoulder_p", "r_shoulder_r", "r_elbow", "r_wrist", "r_shoulder_yaw"]
MOTIONS = {
    "take2": {
        "ref": os.path.join(DATA, "take2_full_r4_50hz.npz"),
        "segments": [("开场", 0.0, 8.0), ("律动", 8.0, 27.0), ("踢腿", 27.0, 44.0), ("结尾", 44.0, 58.14),
                     ("turn1", 19.9, 22.18), ("turn2", 31.1, 34.2), ("turn3", 41.48, 44.44)],
        "kw_frames": (1425, 2250),      # 28.5-45.0 s
        "dance_s": 58.0,
    },
    "take6": {
        "ref": os.path.join(DATA, "take6_full_r4_50hz.npz"),
        "segments": [("开场", 0.0, 8.0), ("踢腿", 8.0, 12.0), ("律动", 12.0, 28.0), ("结尾", 28.0, 36.6)],
        "kw_frames": (400, 620),
        "dance_s": 36.0,
    },
}
TRACK_RE = re.compile(r"\[track(?::(?P<name>\S+))?\] f=(?P<f>\d+)/(?P<n>\d+) "
                      r"worst j(?P<j>\d+) e(?P<e>[0-9.]+) q=(?P<q>.*)")
ANK_LIMIT = -0.87267


def motion_of(name):
    return "take2" if name and "take2" in name.lower() else "take6"


def parse_rounds(path):
    lines = open(path, "rb").read().decode("utf-8", "replace").splitlines()
    rounds, cur = [], None
    for l in lines:
        if "23DoF motion tracking" in l:
            cur = {"name": None, "frames": [], "worst": [], "q": []}
            rounds.append(cur)
        m = TRACK_RE.search(l)
        if m and cur is not None:
            f = int(m.group("f"))
            # a frame reset inside one tracking state entry marks a new dance
            if cur["frames"] and f < cur["frames"][-1]:
                cur = {"name": None, "frames": [], "worst": [], "q": []}
                rounds.append(cur)
            name = m.group("name")
            if name:
                cur["name"] = name
            cur["frames"].append(f)
            cur["worst"].append(float(m.group("e")))
            cur["q"].append([float(x) for x in m.group("q").split()][:23])
        if "Damping latched" in l and cur is not None:
            cur["latch"] = l.strip()[:100]
        if "motion complete" in l and cur is not None:
            cur["complete"] = True
    return rounds


def main():
    log, idx = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    rounds = parse_rounds(log)
    if not rounds:
        sys.exit("no [track] rounds found")
    if idx == 0:
        for i, r in enumerate(rounds, 1):
            f = r["frames"]
            print(f"round {i}: name={r.get('name')} frames {min(f)}..{max(f)} n={len(f)} "
                  f"complete={r.get('complete', False)}")
        return
    if not out:
        sys.exit("output path required for round ingest")
    r = rounds[idx - 1]
    name = r.get("name") or ""
    mo = MOTIONS[motion_of(name)]
    ref = np.load(mo["ref"])["joint_pos"]
    f = np.array(r["frames"])
    q = np.array(r["q"])
    td = f / 50.0
    err = np.abs(q - ref[f])
    dance = td < mo["dance_s"]
    kw = (f >= mo["kw_frames"][0]) & (f < mo["kw_frames"][1])
    nan = float("nan")
    metrics = {
        "version": "HW_" + (name or "take6"), "tag": out.split("/")[-1].replace(".metrics.json", ""),
        "ref_chain": os.path.basename(mo["ref"]), "t_entry": 0.0, "t_entry_source": "track-frames",
        "date": "2026-10-09", "in_service": False, "hardware": True,
        "latches": 0 if r.get("complete") else 1,
        "note": "hardware 1 Hz [track] snapshots; joint-space only. "
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
        "joints": {"names": JN,
                   "err_p50": [float(np.percentile(err[dance, j], 50)) for j in range(23)],
                   "err_p90": [float(np.percentile(err[dance, j], 90)) for j in range(23)],
                   "std_ratio": [float(np.std(q[dance, j]) / max(np.std(ref[f[dance], j]), 1e-6)) for j in range(23)]},
        "series": {"td": td.tolist(),
                   "worst_err": np.array(r["worst"]).tolist(),
                   "ank_p": q[:, 10].tolist(),
                   "ank_p_ref": ref[f, 10].tolist(),
                   "q": None},
        "routing": [],
    }
    for nm, a, b in mo["segments"]:
        m = (td >= a) & (td < b) & dance
        metrics["segments"][nm] = {
            "mpjpe_p50": nan, "mpjpe_p90": nan, "fsr_L": nan, "fsr_R": nan,
            "joint_err_p50": float(np.percentile(err[m].max(axis=1), 50)) if m.any() else nan,
            "leg_err_p50": float(np.percentile(err[m][:, :12].max(axis=1), 50)) if m.any() else nan,
            "success": bool(r.get("complete") and m.any() and np.percentile(err[m].max(axis=1), 50) < 0.5)}
    json.dump(metrics, open(out, "w"), ensure_ascii=False, indent=1)
    s = metrics["summary"]
    print("%s: motion %s frames %d-%d complete=%s joint_err_mean %.3f kw_ank_p_min %.3f ankle_margin %.3f" % (
        metrics["tag"], name or "(legacy)", f.min(), f.max(), r.get("complete", False), s["joint_err_mean"],
        s["kw_ank_p_min"], s["ankle_margin_rad"]))


if __name__ == "__main__":
    main()
