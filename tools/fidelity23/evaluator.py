#!/usr/bin/env python3
"""Industry-aligned motion fidelity evaluator for the G1 23DoF tracking stack.

Metric taxonomy follows physics-based humanoid imitation practice
(Mimicking-Bench arXiv:2412.17730; BeyondMimic arXiv:2508.07200):

  physical plausibility : foot skating ratio (stance frames with horizontal
                          foot speed > 0.10 m/s), slide magnitude p50/p90,
                          ground penetration (depth/duration), float (sole
                          height > 5 mm on reference-contact frames)
  tracking fidelity     : world-frame 24-body position error (rigidly
                          aligned to the robot at entry, mirroring the
                          controller's entry yaw alignment), per-joint
                          position error, per-segment success
  amplitude fidelity    : per-joint std ratio (actual/reference), kick apex
                          ratio, pelvis ending drift, support-ankle margin

Usage (scan mode, per-version meta.json):
  evaluator.py --data-root data --xml <g1_23dof.xml> [--force]

Run layout expected under <data-root>/runs/<VERSION>/:
  <tag>.npz (t, q[29 motors], quat[wxyz])  <tag>_sport.npz (ts, pos)
  optional <tag>.t_entry (epoch float; missing entries are inferred by
  trajectory alignment)  meta.json (ref_npz/ref_csv chain mapping)
Outputs metrics.json next to each capture.

Routing suggestions map measured deficits onto this project's established
levers (R-series data surgery windows / adaptive-sampling emphasis / reward
std, with the F11/F12 ankle-fold caveat) so the numbers drive the next move.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco as mj

KEEP = list(range(13)) + list(range(15, 20)) + list(range(22, 27))
ANK_LIMIT = -0.87267
FOOT_SPEED_SKATE = 0.10          # m/s, skating threshold (GMD-lineage)
CONTACT_SOLE = 0.010             # m, actual-contact detection
PENETRATION = -0.002             # m
FLOAT_HEIGHT = 0.005             # m
MPJPE_SUCCESS = 0.15             # m, segment-success bar
SUB = 6                          # 900 Hz -> 150 Hz working rate
SERIES_SUB = 15                  # -> 10 Hz stored series
LEG_JOINTS = list(range(12))     # policy order j0-11 = both legs

# Motion-structure layout: segments / kick window / ending-drift start are
# take6 values and the DEFAULT for any version whose meta.json does not carry
# its own (a different dance re-derives them from its own reference chain;
# the dance/series windows derive from the last segment end).
DEFAULT_SEGMENTS = [("开场", 0.0, 8.0), ("踢腿", 8.0, 12.0), ("律动", 12.0, 28.0), ("结尾", 28.0, 36.6)]
DEFAULT_KICK_WINDOW = (8.5, 12.0)
DEFAULT_ENDING_DRIFT_FROM = 28.5


def layout_from_meta(meta):
    segs = [(s[0], float(s[1]), float(s[2])) for s in (meta.get("segments") or DEFAULT_SEGMENTS)]
    if not segs:
        raise ValueError("empty segments in meta.json")
    end = max(s[2] for s in segs)   # turn sub-windows may be listed after the finale
    turns = [(f"turn{i + 1}", float(a), float(b)) for i, (a, b) in enumerate(meta.get("turn_windows") or [])]
    return {"segments": segs, "turns": turns,
            "kick_window": tuple(float(v) for v in meta.get("kick_window", DEFAULT_KICK_WINDOW)),
            "ending_drift_from": float(meta.get("ending_drift_from", DEFAULT_ENDING_DRIFT_FROM)),
            "dance_start": 0.6, "dance_end": end - 0.6, "series_end": end + 1.4}


def yaw_of(qwxyz):
    return np.arctan2(2 * (qwxyz[0] * qwxyz[3] + qwxyz[1] * qwxyz[2]),
                      1 - 2 * (qwxyz[2] ** 2 + qwxyz[3] ** 2))


class Rig:
    def __init__(self, xml, ref_npz, ref_csv, layout):
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)
        self.foot_geoms = []
        self.foot_bodies = {"l": -1, "r": -1}
        for gi in range(self.model.ngeom):
            nm = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, gi) or ""
            body = self.model.geom_bodyid[gi]
            bn = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body) or ""
            if "ankle_roll" in bn and "collision" in nm:
                side = "l" if bn.startswith("l") else "r"
                self.foot_geoms.append((gi, side))
                self.foot_bodies[side] = body
        self.joint_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j + 1) for j in range(23)]
        self.body_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, self.model.nbody)]
        self.ref = np.load(ref_npz)
        self.ref_q = self.ref["joint_pos"]           # (1832,23)
        self.ref_body = self.ref["body_pos_w"]       # (1832,24,3)
        self.ref_root_yaw = np.array([yaw_of(q) for q in self.ref["body_quat_w"][:, 0]])
        # reference contact + kick apex from the 30 Hz chain CSV (proven route)
        chain = np.loadtxt(ref_csv, delimiter=",")
        n = chain.shape[0]
        cont = np.zeros(n, dtype=bool)
        soleL_ref = np.zeros(n)
        soleR_ref = np.zeros(n)
        for f in range(n):
            self.data.qpos[:3] = chain[f, :3]
            q = chain[f, 3:7]
            self.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
            self.data.qpos[7:] = chain[f, 7:]
            mj.mj_forward(self.model, self.data)
            for gi, s in self.foot_geoms:
                z = self.data.geom_xpos[gi][2] - self.model.geom_size[gi][2]
                if s == "l":
                    soleL_ref[f] = z
                else:
                    soleR_ref[f] = z
            cont[f] = min(soleL_ref[f], soleR_ref[f]) < 0.015
        self.ref_cont = cont
        tt = np.arange(n) / 30.0
        kw0, kw1 = layout["kick_window"]
        kw = (tt >= kw0) & (tt <= kw1)
        self.ref_kick_apex = {"l": float(soleL_ref[kw].max()), "r": float(soleR_ref[kw].max())}
        self.layout = layout

    def fk(self, pos, quat_wxyz, q29):
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = quat_wxyz
        self.data.qpos[7:] = [q29[k] for k in KEEP]
        mj.mj_forward(self.model, self.data)
        bodies = np.array([self.data.xpos[i] for i in range(1, self.model.nbody)])
        sole = {}
        for gi, s in self.foot_geoms:
            z = self.data.geom_xpos[gi][2] - self.model.geom_size[gi][2]
            sole.setdefault(s, min(sole.get(s, 9e9), z))
        xy = {}
        for gi, s in self.foot_geoms:
            xy.setdefault(s, self.data.geom_xpos[gi][:2].copy())
        return bodies, sole, xy


def infer_t_entry(t, q23, ref_q, hint_range=12.0):
    """Align actual joint trajectory onto the reference by minimizing the
    mean joint error over the first 20 s of the dance. Returns an absolute
    epoch (search starts 2 s before the capture)."""
    step = max(int(round((t[1] - t[0]) * 50)), 1)
    idx = np.arange(0, len(t), step)
    tq, qq = t[idx], q23[idx]
    best, best_te = None, t[0]
    for te in np.arange(t[0] - 2.0, t[0] + hint_range, 0.02):
        sel = (tq >= te + 1.0) & (tq < te + 21.0)
        if sel.sum() < 200:
            continue
        frames = np.clip(((tq[sel] - te) * 50).astype(int), 0, ref_q.shape[0] - 1)
        err = np.abs(qq[sel] - ref_q[frames]).mean()
        if best is None or err < best:
            best, best_te = err, float(te)
    return best_te


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def evaluate(capture_npz, rig, t_entry=None):
    cap = np.load(capture_npz, allow_pickle=True)
    t = cap["t"]
    q = cap["q"][:len(t)]
    quat = cap["quat"][:len(t)]
    q23 = q[:, KEEP]
    sp = np.load(capture_npz.replace(".npz", "_sport.npz"), allow_pickle=True)
    pos = np.stack([np.interp(t, sp["ts"], sp["pos"][:, i]) for i in range(3)], axis=1)
    if t_entry is None:
        t_entry = infer_t_entry(t, q23, rig.ref_q)
    td = t - t_entry

    idx = np.arange(0, len(t), SUB)
    tdw, qw23, qquat, ppos, qs = td[idx], q23[idx], quat[idx], pos[idx], q[idx]
    dt = np.gradient(tdw)

    bodies_all, soles, xys, robot_yaw = [], [], [], []
    for i in range(len(tdw)):
        b, sole, xy = rig.fk(ppos[i], qquat[i], qs[i])
        bodies_all.append(b)
        soles.append(sole)
        xys.append(xy)
        robot_yaw.append(yaw_of(qquat[i]))
    bodies = np.array(bodies_all)                       # (n,24,3)
    soleL = np.array([s["l"] for s in soles])
    soleR = np.array([s["r"] for s in soles])
    xyL = np.array([x["l"] for x in xys])
    xyR = np.array([x["r"] for x in xys])
    robot_yaw = np.array(robot_yaw)

    # reference interpolated onto the capture clock
    fr = np.clip((tdw * 50).astype(int), 0, rig.ref_q.shape[0] - 1)
    refq = rig.ref_q[fr]
    crow = np.clip((tdw * 30).astype(int), 0, len(rig.ref_cont) - 1)
    refcont = rig.ref_cont[crow]

    # world alignment at entry (mirrors controller's entry yaw alignment)
    i0 = np.searchsorted(tdw, 1.0)                      # dance actually moving by td=1s
    i0 = min(max(i0, 1), len(tdw) - 2)
    dyaw = robot_yaw[i0] - rig.ref_root_yaw[fr[i0]]
    Rz = np.array([[np.cos(dyaw), -np.sin(dyaw), 0], [np.sin(dyaw), np.cos(dyaw), 0], [0, 0, 1]])
    off = ppos[i0] - Rz @ rig.ref_body[fr[i0], 0]
    refb = np.einsum("ij,fkj->fki", Rz, rig.ref_body[fr]) + off
    mpjpe = np.linalg.norm(bodies - refb, axis=2).mean(axis=1)

    # physical plausibility
    lay = rig.layout
    dance = (tdw > lay["dance_start"]) & (tdw < lay["dance_end"])
    contL, contR = soleL < CONTACT_SOLE, soleR < CONTACT_SOLE
    vL = np.linalg.norm(np.gradient(xyL, axis=0) / dt[:, None], axis=1)
    vR = np.linalg.norm(np.gradient(xyR, axis=0) / dt[:, None], axis=1)
    fsr = {}
    for nm, v, c in (("L", vL, contL), ("R", vR, contR)):
        m = c & dance
        mr = refcont & dance
        fsr[nm] = {"ratio": float((v[m] > FOOT_SPEED_SKATE).mean()) if m.sum() else float("nan"),
                   "ratio_ref": float((v[mr] > FOOT_SPEED_SKATE).mean()) if mr.sum() else float("nan"),
                   "slide_p50": pct(v[m], 50), "slide_p90": pct(v[m], 90),
                   "slide_ref_p50": pct(v[mr], 50), "slide_ref_p90": pct(v[mr], 90),
                   "n": int(m.sum())}
    pen_m = dance & ((soleL < PENETRATION) | (soleR < PENETRATION))
    penetration = {"duration_frac": float(pen_m.mean()),
                   "depth_max": float(min(soleL[dance].min() if dance.any() else 0,
                                          soleR[dance].min() if dance.any() else 0))}
    fl = {}
    for nm, sole in (("L", soleL), ("R", soleR)):
        m = refcont & dance & (sole > FLOAT_HEIGHT)
        fl[nm] = {"duration_frac": float(m.mean() if dance.sum() else 0), "mean_height": pct(sole[m], 50)}

    # tracking
    jerr = np.abs(qw23 - refq)
    jnames = rig.joint_names
    joints = {"names": jnames,
              "err_p50": [pct(jerr[dance, j], 50) for j in range(23)],
              "err_p90": [pct(jerr[dance, j], 90) for j in range(23)],
              "std_ratio": [float(np.std(qw23[dance, j]) / max(np.std(refq[dance, j]), 1e-6)) for j in range(23)]}

    # kick apex + ending drift + ankle margin
    kw0, kw1 = lay["kick_window"]
    kw = (tdw > kw0) & (tdw < kw1)
    apex = {nm: float(sole[kw].max() * 100) for nm, sole in (("L", soleL), ("R", soleR))}
    apex_ratio = {"L": apex["L"] / (rig.ref_kick_apex["l"] * 100), "R": apex["R"] / (rig.ref_kick_apex["r"] * 100)}
    iw = np.searchsorted(tdw, lay["ending_drift_from"])
    iw = min(iw, len(tdw) - 1)
    drel = np.linalg.norm(ppos[iw:, :2] - ppos[iw, :2], axis=1)
    dpath = np.linalg.norm(np.diff(ppos[iw:, :2], axis=0), axis=1).sum()
    drift = {"net": float(drel[-1]), "max": float(drel.max()), "path": float(dpath)}
    ankmin = float(min(qw23[dance, 4].min(), qw23[dance, 10].min()))
    ankle_margin = ankmin - ANK_LIMIT

    # root-XY error vs the (entry-aligned) reference root + turn-window attribution
    root_err = np.linalg.norm(ppos[:, :2] - refb[:, 0, :2], axis=1)
    ref_yaw_t = rig.ref_root_yaw[fr]
    yaw_err = (robot_yaw - ref_yaw_t + np.pi) % (2 * np.pi) - np.pi
    turns = {}
    for nm, a, b in lay["turns"]:
        m = (tdw >= a) & (tdw < b)
        if int(m.sum()) < 4:
            continue
        ii = np.where(m)[0]
        lw = m & dance
        leg_std = [float(np.std(qw23[lw, j]) / max(np.std(refq[lw, j]), 1e-6))
                   for j in LEG_JOINTS] if int(lw.sum()) > 8 else [float("nan")]
        turns[nm] = {
            "root_path_m": float(np.linalg.norm(np.diff(ppos[ii, :2], axis=0), axis=1).sum()),
            "ref_path_m": float(np.linalg.norm(np.diff(refb[ii, 0, :2], axis=0), axis=1).sum()),
            "root_net_m": float(np.linalg.norm(ppos[ii[-1], :2] - ppos[ii[0], :2])),
            "root_err_end_m": float(root_err[ii[-1]]), "root_err_p50_m": pct(root_err[m], 50),
            "yaw_delta_deg": float(np.degrees(np.diff(np.unwrap(robot_yaw[ii])).sum())),
            "ref_yaw_delta_deg": float(np.degrees(np.diff(np.unwrap(ref_yaw_t[ii])).sum())),
            "yaw_err_end_deg": float(np.degrees(yaw_err[ii[-1]])),
            "leg_std_ratio_p50": float(np.median(leg_std)),
            "leg_err_p50": pct(jerr[lw][:, LEG_JOINTS].max(axis=1), 50) if int(lw.sum()) else float("nan"),
        }
    di = np.where(dance)[0]
    whole_root = {"net_vs_ref_m": float("nan"), "p50_vs_ref_m": float("nan"),
                  "net_m": float("nan"), "path_m": float("nan"), "ref_path_m": float("nan")}
    if len(di):
        whole_root = {"net_vs_ref_m": float(root_err[di[-1]]),
                      "p50_vs_ref_m": pct(root_err[dance], 50),
                      "net_m": float(np.linalg.norm(ppos[di[-1], :2] - ppos[di[0], :2])),
                      "path_m": float(np.linalg.norm(np.diff(ppos[di, :2], axis=0), axis=1).sum()),
                      "ref_path_m": float(np.linalg.norm(np.diff(refb[di, 0, :2], axis=0), axis=1).sum())}

    segments = {}
    for nm, a, b in lay["segments"]:
        m = (tdw >= a) & (tdw < b)
        seg = {
            "mpjpe_p50": pct(mpjpe[m & dance], 50), "mpjpe_p90": pct(mpjpe[m & dance], 90),
            "fsr_L": fsr["L"]["ratio"] if (contL & m & dance).sum() > 20 else float("nan"),
            "fsr_R": fsr["R"]["ratio"] if (contR & m & dance).sum() > 20 else float("nan"),
            "joint_err_p50": pct(jerr[m & dance].max(axis=1), 50) if (m & dance).any() else float("nan"),
            "success": bool((m & dance).any() and pct(mpjpe[m & dance], 50) < MPJPE_SUCCESS),
        }
        segments[nm] = seg

    # stored series (10 Hz) for the web charts
    sidx = np.arange(0, len(tdw), SERIES_SUB)
    sm = (tdw[sidx] > -2) & (tdw[sidx] < lay["series_end"])
    series = {
        "td": (tdw[sidx][sm]).tolist(),
        "mpjpe_world": (mpjpe[sidx][sm]).tolist(),
        "soleL": (soleL[sidx][sm]).tolist(), "soleR": (soleR[sidx][sm]).tolist(),
        "speedL": (vL[sidx][sm]).tolist(), "speedR": (vR[sidx][sm]).tolist(),
        "drift": (np.concatenate([np.zeros(iw), drel])[sidx][sm]).tolist(),
        "contL": (contL[sidx][sm]).astype(int).tolist(), "contR": (contR[sidx][sm]).astype(int).tolist(),
        "xyL": (xyL[sidx][sm]).tolist(), "xyR": (xyR[sidx][sm]).tolist(),
        "refxyL": (refb[sidx][sm, rig.foot_bodies["l"], :2]).tolist(),
        "refxyR": (refb[sidx][sm, rig.foot_bodies["r"], :2]).tolist(),
        "rootXY": (ppos[sidx][sm][:, :2]).tolist(),
        "refXY": (refb[sidx][sm, 0, :2]).tolist(),
        "rootErr": (root_err[sidx][sm]).tolist(),
        "yawErrDeg": (np.degrees(yaw_err[sidx][sm])).tolist(),
    }

    summary = {
        "mpjpe_world_p50": pct(mpjpe[dance], 50), "mpjpe_world_p90": pct(mpjpe[dance], 90),
        "fsr_L": fsr["L"]["ratio"], "fsr_R": fsr["R"]["ratio"],
        "fsr_ref_L": fsr["L"]["ratio_ref"], "fsr_ref_R": fsr["R"]["ratio_ref"],
        "slide_p50_L": fsr["L"]["slide_p50"], "slide_p50_R": fsr["R"]["slide_p50"],
        "slide_ref_p50_L": fsr["L"]["slide_ref_p50"], "slide_ref_p50_R": fsr["R"]["slide_ref_p50"],
        "penetration": penetration, "float": fl,
        "kick_apex_cm": apex, "kick_apex_ratio": apex_ratio,
        "ending_drift_m": drift, "ankle_margin_rad": ankle_margin,
        "whole_dance_root": whole_root, "turns": turns,
        "joint_err_p50_max": float(max(joints["err_p50"])),
        "dance_samples": int(dance.sum()),
    }
    return {"t_entry": float(t_entry), "summary": summary, "segments": segments,
            "joints": joints, "series": series}


def routing(m):
    """Map measured deficits onto the project's established levers."""
    out = []
    s = m["summary"]
    if s["fsr_ref_L"] > 0.10 or s["fsr_ref_R"] > 0.10:
        worst = "L" if s["fsr_ref_L"] > s["fsr_ref_R"] else "R"
        seg = max(m["segments"], key=lambda k: m["segments"][k].get("fsr_" + worst, 0) or 0)
        out.append({"severity": "high", "area": f"足滑移({worst})",
                    "evidence": f"参考支撑相滑移率 {s['fsr_ref_' + worst]*100:.0f}%（阈值 10%），悬浮占比 L {s['float']['L']['duration_frac']*100:.0f}%/R {s['float']['R']['duration_frac']*100:.0f}%，最差段 {seg}",
                    "suggestion": "支撑相滑移超标：候选为 R 系数据手术窗（参考接触一致性/ZMP）或确认软接触仿真成分（H0 真机仲裁）。注意 F3 已证伪'奖励罚滑移'路线。"})
    if s["kick_apex_ratio"]["L"] < 0.85:
        out.append({"severity": "high", "area": "左踢幅度",
                    "evidence": f"顶点比 {s['kick_apex_ratio']['L']*100:.0f}%（达标线 85%）",
                    "suggestion": "幅度亏损：自适应采样侧重（F7 已验证通道）；奖励侧仅剩宽 std 思路——F11/F12 已证腿部 std 锐化会折叠踝，慎用。"})
    if s["ending_drift_m"]["net"] > 1.0:
        out.append({"severity": "high", "area": "结尾漂移",
                    "evidence": f"净漂移 {s['ending_drift_m']['net']:.2f} m（阈值 1 m）",
                    "suggestion": "结尾失稳走位：F13 宽 std 臂项已验证可修（0.83 m），代价踢腿 86→73%。"})
    for tn, tv in (s.get("turns") or {}).items():
        if tv.get("root_err_end_m", 0) == tv.get("root_err_end_m", 0) and tv["root_err_end_m"] > 0.5:
            out.append({"severity": "high", "area": f"转身漂移({tn})",
                        "evidence": f"窗口末根误差 {tv['root_err_end_m']:.2f} m；机器人路径 {tv['root_path_m']:.2f} m vs 参考 {tv['ref_path_m']:.2f} m；终末 yaw 误差 {tv['yaw_err_end_deg']:.0f}°",
                        "suggestion": "转身段根失跟：观测无根位置（结构性开环），优先腿关节跟踪修复（F17）让脚步落地；仍超 0.5 m 再立观测 v2。"})
            break
    tls = [v.get("leg_std_ratio_p50") for v in (s.get("turns") or {}).values()]
    tls = [v for v in tls if v == v]
    if tls and min(tls) < 0.75:
        out.append({"severity": "medium", "area": "转身腿幅度",
                    "evidence": f"转身窗口腿 std_ratio p50 最低 {min(tls):.2f}（阈值 0.75）",
                    "suggestion": "转身段腿欠跟踪：F17 腿关节 joint-space 跟踪项的直接靶标。"})
    if s["float"]["L"]["duration_frac"] > 0.15 or s["float"]["R"]["duration_frac"] > 0.15:
        out.append({"severity": "medium", "area": "接触悬浮",
                    "evidence": f"悬浮时长占比 L {s['float']['L']['duration_frac']*100:.0f}% / R {s['float']['R']['duration_frac']*100:.0f}%",
                    "suggestion": "参考接触窗内鞋底悬空：R2/R4 系接触一致性手术候选；同时区分吊带期（入场后前 ~5 s）不计入。"})
    if s["ankle_margin_rad"] < 0.15:
        out.append({"severity": "medium", "area": "支撑踝余量",
                    "evidence": f"距机械限位 {s['ankle_margin_rad']:.3f} rad（阈值 0.15）",
                    "suggestion": "踝余量薄：R6 类参考踝钳位（已验证可买余量，注意深蹲段几何地板）。"})
    jmax = int(np.argmax(m["joints"]["err_p90"]))
    if m["joints"]["err_p90"][jmax] > 0.5:
        out.append({"severity": "low", "area": f"关节 {m['joints']['names'][jmax]}",
                    "evidence": f"误差 p90 {m['joints']['err_p90'][jmax]:.2f} rad",
                    "suggestion": "单关节跟踪误差大：检查该窗参考质量（GVHMR 伪影）优先于调奖励。"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--xml", default=os.path.expanduser("~/sim2sim_g1/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated version dirs to (re)evaluate")
    args = ap.parse_args()
    only = [v.strip() for v in args.only.split(",") if v.strip()]
    runs_root = os.path.join(args.data_root, "runs")
    refs = os.path.join(args.data_root, "refs")
    for ver in sorted(os.listdir(runs_root)):
        if only and ver not in only:
            continue
        vdir = os.path.join(runs_root, ver)
        meta_p = os.path.join(vdir, "meta.json")
        if not os.path.isdir(vdir) or not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p))
        rig = Rig(args.xml, os.path.join(refs, meta["ref_npz"]), os.path.join(refs, meta["ref_csv"]),
                  layout_from_meta(meta))
        for f in sorted(os.listdir(vdir)):
            if not f.endswith(".npz") or f.endswith("_sport.npz") or f.endswith(".fk.npz"):
                continue
            tag = f[:-4]
            out_p = os.path.join(vdir, tag + ".metrics.json")
            if os.path.exists(out_p) and not args.force:
                continue
            te_file = os.path.join(vdir, tag + ".t_entry")
            te = float(open(te_file).read()) if os.path.exists(te_file) else None
            t0 = time.time()
            m = evaluate(os.path.join(vdir, f), rig, te)
            m.update({"version": ver, "tag": tag, "ref_chain": meta["ref_npz"],
                      "t_entry_source": "file" if te is not None else "inferred",
                      "date": meta.get("date", ""), "latches": meta.get("latches", 0),
                      "in_service": meta.get("in_service", False)})
            m["routing"] = routing(m)
            json.dump(m, open(out_p, "w"), ensure_ascii=False, indent=1)
            s = m["summary"]
            te_max = max([v.get("root_err_end_m", float("nan")) for v in (s.get("turns") or {}).values()],
                         default=float("nan"))
            te_txt = f" turn_err_end {te_max:.2f}m" if te_max == te_max else ""
            print(f"{ver}/{tag}: mpjpe {s['mpjpe_world_p50']:.3f}m FSR L/R {s['fsr_L']*100:.0f}/{s['fsr_R']*100:.0f}% "
                  f"apex {s['kick_apex_ratio']['L']*100:.0f}% drift {s['ending_drift_m']['net']:.2f}m "
                  f"ankle {s['ankle_margin_rad']:.3f}{te_txt} t_entry {m['t_entry_source']} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
