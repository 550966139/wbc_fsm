#!/usr/bin/env python
"""measure.py - G1 23DoF reference-motion feasibility ruler (MC campaign S1).

Consolidates the measurement definitions that were scattered across the
take6/take2 R-series surgery scripts (ground_motion.py / r2_contact23.py /
r3_root23.py on the 224 pipelines) into one general tool that runs on any
30-column reference CSV:

  columns [0:3] root xyz (world), [3:7] root quat (x y z w), [7:30] 23 joints
  30 fps, world frame.

Per foot: contact fraction / segments, sole-clearance percentiles,
planted-foot skate, worst per-segment drift. Whole body: ZMP vs support
polygon margins per support class (L / R / LR), root drift, joint limits,
frame-to-frame joint continuity.

The math replicates the R-series exactly (same MuJoCo FK via g1_23dof.xml,
1.5 cm sole-clearance contact threshold, Hanning(2k+1, k=9)-smoothed
derivatives in the ZMP estimator, sole rectangle around the ankle_roll
body), so the historical ledger numbers reproduce on the historical files:
take6 raw GMR float p50 ~ +0.050 m, take6 trans_r2 single-support ZMP
below-margin ~95%/89%.

Usage:
  python measure.py CSV [CSV ...] [--xml g1_23dof.xml]
      [--config g1_23dof_tracking.json] [--window A:B] [--label NAME]
      [--json OUT.json]

--window A:B restricts ZMP/summary statistics to frames [A, B) (dance window
without the lead-in/out transitions), matching how r3_root23.py scored.
"""

import argparse
import json
import os
import sys

import numpy as np
import mujoco as mj


def default_g1_xml(scene=False):
    """Locate the g1_23dof robot/scene XML without hardcoding one machine.

    Checks $MC_G1_XML_DIR first, then the known rigs (wsk sim2sim replica,
    224 training checkout). Exits with the candidate list when not found.
    """
    name = "scene_g1_23dof.xml" if scene else "g1_23dof.xml"
    home = os.path.expanduser("~")
    cands = []
    env = os.environ.get("MC_G1_XML_DIR")
    if env:
        cands.append(os.path.join(env, name))
    cands += [
        f"{home}/sim2sim_g1/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/{name}",
        f"{home}/dsh-workspace/unitree-rl/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/{name}",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    sys.exit("g1 xml not found; set MC_G1_XML_DIR or pass --xml. Tried:\n  " + "\n  ".join(cands))

FPS = 30.0
DT = 1.0 / FPS
G = 9.81
CONTACT_M = 0.015  # sole clearance below which a foot is planted (R-series)
ZMP_MARGIN_M = 0.008  # R3 target margin; strict outside line (0) also reported
SMOOTH_K = 9  # Hanning half-width of the r3 zmp derivative smoothing
# G1 sole rectangle (x forward, y left) around the ankle_roll body origin.
# Robot spec carried over from r3_root23.py support_poly().
SOLE_RECT = [(-0.05, 0.04), (-0.05, -0.04), (0.10, 0.04), (0.10, -0.04)]


def rotvec(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(tr)
    if ang < 1e-9:
        return np.zeros(3)
    return ang * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))


class Rigger:
    """MuJoCo FK wrapper mirroring the R-series scripts' model usage."""

    def __init__(self, xml):
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)
        m = self.model
        self.body = {}
        for bi in range(m.nbody):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bi) or ""
            if "ankle_roll" in nm:
                side = "left" if nm.startswith("left") else "right"
                self.body[side] = bi
        if set(self.body) != {"left", "right"}:
            sys.exit("could not locate both ankle_roll bodies in %s: %r" % (xml, self.body))
        self.foot_geoms = {"left": [], "right": []}
        for gi in range(m.ngeom):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gi) or ""
            for side in ("left", "right"):
                if nm.startswith("%s_foot" % side) and "collision" in nm:
                    self.foot_geoms[side].append(
                        (gi, float(m.geom_size[gi][1]), float(m.geom_size[gi][0]))
                    )
        self.mass = np.array([m.body_mass[i] for i in range(1, m.nbody)])
        self.inertia = np.array([m.body_inertia[i] for i in range(1, m.nbody)])
        self.nb = m.nbody - 1

    def fk(self, root, q23):
        d = self.data
        d.qpos[:3] = root[:3]
        q = root[3:7]
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[7:] = q23
        mj.mj_forward(self.model, d)

    def sole_clearance(self, side):
        best = np.inf
        for gi, hl, rad in self.foot_geoms[side]:
            R = self.data.geom_xmat[gi].reshape(3, 3)
            c = self.data.geom_xpos[gi]
            best = min(best, (c + R[:, 2] * hl)[2] - rad, (c - R[:, 2] * hl)[2] - rad)
        return best


def measure_fk(rig, m):
    """Per-frame body poses and foot states for the whole CSV."""
    n = m.shape[0]
    bp = np.zeros((n, rig.nb, 3))
    bR = np.zeros((n, rig.nb, 3, 3))
    clr = {s: np.zeros(n) for s in ("left", "right")}
    fpos = {s: np.zeros((n, 3)) for s in ("left", "right")}
    frot = {s: np.zeros((n, 3, 3)) for s in ("left", "right")}
    for f in range(n):
        rig.fk(m[f, :7], m[f, 7:])
        for i in range(rig.nb):
            bp[f, i] = rig.data.xpos[i + 1]
            bR[f, i] = rig.data.xmat[i + 1].reshape(3, 3)
        for s in ("left", "right"):
            bi = rig.body[s]
            fpos[s][f] = rig.data.xpos[bi]
            frot[s][f] = rig.data.xmat[bi].reshape(3, 3)
            clr[s][f] = rig.sole_clearance(s)
    return bp, bR, clr, fpos, frot


def zmp_compute(rig, bp, bR):
    """Full-body ZMP, ported verbatim from r3_root23.py (incl. smoothing)."""
    n = bp.shape[0]
    ker = np.hanning(2 * SMOOTH_K + 1)
    ker /= ker.sum()

    def smooth(a):
        # convolve each body/component time series, as r3_root23.py did
        return np.stack(
            [np.apply_along_axis(lambda x: np.convolve(x, ker, "same"), 0, a[:, :, c]) for c in range(3)],
            axis=-1,
        )

    vel = smooth(np.gradient(bp, DT, axis=0, edge_order=2))
    ome = np.zeros((n, rig.nb, 3))
    for f in range(n - 1):
        for i in range(rig.nb):
            ome[f, i] = rotvec(bR[f + 1, i] @ bR[f, i].T) / DT
    ome[-1] = ome[-2]
    ome = smooth(ome)

    m_tot = rig.mass.sum()
    r_com = (bp * rig.mass[None, :, None]).sum(axis=1) / m_tot
    v_com = (vel * rig.mass[None, :, None]).sum(axis=1) / m_tot
    L = np.zeros((n, 3))
    for i in range(rig.nb):
        r = bp[:, i] - r_com
        v = vel[:, i] - v_com
        L += np.cross(r, rig.mass[i] * v)
        for f in range(n):
            R = bR[f, i]
            L[f] += R @ (rig.inertia[i] * (R.T @ ome[f, i]))
    Ld = np.gradient(L, DT, axis=0, edge_order=2)
    a_com = np.gradient(v_com, DT, axis=0, edge_order=2)
    F = m_tot * (a_com + np.array([0, 0, G]))
    zmp = np.zeros((n, 2))
    zmp[:, 0] = r_com[:, 0] - (Ld[:, 1] + r_com[:, 2] * F[:, 0]) / F[:, 2]
    zmp[:, 1] = r_com[:, 1] + (Ld[:, 0] - r_com[:, 2] * F[:, 1]) / F[:, 2]
    return zmp, r_com


def support_poly(cm, bp, bR, rig, f):
    """Convex hull of contact sole rectangles, as in r3_root23.py."""
    pts = []
    for s in ("left", "right"):
        if not cm[s][f]:
            continue
        bi = rig.body[s]
        c = bp[f, bi - 1, :2]
        R = bR[f, bi - 1]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cs, sn = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cs, -sn], [sn, cs]])
        for dx, dy in SOLE_RECT:
            pts.append(c + Rz @ np.array([dx, dy]))
    if not pts:
        return None
    pts = np.array(pts)
    ctr = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
    return pts[np.argsort(ang)]


def poly_margin(poly, p):
    """Signed distance from p to the convex polygon (positive inside)."""
    n = len(poly)
    ctr = poly.mean(axis=0)
    inside = True
    dmin = 1e9
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        cr = e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])
        if cr < 0:
            inside = False
        dmin = min(dmin, abs(cr) / np.linalg.norm(e))
    return dmin if inside else -dmin


def segments(mask, min_len=3):
    segs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


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


def measure_csv(path, rig, lo=None, up=None, window=None, label=None):
    m = np.loadtxt(path, delimiter=",")
    if m.ndim != 2 or m.shape[1] != 30:
        sys.exit("%s: expected 30-column CSV, got shape %r" % (path, m.shape))
    n = m.shape[0]
    bp, bR, clr, fpos, frot = measure_fk(rig, m)
    cm = {s: clr[s] < CONTACT_M for s in ("left", "right")}
    zmp, r_com = zmp_compute(rig, bp, bR)

    f0, f1 = (0, n) if window is None else (max(0, window[0]), min(n, window[1]))

    rep = {"file": path, "label": label or path.split("/")[-1], "frames": n, "window": [f0, f1]}

    # --- feet: contact / clearance / skate ---
    feet = {}
    for s in ("left", "right"):
        segs = segments(cm[s])
        contact_frames = np.where(cm[s])[0]
        # planted-foot skate: XY speed of frames whose *later* frame is contact
        v = np.linalg.norm(np.diff(fpos[s][:, :2], axis=0), axis=1) * FPS
        vc = v[cm[s][1:]] if cm[s].any() else v[:0]
        # contact-frame sole clearance
        cc = clr[s][contact_frames] if len(contact_frames) else clr[s][:0]
        worst_drift = 0.0
        for a, b in segs:
            worst_drift = max(worst_drift, np.linalg.norm(fpos[s][b, :2] - fpos[s][a, :2]))
        feet[s] = {
            "contact_fraction": float(cm[s][f0:f1].mean()),
            "contact_frames_total": int(cm[s].sum()),
            "segments": len(segs),
            "sole_clearance_contact": stats_pct(cc),
            "skate_contact": stats_pct(vc),
            "worst_segment_drift_m": float(worst_drift),
        }
    rep["feet"] = feet
    # ground_motion-style whole-motion min-sole offset (both feet, all frames)
    rep["min_sole_offset_p50_m"] = float(np.percentile(np.minimum(clr["left"], clr["right"]), 50))

    # --- ZMP vs support polygon ---
    zstats = {"L": [], "R": [], "LR": [], "none": 0}
    outside = {"L": 0, "R": 0, "LR": 0}
    for f in range(f0, f1):
        poly = support_poly(cm, bp, bR, rig, f)
        if poly is None:
            zstats["none"] += 1
            continue
        mg = poly_margin(poly, zmp[f])
        key = ("L" if cm["left"][f] else "") + ("R" if cm["right"][f] else "")
        zstats[key].append(mg)
        if mg < 0:
            outside[key] += 1
    zmp_rep = {}
    for key in ("L", "R", "LR"):
        v = np.array(zstats[key]) if zstats[key] else np.array([])
        if len(v) == 0:
            continue
        zmp_rep[key] = {
            "n": int(len(v)),
            "margin_cm_p50": float(np.median(v)) * 100,
            "margin_cm_p10": float(np.percentile(v, 10)) * 100,
            "margin_cm_min": float(v.min()) * 100,
            "below_8mm_pct": float((v < ZMP_MARGIN_M).mean() * 100),
            "outside_pct": float((v < 0).mean() * 100),
        }
    rep["zmp"] = zmp_rep
    rep["zmp"]["no_support_frames"] = zstats["none"]

    # --- root / joints ---
    rep["root"] = {
        "z_p50_m": float(np.percentile(m[:, 2], 50)),
        "end_drift_m": float(np.linalg.norm(m[-1, :2] - m[0, :2])),
        "max_excursion_m": float(np.linalg.norm(m[:, :2] - m[0, :2], axis=1).max()),
    }
    dq = np.abs(np.diff(m[:, 7:], axis=0))
    rep["joints"] = {
        "frame_dq_max_rad": float(dq.max()),
        "frame_dq_p999_rad": float(np.percentile(dq, 99.9)),
    }
    if lo is not None:
        viol = int(((m[:, 7:] < lo - 1e-9) | (m[:, 7:] > up + 1e-9)).sum())
        rep["joints"]["limit_violation_cells"] = viol
    return rep, m, bp, bR, cm, zmp


def print_report(rep):
    print("=" * 78)
    print("%s  (%d frames, window %s)" % (rep["label"], rep["frames"], rep["window"]))
    print("  min-sole offset p50 (ground_motion convention): %+.4f m" % rep["min_sole_offset_p50_m"])
    for s in ("left", "right"):
        f = rep["feet"][s]
        sc, sk = f["sole_clearance_contact"], f["skate_contact"]
        print(
            "  %-5s contact %5.1f%% (%d segs) | clearance p50 %+0.4f min %+0.4f | "
            "skate p50 %.3f p90 %.3f p95 %.3f | worst seg drift %.3f m"
            % (
                s,
                f["contact_fraction"] * 100,
                f["segments"],
                sc["p50"] if sc else float("nan"),
                sc["min"] if sc else float("nan"),
                sk["p50"] if sk else float("nan"),
                sk["p90"] if sk else float("nan"),
                sk["p95"] if sk else float("nan"),
                f["worst_segment_drift_m"],
            )
        )
    for key in ("L", "R", "LR"):
        z = rep["zmp"].get(key)
        if not z:
            continue
        print(
            "  ZMP %-2s n=%4d: margin p50 %+6.1fcm p10 %+6.1fcm min %+6.1fcm | "
            "below-8mm %3d%% | OUTSIDE %3d%%"
            % (
                key,
                z["n"],
                z["margin_cm_p50"],
                z["margin_cm_p10"],
                z["margin_cm_min"],
                round(z["below_8mm_pct"]),
                round(z["outside_pct"]),
            )
        )
    print(
        "  root: z p50 %.3f end-drift %.2f max-excursion %.2f | dq max %.3f rad | limit cells %s"
        % (
            rep["root"]["z_p50_m"],
            rep["root"]["end_drift_m"],
            rep["root"]["max_excursion_m"],
            rep["joints"]["frame_dq_max_rad"],
            rep["joints"].get("limit_violation_cells", "n/a"),
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--xml", default=default_g1_xml())
    ap.add_argument("--config", default=None, help="tracking config JSON for joint limits")
    ap.add_argument("--window", default=None, help="stats window A:B (frames, [A,B))")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    lo = up = None
    if args.config:
        cfg = json.load(open(args.config))
        lo = np.array(cfg["joint_lower"])
        up = np.array(cfg["joint_upper"])
    rig = Rigger(args.xml)
    window = None
    if args.window:
        a, b = args.window.split(":")
        window = (int(a), int(b))

    reps = []
    for path in args.csv:
        rep = measure_csv(path, rig, lo, up, window, args.label)[0]
        reps.append(rep)
        print_report(rep)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(reps, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
