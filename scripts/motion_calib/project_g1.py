#!/usr/bin/env python
"""project_g1.py - automatic G1-domain reference projection (MC S3/S4).

Generalizes the hand-written take6/take2 R-series surgery into one automatic
solver that runs on any 30-column G1 reference CSV (post-GMR, pre-transitions):

  1. auto-grounding: per-contact-segment root-z re-anchoring (kills the
     world-root vertical wander that a single constant offset cannot);
  2. contact pinning: during stance segments the foot is pinned to its
     segment-start XY, lifted out of penetration, roll/pitch flattened with
     yaw kept (r2_contact23 generalized - no hand windows, no exemptions);
  3. ZMP root re-solve: whole-body ZMP (measure.py math) is pushed inside the
     contact support polygon by a smoothed, clamped root-XY shift with
     foot-hold-preserving leg re-IK (r3_root23 generalized, two passes).

Everything is driven by the measured state; there are no per-motion
constants. Fidelity is guarded and reported: untouched outside edits,
body-position deviation vs the input, joint limits, frame continuity.

Usage:
  python project_g1.py IN.csv OUT.csv [--xml ...] [--config ...]
      [--margin 0.008] [--dmax 0.10] [--report OUT.json]
"""

import argparse
import json
import sys

import numpy as np
import mujoco as mj

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from measure import default_g1_xml  # noqa: E402
from measure import (  # noqa: E402
    CONTACT_M,
    Rigger,
    measure_fk,
    poly_margin,
    rotvec,
    segments,
    support_poly,
    zmp_compute,
)

FPS = 30.0
DT = 1.0 / FPS
LEG = {"left": list(range(0, 6)), "right": list(range(6, 12))}
GN = ["l_hip_p", "l_hip_r", "l_hip_y", "l_knee", "l_ank_p", "l_ank_r",
      "r_hip_p", "r_hip_r", "r_hip_y", "r_knee", "r_ank_p", "r_ank_r"]


def blend_rot(R0, R1, w):
    rv = rotvec(R1 @ R0.T)
    th = np.linalg.norm(rv)
    if th < 1e-9 or w >= 1.0:
        return R1 if w >= 1.0 else R0
    ax = rv / th
    a2 = w * th
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a2) * K + (1 - np.cos(a2)) * K @ K) @ R0


def smooth_series(x, k=9):
    ker = np.hanning(2 * k + 1)
    ker /= ker.sum()
    return np.convolve(x, ker, "same")


def solve_legs(rig, m, f, sides, foot_pos, foot_rot, q_init, q_ref, lo, up,
               iters=80, step=0.2, mu=0.005):
    """DLS IK of the given legs toward 6-D foot pose targets (r2/r3 pattern)."""
    q23 = q_init.copy()
    legs = []
    for s in sides:
        legs += LEG[s]
    for _ in range(iters):
        rig.fk(m[f, :7], q23)
        E, rows_J = [], []
        for s in sides:
            bi = rig.body[s]
            pe = foot_pos[s][f] - rig.data.xpos[bi]
            Re = rotvec(foot_rot[s][f] @ rig.data.xmat[bi].reshape(3, 3).T)
            jacp = np.zeros((3, rig.model.nv))
            jacr = np.zeros((3, rig.model.nv))
            mj.mj_jacBody(rig.model, rig.data, jacp, jacr, bi)
            dofs = np.array([6 + j for j in LEG[s]])
            E.append(np.concatenate([pe, Re]))
            Jb = np.zeros((6, len(legs)))
            Jb[:, [legs.index(j) for j in LEG[s]]] = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            rows_J.append(Jb)
        e = np.concatenate(E)
        J = np.vstack(rows_J)
        if np.abs(e).max() < 8e-4:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + 3e-4 * np.eye(len(e)), e)
        dq = np.clip(dq, -step, step)
        q23[legs] += 0.6 * dq + mu * (q_ref[legs] - q23[legs])
        q23[legs] = np.clip(q23[legs], lo[legs], up[legs])
    return q23


def foot_pose_targets(rig, m, clr, fpos, frot, cm):
    """Pinning targets: segment-start XY, grounded Z, yaw-keep flatten."""
    targets = {}
    for s in ("left", "right"):
        segs = segments(cm[s])
        tgt_p = fpos[s].copy()
        tgt_R = frot[s].copy()
        for a, b in segs:
            Rm = frot[s][a:min(b + 1, a + 6)].mean(axis=0)
            U, _, Vt = np.linalg.svd(Rm)
            Rstand = U @ Vt
            pin_xy = fpos[s][a, :2].copy()
            for f in range(a, b + 1):
                z_ground = fpos[s][f, 2] - min(clr[s][f], 0.0)
                w = min(1.0, (f - a + 1) / 4.0, (b - f + 1) / 4.0)
                p = fpos[s][f].copy()
                p[:2] = (1 - w) * p[:2] + w * pin_xy
                p[2] = (1 - w) * p[2] + w * z_ground
                dR = frot[s][f] @ Rstand.T
                yaw = np.arctan2(dR[1, 0], dR[0, 0])
                c, sn = np.cos(yaw), np.sin(yaw)
                flat = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]) @ Rstand
                tgt_p[f] = p
                tgt_R[f] = blend_rot(frot[s][f], flat, w)
        targets[s] = (tgt_p, tgt_R)
    return targets


def auto_ground(rig, m, clr, cm):
    """Root-z anchoring: global origin-scaling offset first (ground_motion.py
    logic: p50 of the min-sole series), then per-contact-segment refinement
    against the world-root vertical wander."""
    n = m.shape[0]
    delta0 = float(np.percentile(np.minimum(clr["left"], clr["right"]), 50))
    m = m.copy()
    m[:, 2] -= delta0
    bp, bR, clr2, _, _ = measure_fk(rig, m)
    cm2 = {s: clr2[s] < CONTACT_M for s in ("left", "right")}
    offs = []
    for s in ("left", "right"):
        for a, b in segments(cm2[s]):
            offs.append((a, b, float(np.median(clr2[s][a:b + 1]))))
    anchor = np.zeros(n)
    if offs:
        offs.sort()
        for a, b, c in offs:
            anchor[a:b + 1] = c
        idx = np.arange(n)
        known = np.where(np.abs(anchor) > 0)[0]
        if known.size:
            anchor = np.interp(idx, known, anchor[known])
        delta = smooth_series(anchor, 15)
        m[:, 2] -= delta
    return m, delta0


def zmp_shift_pass(rig, m, cm, margin, dmax, lo, up, gain=1.0, base_xy=None, budget=0.12):
    """One r3-style pass: ZMP deficit -> smoothed clamped root XY shift,
    contact feet re-IKed to hold world pose (rollback halving). Cumulative
    root displacement vs base_xy is capped at `budget` per frame (fidelity)."""
    n = m.shape[0]
    bp, bR, clr, fpos, frot = measure_fk(rig, m)
    zmp, _ = zmp_compute(rig, bp, bR)
    D = np.zeros((n, 2))
    for f in range(n):
        poly = support_poly(cm, bp, bR, rig, f)
        if poly is None:
            continue
        mg = poly_margin(poly, zmp[f])
        if mg < margin:
            # target: margin inside along the closest edge
            ctr = poly.mean(axis=0)
            dmins, edges = [], []
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                e = b - a
                cr = e[0] * (zmp[f][1] - a[1]) - e[1] * (zmp[f][0] - a[0])
                dmins.append(abs(cr) / np.linalg.norm(e))
                edges.append((a + b) / 2 + np.array([e[1], -e[0]]) / np.linalg.norm(e) * margin
                             * (1 if np.array([e[1], -e[0]]) @ (ctr - (a + b) / 2) > 0 else -1))
            i_min = int(np.argmin(dmins))
            step_v = edges[i_min] - zmp[f]
            nv = np.linalg.norm(step_v)
            if nv > 0.12:  # per-pass displacement cap; big deficits converge over passes
                step_v *= 0.12 / nv
            D[f] = gain * step_v
    for c in range(2):
        D[:, c] = smooth_series(D[:, c], 3)
    nrm = np.linalg.norm(D, axis=1, keepdims=True)
    D *= np.minimum(1.0, dmax / np.maximum(nrm, 1e-9))
    if base_xy is not None:
        used = np.linalg.norm(m[:, :2] - base_xy, axis=1)
        allow = np.maximum(budget - used, 0.0)
        dn = np.linalg.norm(D, axis=1)
        scale = np.ones(n)
        hit = (dn > 1e-9) & (allow < dn)
        scale[hit] = allow[hit] / dn[hit]
        scale = smooth_series(scale, 5)
        scale = np.minimum(scale, 1.0)
        D *= scale[:, None]
    for f in range(n):
        if f < 15:
            D[f] *= f / 15.0
        if f >= n - 15:
            D[f] *= (n - f) / 15.0
    out = m.copy()
    rolled = 0
    prev_q = None
    for f in range(n):
        if np.linalg.norm(D[f]) < 1e-6:
            prev_q = None
            continue
        sides = [s for s in ("left", "right") if cm[s][f]]
        out[f, :2] += D[f]
        if not sides:
            continue
        q_ref = m[f, 7:].copy()
        if prev_q is not None:
            legs = []
            for s in sides:
                legs += LEG[s]
            q_ref[legs] = prev_q[legs]
        best = None
        scale = 1.0
        for _try in range(3):
            out[f, :2] = m[f, :2] + D[f] * scale
            q23 = solve_legs(rig, out, f, sides, fpos, frot, q_ref, q_ref, lo, up)
            rig.fk(out[f, :7], q23)
            herr = max(np.linalg.norm(rig.data.xpos[rig.body[s]] - fpos[s][f]) for s in sides)
            if best is None or herr < best[0]:
                best = (herr, q23, scale)
            if herr <= 0.003:
                break
            scale *= 0.5
        herr, q23, scale = best
        if herr > 0.003:
            rolled += 1
        out[f, :2] = m[f, :2] + D[f] * scale
        out[f, 7:] = q23
        D[f] = D[f] * scale
        prev_q = q23 if herr <= 0.009 else None
    return out, rolled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--xml", default=default_g1_xml())
    ap.add_argument("--config", default=None)
    ap.add_argument("--margin", type=float, default=0.008)
    ap.add_argument("--dmax", type=float, default=0.10)
    ap.add_argument("--budget", type=float, default=0.12, help="cumulative per-frame root XY shift budget (m)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    lo = up = None
    if args.config:
        cfg = json.load(open(args.config))
        lo = np.array(cfg["joint_lower"])
        up = np.array(cfg["joint_upper"])
    else:
        sys.exit("--config is required for joint limits")
    rig = Rigger(args.xml)

    m0 = np.loadtxt(args.src, delimiter=",")
    n = m0.shape[0]
    bp0, _, _, _, _ = measure_fk(rig, m0)

    # stage 1: auto grounding (global offset then per-segment wander)
    bp, bR, clr, fpos, frot = measure_fk(rig, m0)
    cm = {s: clr[s] < CONTACT_M for s in ("left", "right")}
    m, gmax = auto_ground(rig, m0, clr, cm)
    print("[ground] global offset %.3f m (+per-segment refinement)" % gmax)

    # stage 2: contact pinning (r2 generalized)
    bp, bR, clr, fpos, frot = measure_fk(rig, m)
    cm = {s: clr[s] < CONTACT_M for s in ("left", "right")}
    targets = foot_pose_targets(rig, m, clr, fpos, frot, cm)
    m = m.copy()
    n_pins = 0
    for f in range(n):
        sides = [s for s in ("left", "right") if cm[s][f]]
        if not sides:
            continue
        q23 = solve_legs(rig, m, f, sides,
                         {s: targets[s][0] for s in sides},
                         {s: targets[s][1] for s in sides},
                         m[f, 7:], m[f, 7:], lo, up)
        m[f, 7:] = q23
        n_pins += 1
    print("[pin] frames edited %d / %d" % (n_pins, n))

    # re-measure contact after pinning for the ZMP passes
    bp, bR, clr, fpos2, frot2 = measure_fk(rig, m)
    cm = {s: clr[s] < CONTACT_M for s in ("left", "right")}

    # stage 3: ZMP passes (r3 generalized) - iterate to saturation under a
    # cumulative root-shift budget (choreography fidelity cap)
    base_xy = m[:, :2].copy()
    rolled_total = []
    gains = [0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]
    prev_outside = None
    for ip, g in enumerate(gains):
        m, rr = zmp_shift_pass(rig, m, cm, args.margin, min(args.dmax, 0.06), lo, up, gain=g, base_xy=base_xy, budget=args.budget)
        rolled_total.append(rr)
        # progress metric: single+double support outside fraction
        bp, bR = measure_fk(rig, m)[:2]
        clr3 = measure_fk(rig, m)[2]
        cm3 = {s: clr3[s] < CONTACT_M for s in ("left", "right")}
        zmp3, _ = zmp_compute(rig, bp, bR)
        tot = bad = 0
        for f in range(n):
            poly = support_poly(cm3, bp, bR, rig, f)
            if poly is None:
                continue
            tot += 1
            if poly_margin(poly, zmp3[f]) < 0:
                bad += 1
        outside = 100.0 * bad / max(tot, 1)
        print("[zmp] pass %d (gain %.1f): rolled %d, outside %.1f%%" % (ip + 1, g, rr, outside))
        if prev_outside is not None and prev_outside - outside < 0.5:
            break
        prev_outside = outside

    # stage 4: de-kink - the projection must not add continuity spikes the
    # raw input did not have (blend offending steps over 5 frames)
    dq_cap = max(float(np.abs(np.diff(m0[:, 7:], axis=0)).max()), 0.35)
    for _ in range(3):
        steps = np.abs(np.diff(m[:, 7:], axis=0)).max(axis=1)
        bad = np.where(steps > dq_cap)[0]
        if len(bad) == 0:
            break
        for f in bad:
            a, b = max(0, f - 2), min(n, f + 3)
            seg = m[a:b, 7:]
            w = np.linspace(0, 1, len(seg))[:, None]
            ref = seg[0][None, :] * (1 - w) + seg[-1][None, :] * w
            m[a:b, 7:] = 0.5 * seg + 0.5 * ref
    print("[dekink] cap %.3f rad/frame" % dq_cap)

    # fidelity report
    bp2, _, _, _, _ = measure_fk(rig, m)
    dev = np.linalg.norm(bp2 - bp0, axis=2)
    dq = np.abs(m[:, 7:] - m0[:, 7:])
    viol = int(((m[:, 7:] < lo - 1e-9) | (m[:, 7:] > up + 1e-9)).sum())
    rep = {
        "src": args.src, "dst": args.dst, "frames": n,
        "ground_max_shift_m": gmax, "pin_frames": n_pins,
        "zmp_rolled": rolled_total,
        "body_dev_m": {"p50": float(np.percentile(dev, 50)), "p95": float(np.percentile(dev, 95)),
                       "max": float(dev.max())},
        "joint_dq_rad": {"p95": float(np.percentile(dq, 95)), "max": float(dq.max())},
        "frame_dq_max_rad": float(np.abs(np.diff(m[:, 7:], axis=0)).max()),
        "limit_violation_cells": viol,
        "root_shift_m": {"xy_p95": float(np.percentile(np.linalg.norm(m[:, :2] - m0[:, :2], axis=1), 95)),
                         "z_p95": float(np.percentile(np.abs(m[:, 2] - m0[:, 2]), 95))},
    }
    print("[fidelity] body dev p50 %.3f p95 %.3f max %.3f m | joint dq p95 %.3f max %.3f | "
          "root xy p95 %.3f | limit cells %d"
          % (rep["body_dev_m"]["p50"], rep["body_dev_m"]["p95"], rep["body_dev_m"]["max"],
             rep["joint_dq_rad"]["p95"], rep["joint_dq_rad"]["max"],
             rep["root_shift_m"]["xy_p95"], viol))
    np.savetxt(args.dst, m, delimiter=",", fmt="%.10g")
    print("wrote", args.dst)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
