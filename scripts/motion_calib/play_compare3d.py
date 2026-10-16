#!/usr/bin/env python
"""play_compare3d.py - dual native MuJoCo 3D playback: raw vs MC-calibrated.

Opens two mujoco.viewer windows side by side (left-ish: raw GMR reference,
right-ish: MC-calibrated), stepping the SAME frame index of both 30-column
CSVs at 30 fps in a loop. Uses MuJoCo's own renderer (orbit: left-drag,
pan: right-drag, zoom: scroll) - no custom rendering code.

Usage:
  python play_compare3d.py [--old RAW.csv] [--new MC.csv] [--speed 1.0]
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import default_g1_xml


def load(m):
    # CSV [xyz, quat xyzw, joints] -> qpos [xyz, quat wxyz, joints]
    q = m.copy()
    q[:, 3] = m[:, 6]
    q[:, 4:7] = m[:, 3:6]
    return q


def set_frame(data, q):
    data.qpos[: len(q)] = q  # scene model carries 6 extra dummy joints
    mujoco.mj_forward(data.model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=default_g1_xml(scene=True))
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--label-old", default="RAW GMR")
    ap.add_argument("--label-new", default="MC calibrated")
    ap.add_argument("--smooth", type=int, default=3, help="display-only joint smoothing (hanning half-width, 0=off)")
    args = ap.parse_args()

    qo = load(np.loadtxt(args.old, delimiter=","))
    qn = load(np.loadtxt(args.new, delimiter=","))
    if args.smooth > 0:
        def smooth_disp(q):
            ker = np.hanning(2 * args.smooth + 1)
            ker /= ker.sum()
            out = q.copy()
            for c in range(q.shape[1]):
                out[:, c] = np.convolve(q[:, c], ker, "same")
            return out
        print("display smoothing k=%d (artifact files unchanged)" % args.smooth)
        qo, qn = smooth_disp(qo), smooth_disp(qn)
    n = min(len(qo), len(qn))
    print("frames:", n, "@30fps =", round(n / 30.0, 1), "s | speed", args.speed)
    print("first-opened window = RAW GMR, second = MC calibrated")

    mo = mujoco.MjModel.from_xml_path(args.xml)
    mn = mujoco.MjModel.from_xml_path(args.xml)
    do, dn = mujoco.MjData(mo), mujoco.MjData(mn)

    with mujoco.viewer.launch_passive(mo, do) as vo, \
         mujoco.viewer.launch_passive(mn, dn) as vn:
        vo.cam.lookat[:] = [0.0, 0.0, 0.9]
        vn.cam.lookat[:] = [0.0, 0.0, 0.9]
        for v in (vo, vn):
            v.cam.distance = 3.2
            v.cam.azimuth = 90.0
            v.cam.elevation = -10.0
        # wall-clock-driven frame index: render rate decoupled from motion rate
        t0 = time.time()
        k_last = -1
        while vo.is_running() and vn.is_running():
            k = int((time.time() - t0) * 30.0 * args.speed) % n
            if k != k_last:
                set_frame(do, qo[k])
                set_frame(dn, qn[k])
                k_last = k
                if k % 150 == 0:
                    print("t = %.1f s" % (k / 30.0), flush=True)
            vo.sync()
            vn.sync()
            time.sleep(0.004)
    print("viewer closed")


if __name__ == "__main__":
    main()
