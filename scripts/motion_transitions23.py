#!/usr/bin/env python3
"""Prepend/append stand-transition segments to a 23DoF tracking motion CSV.

Tracking policies are trained with reference-state initialization (episodes
start AT the reference pose), so they never learn a stand -> first-frame
transition. At deployment the robot enters from the fixed stand, so the motion
file itself must start (and ideally end) at the default stand pose:

  * lead-in:  N frames lerped from default stand (joints/root pos) to frame 0,
              root quat held at frame-0 value (no yaw spin);
  * lead-out: N frames lerped from the last frame back to default stand,
              root quat held at the last frame's value.

Frame 0 of the output equals default_joint_pos, so the deployment entry gate
(robot at fixed stand vs motion frame 0, 0.15 rad) passes naturally, and the
policy tracks the gentle transition as ordinary reference motion.

CSV format: 30 columns (root_xyz, root_quat_xyzw, 23 joints), 30 fps.

Usage:
    motion_transitions23.py --in take6_23dof.csv --out take6_23dof_trans.csv \
        [--config config/g1_23dof_tracking.json] [--seconds 1.0]
"""

import argparse
import json
from pathlib import Path

import numpy as np

FPS = 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config/g1_23dof_tracking.json")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--stand-root-z", type=float, default=0.78)
    args = ap.parse_args()

    defaults = np.array(json.loads(Path(args.config).read_text())["default_joint_pos"])
    assert defaults.shape == (23,)

    m = np.loadtxt(args.inp, delimiter=",")
    assert m.ndim == 2 and m.shape[1] == 30 and m.shape[0] >= 2, m.shape

    n = max(2, round(args.seconds * FPS))
    t = ((np.arange(n) + 1) / (n + 1))[:, None]  # open-interval blend, ends meet exactly

    stand_row = np.zeros(30)
    stand_row[2] = args.stand_root_z
    stand_row[6] = 1.0  # quat xyzw = identity
    stand_row[7:] = defaults

    lead_in = stand_row + t * (m[0] - stand_row)
    lead_in[:, 3:7] = m[0, 3:7]  # hold frame-0 root orientation, no yaw spin
    lead_out = m[-1] + t * (stand_row - m[-1])
    lead_out[:, 3:7] = m[-1, 3:7]

    out = np.vstack([lead_in, m, lead_out])
    norms = np.linalg.norm(out[:, 3:7], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "non-unit quat introduced"
    np.savetxt(args.out, out, delimiter=",", fmt="%.9g")
    print(
        f"{args.out}: {out.shape[0]} frames @30 fps "
        f"({n} lead-in + {m.shape[0]} motion + {n} lead-out)"
    )


if __name__ == "__main__":
    main()
