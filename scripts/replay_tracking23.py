#!/usr/bin/env python3
"""Offline ONNX replay for tracking-policy safety-guard calibration.

Replays each configured deployment motion CSV through its exported tracking
policy under the idealized assumption "robot state == reference state" and
measures the policy's action envelope:

  * target step:  max_k max_j |target[k] - target[k-1]|   -> max_target_step
  * tracking err: max_k max_j |target[k] - ref_pos[k]|    -> max_tracking_error

Suggested guard values are envelope x margin (default 1.5), rounded up to one
decimal — the same practice used for the velocity-policy calibrations. The
observation layout mirrors the deployment contract exactly (ref pos/vel,
identity anchor term, zero gyro, q-defaults, dq, chained last action) and the
target transform uses the config's own default_joint_pos/action_scale arrays,
so the envelope reflects what the deployed controller would command.

Supports both the legacy single-motion config and the "motions" array (a
multi-dance library); every motion has its own guards, so calibration always
runs per motion.

Usage:
    replay_tracking23.py [--config config/g1_23dof_tracking.json] [--margin 1.5]
                         [--write] [--motion NAME]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


def replay(model_path, motion_path, defaults, scale):
    motion = np.loadtxt(motion_path, delimiter=",")
    assert motion.ndim == 2 and motion.shape[1] == 50, motion.shape
    ref_pos, ref_vel = motion[:, :23], motion[:, 23:46]

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    width = sess.get_inputs()[0].shape[1]
    assert list(sess.get_inputs()[0].shape) == [1, width] and width in (124, 127)
    assert list(sess.get_outputs()[0].shape) == [1, 23]

    last_action = np.zeros(23, dtype=np.float32)
    targets = np.zeros((motion.shape[0], 23), dtype=np.float64)
    anchor_identity = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # v2 (127) inserts the estimated base_lin_vel block before the gyro; the
    # idealized replay zeroes it (same convention as the gyro block).
    vel_block = [np.zeros(3, dtype=np.float32)] if width == 127 else []
    for k in range(motion.shape[0]):
        obs = np.concatenate(
            [
                ref_pos[k],
                ref_vel[k],
                anchor_identity,
                *vel_block,
                np.zeros(3, dtype=np.float32),
                ref_pos[k] - defaults,
                ref_vel[k],
                last_action,
            ]
        ).astype(np.float32)
        assert obs.shape == (width,)
        action = sess.run(None, {in_name: obs[None, :]})[0][0]
        targets[k] = defaults + action * scale
        last_action = action.astype(np.float32)

    step = float(np.max(np.abs(np.diff(targets, axis=0))))
    err = float(np.max(np.abs(targets - ref_pos)))
    return motion.shape[0], step, err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/g1_23dof_tracking.json")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--write", action="store_true", help="write suggested guards into the config JSON")
    ap.add_argument("--motion", default=None, help="calibrate only the named motion (motions[] configs)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    repo = cfg_path.resolve().parent.parent
    defaults = np.array(cfg["default_joint_pos"], dtype=np.float64)
    scale = np.array(cfg["action_scale"], dtype=np.float64)
    assert defaults.shape == (23,) and scale.shape == (23,)

    if "motions" in cfg:
        entries = [(m.get("name", f"motion{i}"), repo / m["model_path"], repo / m["motion_path"], m)
                   for i, m in enumerate(cfg["motions"])]
    else:
        entries = [(cfg.get("name", "tracking"), repo / cfg["model_path"],
                    repo / cfg["motion_path"], cfg)]

    for name, model_path, motion_path, node in entries:
        if args.motion is not None and name != args.motion:
            continue
        frames, step, err = replay(model_path, motion_path, defaults, scale)
        sug_step = math.ceil(step * args.margin * 10) / 10
        sug_err = math.ceil(err * args.margin * 10) / 10
        print(f"motion \"{name}\": frames {frames}, margin x{args.margin}")
        print(f"  target-step envelope  {step:.3f} rad -> suggest max_target_step {sug_step}")
        print(f"  tracking-err envelope {err:.3f} rad -> suggest max_tracking_error {sug_err}")
        cur = node.get("safety", {})
        print(f"  current guards: max_target_step {cur.get('max_target_step')}, "
              f"max_tracking_error {cur.get('max_tracking_error')}")
        if args.write:
            node.setdefault("safety", {})["max_target_step"] = sug_step
            node["safety"]["max_tracking_error"] = sug_err

    if args.write:
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {cfg_path}")


if __name__ == "__main__":
    main()
