#!/usr/bin/env python3
"""Convert a tracking-training NPZ into the deployment motion CSV.

The deployment contract (unitree_rl_mjlab_g1_23dof_tracking_v1) consumes one
CSV row per 50 Hz control frame with 50 columns:

    ref_joint_pos[23], ref_joint_vel[23], ref_anchor_quat_wxyz[4]

Values are copied verbatim from the training NPZ produced by
unitree_rl_mjlab/scripts/csv_to_npz.py so deployment observations stay
bit-identical to training (especially the finite-differenced joint_vel).

Usage:
    npz_to_motion_csv.py --npz dance1_subject2_50hz.npz --out motion.csv \
        [--body-name torso_link | --body-index 13]
"""

import argparse

import numpy as np

# Body order of the unitree_rl_mjlab g1_23dof robot (MJCF body order, matches
# the body_* arrays written by csv_to_npz.py).
G1_23DOF_BODIES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="training NPZ from csv_to_npz.py")
    ap.add_argument("--out", required=True, help="output deployment CSV path")
    ap.add_argument("--body-name", default="torso_link", help="anchor body name")
    ap.add_argument("--body-index", type=int, default=None, help="override anchor body index")
    args = ap.parse_args()

    data = np.load(args.npz)
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 23:
        raise SystemExit(f"joint_pos shape {joint_pos.shape}, expected (N, 23)")
    if joint_vel.shape != joint_pos.shape:
        raise SystemExit(f"joint_vel shape {joint_vel.shape} != joint_pos {joint_pos.shape}")
    if body_quat.ndim != 3 or body_quat.shape[2] != 4 or body_quat.shape[0] != joint_pos.shape[0]:
        raise SystemExit(f"body_quat_w shape {body_quat.shape}, expected (N, M, 4)")
    if fps is not None and abs(fps - 50.0) > 1e-6:
        raise SystemExit(f"npz fps {fps}, deployment cursor assumes 50 Hz")

    if args.body_index is not None:
        idx = args.body_index
    else:
        if args.body_name not in G1_23DOF_BODIES:
            raise SystemExit(f"unknown body {args.body_name!r}; use --body-index")
        idx = G1_23DOF_BODIES.index(args.body_name)
    if not 0 <= idx < body_quat.shape[1]:
        raise SystemExit(f"body index {idx} out of range (M={body_quat.shape[1]})")

    quat = body_quat[:, idx, :]
    norms = np.linalg.norm(quat, axis=1)
    if not np.all(np.isfinite(joint_pos)) or not np.all(np.isfinite(joint_vel)):
        raise SystemExit("non-finite joint data")
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = int(np.argmax(np.abs(norms - 1.0)))
        raise SystemExit(f"non-unit quat at frame {bad}: norm {norms[bad]}")

    rows = np.concatenate([joint_pos, joint_vel, quat], axis=1)
    np.savetxt(args.out, rows, delimiter=",", fmt="%.9g")
    frames = rows.shape[0]
    print(
        f"wrote {args.out}: {frames} frames ({frames / 50.0:.2f} s @50 Hz), "
        f"anchor body {args.body_name} index {idx}"
    )


if __name__ == "__main__":
    main()
