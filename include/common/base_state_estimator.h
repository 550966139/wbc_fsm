#pragma once
#include "common/policy23.h"
#include <array>
#include <cstddef>

namespace g1 {
// Contact-aided base linear velocity estimator (obs-v2, contract v2's extra
// observation block). Frozen offline 2026-10-13 against the sim rig's
// rt/sportmodestate ground truth (two independent F10 captures; crouch-window
// RMSE vxy 0.19 / vz 0.09-0.10 m/s, see ~/sim2sim_g1/est_dev/estimator_proto3.py
// and docs/IMPLEMENTATION_STATUS.md):
//   - 23DoF leg FK + geometric Jacobian to four sole candidates (heel->toe),
//     selecting per leg the candidate whose implied pelvis velocity is closest
//     to the IMU-integrated prediction (handles contact-point migration),
//   - stance gate: torque-load fraction >= 0.28 AND swing speed
//     |dq_hip_pitch| + |dq_knee| <= 1.8 rad/s (keeps swinging legs out),
//   - IMU complementary fusion: gravity-removed accelerometer integration
//     (beta 0.98 at 50 Hz) anchored by the load-weighted leg odometry,
//   - one-pole prefilters: q/dq fc 5 Hz, tau fc 4 Hz, accel fc 15 Hz.
// All inputs come from rt/lowstate (q, dq, tau_est, quaternion, gyroscope,
// accelerometer) — fields available in both sim and hardware dev mode; the
// estimator never depends on ddq (unfilled in the simulator bridge).
// Output: IMU-site linear velocity in the pelvis frame (matches the training
// sensor robot/imu_lin_vel), i.e. pelvis-origin velocity + gyro x r_IMU with
// r_IMU = (0.04525, 0, -0.08339) (imu_in_pelvis site, training XML).
class BaseStateEstimator23 {
public:
  struct Inputs {
    Joints23 q{}, dq{}, tau{};             // policy joint order; tau = motor tau_est
    std::array<float, 4> quat{1, 0, 0, 0}; // w, x, y, z pelvis->world (IMU)
    std::array<float, 3> gyro{}, accel{};
    bool valid = false; // fresh, finite LowState snapshot
  };
  // One 50 Hz control-cycle update; returns the v2 observation block.
  std::array<float, 3> update(const Inputs &in);
  std::array<float, 3> pelvisVelocity() const { return v_; }
  bool stanceLeft() const { return stance_[0]; }
  bool stanceRight() const { return stance_[1]; }

private:
  void filter(const Inputs &in);
  std::array<float, kPolicyDof> q_[2]{}, dq_[2]{}, tau_[2]{}; // filter states
  std::array<float, 3> acc_[1]{};
  std::array<float, kPolicyDof> qf_{}, dqf_{}, tauf_{};
  std::array<float, 3> accf_{};
  std::array<float, 3> v_{};
  std::array<bool, 2> stance_{true, true};
};
} // namespace g1
