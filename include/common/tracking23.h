#pragma once
#include "common/policy23.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace g1 {
// 23DoF whole-body motion-tracking contract (unitree_rl_mjlab_g1_23dof_tracking_v1):
// single float input [1,124], single float output [1,23], normalization embedded
// in-graph. Observation layout (exact order, see docs/tracking23.md):
//   [0:46]   reference joint_pos(23) then reference joint_vel(23), current frame
//   [46:52]  anchor orientation error, rotation matrix first two columns row-major
//   [52:55]  base angular velocity (IMU gyro)
//   [55:78]  joint_pos relative to defaults (23)
//   [78:101] joint_vel (23)
//   [101:124] last action (23)
constexpr std::size_t kTrackingObsSize = 124;
constexpr std::size_t kTrackingRefPos = 0;
constexpr std::size_t kTrackingRefVel = 23;
constexpr std::size_t kTrackingAnchor = 46;
constexpr std::size_t kTrackingGyro = 52;
constexpr std::size_t kTrackingJointPos = 55;
constexpr std::size_t kTrackingJointVel = 78;
constexpr std::size_t kTrackingLastAction = 101;
// Policy joint index of waist_yaw, the anchor body's yaw degree of freedom.
constexpr std::size_t kWaistYawJoint = 12;
// Tracking entry ramp-in: number of 50 Hz cycles over which outgoing targets
// blend from the measured entry pose to the policy targets (0.4 s). Removes
// the controller-handoff command jump without touching the observation chain.
constexpr std::size_t kTrackingEntryRampCycles = 20;

using ObservationTracking23 = std::array<float, kTrackingObsSize>;
using Quat23 = std::array<float, 4>; // w, x, y, z

// wxyz Hamilton product, same convention as mathTools.h quat_mul.
inline Quat23 quatMul23(const Quat23 &a, const Quat23 &b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
          a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
          a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
inline Quat23 normalizedQuat23(const Quat23 &q, const char *label) {
  float norm = 0;
  for (float v : q)
    norm += v * v;
  if (!std::isfinite(norm) || std::abs(norm - 1.f) > .1f)
    throw std::runtime_error(std::string("invalid ") + label + " quaternion");
  const float s = 1.f / std::sqrt(norm);
  return {q[0] * s, q[1] * s, q[2] * s, q[3] * s};
}
inline Quat23 quatAboutZ23(float radians) {
  if (!std::isfinite(radians))
    throw std::runtime_error("non-finite yaw angle");
  const float half = .5f * radians;
  return {std::cos(half), 0.f, 0.f, std::sin(half)};
}
// torso_link world orientation on the hardware frame: IMU quaternion composed
// with the waist yaw joint rotation about the body z axis (training XML).
inline Quat23 robotAnchorQuat23(const Quat23 &imuQuat, float waistYaw) {
  return quatMul23(normalizedQuat23(imuQuat, "IMU"), quatAboutZ23(waistYaw));
}
// Rotation about world z: atan2(R[1][0], R[0][0]) of the quaternion's matrix.
inline float yawOf23(const Quat23 &q) {
  const Quat23 n = normalizedQuat23(q, "anchor");
  return std::atan2(2.f * (n[0] * n[3] + n[1] * n[2]), 1.f - 2.f * (n[2] * n[2] + n[3] * n[3]));
}
// Relative orientation quat_rel = robotAnchor^-1 (x) refAnchor (the orientation
// part of subtract_frame_transforms), as a rotation matrix whose first two
// columns are flattened row-major from the [3,2] slice:
// {R[0][0], R[0][1], R[1][0], R[1][1], R[2][0], R[2][1]}.
inline std::array<float, 6> anchorError6(const Quat23 &robotAnchor, const Quat23 &refAnchor) {
  const Quat23 r = normalizedQuat23(robotAnchor, "robot anchor");
  const Quat23 f = normalizedQuat23(refAnchor, "reference anchor");
  const Quat23 rel = quatMul23({r[0], -r[1], -r[2], -r[3]}, f);
  const float w = rel[0], x = rel[1], y = rel[2], z = rel[3];
  return {1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * y + w * z),
          1.f - 2.f * (x * x + z * z), 2.f * (x * z - w * y), 2.f * (y * z + w * x)};
}
// Velocity-policy entry thresholds, shared by the FixedStand R2+A gate and the
// tracking -> policy handoff gate (the dance outro ends at the stance pose, so
// the same envelope applies).
constexpr float kPolicyEntryMaxJointError = .15f;
constexpr float kPolicyEntryMaxJointVel = 1.f;
// Worst per-joint deviation of the measured pose from a 23-dim reference pose,
// plus the worst absolute joint velocity. Single implementation behind the
// FixedStand R2+A/R2+B gates and the tracking <-> policy handoff gates.
struct EntryGate23 {
  float worstError = 0.f, worstVelocity = 0.f;
  std::size_t worstJoint = 0;
  bool passes(float maxJointError, float maxJointVelocity) const {
    return worstError <= maxJointError && worstVelocity <= maxJointVelocity;
  }
};
inline EntryGate23 entryGate23(const Joints23 &q, const Joints23 &dq, const Joints23 &reference) {
  EntryGate23 gate;
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    if (!std::isfinite(q[i]) || !std::isfinite(dq[i]) || !std::isfinite(reference[i]))
      throw std::runtime_error("non-finite entry gate input");
    const float error = std::abs(q[i] - reference[i]);
    if (error > gate.worstError) {
      gate.worstError = error;
      gate.worstJoint = i;
    }
    gate.worstVelocity = std::max(gate.worstVelocity, std::abs(dq[i]));
  }
  return gate;
}
// Build the 124-dim tracking observation for one control cycle. refPos/refVel
// point at the 23 reference values of the current motion frame; refAnchor is
// the reference anchor quaternion after entry yaw alignment.
inline ObservationTracking23
observationTracking23(const float *refPos, const float *refVel, const Quat23 &robotAnchor,
                      const Quat23 &refAnchor, const std::array<float, 3> &gyro, const Joints23 &q,
                      const Joints23 &dq, const Joints23 &defaults, const Joints23 &lastAction) {
  if (refPos == nullptr || refVel == nullptr)
    throw std::runtime_error("missing tracking reference frame");
  ObservationTracking23 obs{};
  const auto anchor = anchorError6(robotAnchor, refAnchor);
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    obs[kTrackingRefPos + i] = refPos[i];
    obs[kTrackingRefVel + i] = refVel[i];
    obs[kTrackingJointPos + i] = q[i] - defaults[i];
    obs[kTrackingJointVel + i] = dq[i];
    obs[kTrackingLastAction + i] = lastAction[i];
  }
  for (std::size_t i = 0; i < 6; ++i)
    obs[kTrackingAnchor + i] = anchor[i];
  for (std::size_t i = 0; i < 3; ++i)
    obs[kTrackingGyro + i] = gyro[i];
  for (float v : obs)
    if (!std::isfinite(v))
      throw std::runtime_error("non-finite tracking observation");
  return obs;
}
} // namespace g1
