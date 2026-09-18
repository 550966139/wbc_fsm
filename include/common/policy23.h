#pragma once
#include "common/g1_23dof.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace g1 {
using Joints23 = std::array<float, kPolicyDof>;
using Observation23 = std::array<float, 80>;
inline std::array<float, 3> gravityFromQuaternion(const std::array<float, 4> &q) {
  float norm = 0;
  for (float v : q)
    norm += v * v;
  if (!std::isfinite(norm) || std::abs(norm - 1.f) > .1f)
    throw std::runtime_error("invalid IMU quaternion");
  const float s = 1.f / std::sqrt(norm);
  const float w = q[0] * s, x = q[1] * s, y = q[2] * s, z = q[3] * s;
  return {2.f * (w * y - x * z), -2.f * (y * z + w * x), 2.f * (x * x + y * y) - 1.f};
}
// Official velocity-v0 concatenation. Phase advances even when command is zero;
// sin/cos are masked when command norm is below 0.1.
inline Observation23 observation23(const std::array<float, 3> &gyro,
                                   const std::array<float, 3> &gravity,
                                   const std::array<float, 3> &command, const Joints23 &q,
                                   const Joints23 &dq, const Joints23 &defaults,
                                   const Joints23 &lastAction, float &phase, float dt,
                                   float period) {
  if (!std::isfinite(phase) || phase < 0.f || phase >= 1.f || !std::isfinite(dt) || dt <= 0.f ||
      !std::isfinite(period) || period <= 0.f)
    throw std::runtime_error("invalid policy phase or timing");
  Observation23 obs{};
  phase = std::fmod(phase + dt / period, 1.f);
  if (!std::isfinite(phase))
    throw std::runtime_error("policy phase overflow");
  float norm = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    obs[i] = gyro[i];
    obs[3 + i] = gravity[i];
    obs[6 + i] = command[i];
    norm += command[i] * command[i];
  }
  if (std::sqrt(norm) >= .1f) {
    obs[9] = std::sin(phase * 6.28318530718f);
    obs[10] = std::cos(phase * 6.28318530718f);
  }
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    obs[11 + i] = q[i] - defaults[i];
    obs[34 + i] = dq[i];
    obs[57 + i] = lastAction[i];
  }
  for (float v : obs)
    if (!std::isfinite(v))
      throw std::runtime_error("non-finite policy observation");
  return obs;
}
inline Joints23 actionTargets23(const Joints23 &action, const Joints23 &scale,
                                const Joints23 &offsets) {
  Joints23 result{};
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    if (!std::isfinite(action[i]) || !std::isfinite(scale[i]) || !std::isfinite(offsets[i]) ||
        scale[i] <= 0.f)
      throw std::runtime_error("invalid policy action transform");
    result[i] = offsets[i] + action[i] * scale[i];
    if (!std::isfinite(result[i]))
      throw std::runtime_error("non-finite policy action");
  }
  return result;
}
// Training saturates position targets at the actuator ctrlrange (MuJoCo clamps
// silently); the deployment must reproduce that saturation instead of tripping.
inline Joints23 clampTargets23(const Joints23 &targets, const Joints23 &lower,
                               const Joints23 &upper) {
  Joints23 result{};
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    if (!std::isfinite(targets[i]) || !std::isfinite(lower[i]) || !std::isfinite(upper[i]) ||
        lower[i] >= upper[i])
      throw std::runtime_error("invalid joint limits for target clamping");
    result[i] = std::clamp(targets[i], lower[i], upper[i]);
  }
  return result;
}
inline void validateTargets23(const Joints23 &targets, const Joints23 &measured,
                              const Joints23 &previous, const Joints23 &lower,
                              const Joints23 &upper, float maxTrackingError, float maxTargetStep) {
  if (!std::isfinite(maxTrackingError) || maxTrackingError <= 0.f ||
      !std::isfinite(maxTargetStep) || maxTargetStep <= 0.f)
    throw std::runtime_error("invalid command safety limits");
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    if (!std::isfinite(targets[i]) || !std::isfinite(measured[i]) || !std::isfinite(previous[i]) ||
        !std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i] >= upper[i])
      throw std::runtime_error("non-finite or invalid joint command");
    if (targets[i] < lower[i] || targets[i] > upper[i])
      throw std::runtime_error("joint target outside configured limits");
    if (std::abs(targets[i] - measured[i]) > maxTrackingError)
      throw std::runtime_error("excessive joint tracking error");
    if (std::abs(targets[i] - previous[i]) > maxTargetStep)
      throw std::runtime_error("excessive joint target step");
  }
}
} // namespace g1
