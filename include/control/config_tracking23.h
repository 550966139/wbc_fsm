#pragma once
#include "common/tracking23.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace g1 {
// Where State_Tracking23 hands off when the motion completes or is aborted:
// the velocity policy at zero command (verified active balance; the dance
// outro ends at the stance pose inside the policy entry gate), or fixed stand
// as the fallback for bring-up.
enum class MotionEndTarget { Policy23, Stand };
// Strict loader for the 23DoF motion-tracking policy configuration. The shared
// robot contract (defaults, limits, gains, action scale) intentionally mirrors
// config/g1_23dof.json: the velocity and tracking training tasks use the same
// robot config, and commands go through the same setTargets23/validateCommand
// policy form. Only the two command guards are swapped per state at runtime.
struct ConfigTracking23 {
  Joints23 defaults{}, kp{}, kd{}, scale{}, lower{}, upper{};
  std::filesystem::path model, motion;
  float dt = .02f;
  // Pending recalibration from offline ONNX replay once the real tracking
  // policy is trained; initialized to the velocity policy's calibrated values.
  float maxTrackingError = 1.5f, maxTargetStep = 1.f;
  float entryMaxJointError = .15f, entryMaxJointVel = 1.f;
  MotionEndTarget motionEndTarget = MotionEndTarget::Policy23;
  // Motion-end settle-hold: seconds the tracking policy may hold the frozen
  // outro frame while the policy entry gate keeps failing before falling back
  // to fixed stand. The abort path gets no window (mid-dance poses cannot
  // pass anyway).
  float handoffSettleTimeout = 3.f;
  bool loopMotion = false;
  static ConfigTracking23 load(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file)
      throw std::runtime_error("cannot open 23DoF tracking config: " + path.string());
    nlohmann::json j;
    file >> j;
    for (const auto *key : {"policy_dof", "motor_count", "observation_size"})
      if (!j.at(key).is_number_integer())
        throw std::runtime_error("tracking policy dimensions must be integers");
    if (!j.at("joint_ids_map").is_array())
      throw std::runtime_error("joint_ids_map must be an array");
    for (const auto &index : j.at("joint_ids_map"))
      if (!index.is_number_integer())
        throw std::runtime_error("joint mapping indices must be integers");
    if (j.at("policy_contract") != "unitree_rl_mjlab_g1_23dof_tracking_v1" ||
        j.at("policy_dof") != 23 || j.at("motor_count") != 29 ||
        j.at("observation_size") != static_cast<int>(kTrackingObsSize) ||
        j.at("joint_ids_map") != nlohmann::json(kPolicyToMotor))
      throw std::runtime_error("unsupported 23DoF tracking policy contract or mapping");
    ConfigTracking23 c;
    auto array = [&](const char *name) {
      const auto values = j.at(name).get<std::vector<float>>();
      if (values.size() != 23)
        throw std::runtime_error(std::string(name) + " must contain 23 values");
      Joints23 a{};
      std::copy(values.begin(), values.end(), a.begin());
      for (float v : a)
        if (!std::isfinite(v))
          throw std::runtime_error("non-finite config array");
      return a;
    };
    c.defaults = array("default_joint_pos");
    c.kp = array("stiffness");
    c.kd = array("damping");
    c.scale = array("action_scale");
    c.lower = array("joint_lower");
    c.upper = array("joint_upper");
    for (std::size_t i = 0; i < 23; ++i)
      if (c.kp[i] <= 0 || c.kp[i] > 500 || c.kd[i] <= 0 || c.kd[i] > 30 || c.scale[i] <= 0 ||
          c.lower[i] >= c.upper[i] || c.defaults[i] < c.lower[i] || c.defaults[i] > c.upper[i])
        throw std::runtime_error("invalid joint limits, posture, or gains");
    c.dt = j.at("step_dt");
    if (!std::isfinite(c.dt) || std::abs(c.dt - .02f) > 1e-6f)
      throw std::runtime_error("invalid tracking policy timestep");
    const auto &s = j.at("safety");
    c.maxTrackingError = s.at("max_tracking_error");
    c.maxTargetStep = s.at("max_target_step");
    for (float v : {c.maxTrackingError, c.maxTargetStep})
      if (!std::isfinite(v) || v <= 0)
        throw std::runtime_error("invalid tracking safety limit");
    const auto &e = j.at("entry_gate");
    c.entryMaxJointError = e.at("entry_max_joint_error");
    c.entryMaxJointVel = e.at("entry_max_joint_vel");
    if (!std::isfinite(c.entryMaxJointError) || c.entryMaxJointError <= 0 ||
        c.entryMaxJointError > 1.f || !std::isfinite(c.entryMaxJointVel) ||
        c.entryMaxJointVel <= 0 || c.entryMaxJointVel > 30.f)
      throw std::runtime_error("invalid tracking entry gate thresholds");
    if (j.contains("motion_end_target")) {
      if (!j.at("motion_end_target").is_string())
        throw std::runtime_error("motion_end_target must be a string");
      const auto target = j.at("motion_end_target").get<std::string>();
      if (target == "policy23")
        c.motionEndTarget = MotionEndTarget::Policy23;
      else if (target == "stand")
        c.motionEndTarget = MotionEndTarget::Stand;
      else
        throw std::runtime_error("unsupported motion_end_target: " + target);
    }
    if (j.contains("handoff_settle_timeout")) {
      if (!j.at("handoff_settle_timeout").is_number())
        throw std::runtime_error("handoff_settle_timeout must be a number");
      c.handoffSettleTimeout = j.at("handoff_settle_timeout");
      if (!std::isfinite(c.handoffSettleTimeout) || c.handoffSettleTimeout <= 0 ||
          c.handoffSettleTimeout > 30.f)
        throw std::runtime_error("invalid handoff settle timeout");
    }
    if (j.contains("loop_motion")) {
      if (!j.at("loop_motion").is_boolean())
        throw std::runtime_error("loop_motion must be a boolean");
      c.loopMotion = j.at("loop_motion");
    }
    c.model = j.at("model_path").get<std::string>();
    c.motion = j.at("motion_path").get<std::string>();
    if (c.model.empty() || c.motion.empty())
      throw std::runtime_error("empty tracking model or motion path");
    if (c.model.is_relative())
      c.model = path.parent_path().parent_path() / c.model;
    if (c.motion.is_relative())
      c.motion = path.parent_path().parent_path() / c.motion;
    return c;
  }
};
} // namespace g1
