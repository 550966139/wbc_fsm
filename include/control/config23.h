#pragma once
#include "common/policy23.h"
#include "common/safety_supervisor.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace g1 {
struct Config23 {
  Joints23 defaults{}, kp{}, kd{}, scale{}, lower{}, upper{};
  Joints23 standKp{}, standKd{}, standFf{};
  std::array<float, 4> commandLimit{};
  std::filesystem::path model;
  float dt = .02f, gaitPeriod = .6f, standDuration = 3.f;
  control::SafetyLimits safety;
  static Config23 load(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file)
      throw std::runtime_error("cannot open 23DoF config: " + path.string());
    nlohmann::json j;
    file >> j;
    for (const auto *key : {"policy_dof", "motor_count", "observation_size"})
      if (!j.at(key).is_number_integer())
        throw std::runtime_error("policy dimensions must be integers");
    if (!j.at("joint_ids_map").is_array())
      throw std::runtime_error("joint_ids_map must be an array");
    for (const auto &index : j.at("joint_ids_map"))
      if (!index.is_number_integer())
        throw std::runtime_error("joint mapping indices must be integers");
    if (j.at("policy_contract") != "unitree_rl_mjlab_g1_23dof_velocity_v1" ||
        j.at("policy_dof") != 23 || j.at("motor_count") != 29 || j.at("observation_size") != 81 ||
        j.at("joint_ids_map") != nlohmann::json(kPolicyToMotor))
      throw std::runtime_error("unsupported 23DoF policy contract or mapping");
    Config23 c;
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
    // Stand gains are optional; without them the stand form equals the policy
    // contract and FixedStand keeps its pre-stand-gain behaviour.
    c.standKp = c.kp;
    c.standKd = c.kd;
    if (j.contains("stand_stiffness") || j.contains("stand_damping")) {
      c.standKp = array("stand_stiffness");
      c.standKd = array("stand_damping");
      for (std::size_t i = 0; i < 23; ++i)
        if (c.standKp[i] <= 0 || c.standKp[i] > 500 || c.standKd[i] <= 0 || c.standKd[i] > 30)
          throw std::runtime_error("invalid stand gains");
    }
    // Static gravity feedforward of the default stance (model-computed,
    // feet-grounded). Optional like the stand gains; absent means zero.
    if (j.contains("stand_gravity_ff")) {
      c.standFf = array("stand_gravity_ff");
      for (std::size_t i = 0; i < 23; ++i)
        if (std::abs(c.standFf[i]) > 88.f)
          throw std::runtime_error("stand gravity feedforward exceeds joint torque authority");
    }
    c.dt = j.at("step_dt");
    c.gaitPeriod = j.at("gait_period");
    c.standDuration = j.at("stand_duration");
    if (!std::isfinite(c.dt) || std::abs(c.dt - .02f) > 1e-6f || !std::isfinite(c.gaitPeriod) ||
        std::abs(c.gaitPeriod - .6f) > 1e-6f || !std::isfinite(c.standDuration) ||
        c.standDuration < 2 || c.standDuration > 30)
      throw std::runtime_error("invalid policy timing");
    c.commandLimit = j.at("deployment_command_limits").get<std::array<float, 4>>();
    const std::array<float, 4> maxCommand{.5f, .5f, 1.f, .35f};
    for (std::size_t i = 0; i < 4; ++i)
      if (!std::isfinite(c.commandLimit[i]) || c.commandLimit[i] < 0 ||
          c.commandLimit[i] > maxCommand[i])
        throw std::runtime_error("invalid command limits");
    c.safety.projectedGravityThreshold = j.at("orientation_threshold");
    const auto &s = j.at("safety");
    c.safety.recoveryGravityThreshold = s.at("recovery_gravity_threshold");
    c.safety.maxJointVelocity = s.at("max_joint_velocity");
    c.safety.maxTrackingError = s.at("max_tracking_error");
    c.safety.maxTargetStep = s.at("max_target_step");
    for (float v : {c.safety.projectedGravityThreshold, c.safety.recoveryGravityThreshold,
                    c.safety.maxJointVelocity, c.safety.maxTrackingError, c.safety.maxTargetStep})
      if (!std::isfinite(v) || v <= 0)
        throw std::runtime_error("invalid safety limit");
    if (c.safety.projectedGravityThreshold >= 1 ||
        c.safety.recoveryGravityThreshold >= c.safety.projectedGravityThreshold)
      throw std::runtime_error("invalid orientation thresholds");
    auto ms = [&](const char *key, int min, int max) {
      if (!s.at(key).is_number_integer())
        throw std::runtime_error("safety durations must be integer milliseconds");
      if (s.at(key) < min || s.at(key) > max)
        throw std::runtime_error("invalid safety duration");
      int value = s.at(key).get<int>();
      return std::chrono::milliseconds(value);
    };
    c.safety.stateTimeout = ms("state_timeout_ms", 20, 100);
    c.safety.tiltConfirmation = ms("tilt_confirmation_ms", 0, 100);
    c.safety.recoveryDuration = ms("recovery_duration_ms", 100, 5000);
    c.model = j.at("model_path").get<std::string>();
    if (c.model.empty())
      throw std::runtime_error("empty model path");
    if (c.model.is_relative())
      c.model = path.parent_path().parent_path() / c.model;
    return c;
  }
};
} // namespace g1
