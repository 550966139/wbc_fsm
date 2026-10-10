#pragma once
#include "common/tracking23.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>

namespace g1 {
// Where State_Tracking23 hands off when the motion completes or is aborted:
// the velocity policy at zero command (verified active balance; the dance
// outro ends at the stance pose inside the policy entry gate), or fixed stand
// as the fallback for bring-up.
enum class MotionEndTarget { Policy23, Stand };

// One deployable dance: its own ONNX policy, motion CSV, calibrated command
// guards, entry-gate thresholds and end-of-motion behaviour. Guards are
// per-policy calibrations (replay_tracking23.py), never shared across motions.
struct TrackingMotion23 {
  std::string name;
  std::filesystem::path model, motion;
  float maxTrackingError = 1.5f, maxTargetStep = 1.f;
  float entryMaxJointError = .15f, entryMaxJointVel = 1.f;
  MotionEndTarget motionEndTarget = MotionEndTarget::Policy23;
  // Motion-end settle-hold: seconds the tracking policy may hold the frozen
  // outro frame while the policy entry gate keeps failing before falling back
  // to fixed stand. The abort path gets no window (mid-dance poses cannot
  // pass anyway).
  float handoffSettleTimeout = 3.f;
  // Pose threshold for the motion-end/abort policy-entry gate. The default
  // matches the FixedStand R2+A gate (g1::kPolicyEntryMaxJointError), which is
  // sized for stand-hold precision; a tracking policy can hold a legitimate
  // steady-state offset of a few tenths of a rad on arm joints at the outro
  // frame (measured: l_elbow 0.39 rad with |dq| settled to 0.007), so motions
  // with such an offset raise this per-entry instead of losing the handoff.
  float handoffEntryMaxJointError = .15f;
  bool loopMotion = false;
};

// Strict loader for the 23DoF motion-tracking policy configuration. The shared
// robot contract (defaults, limits, gains, action scale) intentionally mirrors
// config/g1_23dof.json: the velocity and tracking training tasks use the same
// robot config, and commands go through the same setTargets23/validateCommand
// policy form. Only the two command guards are swapped per state at runtime.
//
// The configuration names either one motion through the legacy top-level
// model_path/motion_path keys or several through a "motions" array (the two
// forms are mutually exclusive). Array entries carry their own
// model_path/motion_path/name and may override the top-level safety,
// entry_gate, motion_end_target, handoff_settle_timeout,
// handoff_entry_max_joint_error and loop_motion values; the top-level values
// remain the defaults and keep their validation.
struct ConfigTracking23 {
  Joints23 defaults{}, kp{}, kd{}, scale{}, lower{}, upper{};
  float dt = .02f;
  static constexpr std::size_t kMaxMotions = 4;
  std::vector<TrackingMotion23> motions;
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
    // Top-level command guards and entry gate: defaults every motion inherits
    // unless it carries its own override block.
    TrackingMotion23 defaults;
    const auto &s = j.at("safety");
    defaults.maxTrackingError = s.at("max_tracking_error");
    defaults.maxTargetStep = s.at("max_target_step");
    const auto &e = j.at("entry_gate");
    defaults.entryMaxJointError = e.at("entry_max_joint_error");
    defaults.entryMaxJointVel = e.at("entry_max_joint_vel");
    auto motionEndTargetOf = [](const nlohmann::json &node) {
      if (!node.is_string())
        throw std::runtime_error("motion_end_target must be a string");
      const auto target = node.get<std::string>();
      if (target == "policy23")
        return MotionEndTarget::Policy23;
      if (target == "stand")
        return MotionEndTarget::Stand;
      throw std::runtime_error("unsupported motion_end_target: " + target);
    };
    if (j.contains("motion_end_target"))
      defaults.motionEndTarget = motionEndTargetOf(j.at("motion_end_target"));
    if (j.contains("handoff_settle_timeout")) {
      if (!j.at("handoff_settle_timeout").is_number())
        throw std::runtime_error("handoff_settle_timeout must be a number");
      defaults.handoffSettleTimeout = j.at("handoff_settle_timeout");
    }
    if (j.contains("handoff_entry_max_joint_error")) {
      if (!j.at("handoff_entry_max_joint_error").is_number())
        throw std::runtime_error("handoff_entry_max_joint_error must be a number");
      defaults.handoffEntryMaxJointError = j.at("handoff_entry_max_joint_error");
    }
    if (j.contains("loop_motion")) {
      if (!j.at("loop_motion").is_boolean())
        throw std::runtime_error("loop_motion must be a boolean");
      defaults.loopMotion = j.at("loop_motion");
    }
    // Validates one motion's guard/entry/behaviour override block (or the
    // inherited defaults) with the same ranges the top-level values obey.
    auto applyOverrides = [&](TrackingMotion23 &m, const nlohmann::json &node) {
      if (node.contains("safety")) {
        const auto &os = node.at("safety");
        m.maxTrackingError = os.at("max_tracking_error");
        m.maxTargetStep = os.at("max_target_step");
      }
      if (node.contains("entry_gate")) {
        const auto &oe = node.at("entry_gate");
        m.entryMaxJointError = oe.at("entry_max_joint_error");
        m.entryMaxJointVel = oe.at("entry_max_joint_vel");
      }
      if (node.contains("motion_end_target"))
        m.motionEndTarget = motionEndTargetOf(node.at("motion_end_target"));
      if (node.contains("handoff_settle_timeout")) {
        if (!node.at("handoff_settle_timeout").is_number())
          throw std::runtime_error("handoff_settle_timeout must be a number");
        m.handoffSettleTimeout = node.at("handoff_settle_timeout");
      }
      if (node.contains("handoff_entry_max_joint_error")) {
        if (!node.at("handoff_entry_max_joint_error").is_number())
          throw std::runtime_error("handoff_entry_max_joint_error must be a number");
        m.handoffEntryMaxJointError = node.at("handoff_entry_max_joint_error");
      }
      if (node.contains("loop_motion")) {
        if (!node.at("loop_motion").is_boolean())
          throw std::runtime_error("loop_motion must be a boolean");
        m.loopMotion = node.at("loop_motion");
      }
      for (float v : {m.maxTrackingError, m.maxTargetStep})
        if (!std::isfinite(v) || v <= 0)
          throw std::runtime_error("invalid tracking safety limit");
      if (!std::isfinite(m.entryMaxJointError) || m.entryMaxJointError <= 0 ||
          m.entryMaxJointError > 1.f || !std::isfinite(m.entryMaxJointVel) ||
          m.entryMaxJointVel <= 0 || m.entryMaxJointVel > 30.f)
        throw std::runtime_error("invalid tracking entry gate thresholds");
      if (!std::isfinite(m.handoffSettleTimeout) || m.handoffSettleTimeout <= 0 ||
          m.handoffSettleTimeout > 30.f)
        throw std::runtime_error("invalid handoff settle timeout");
      if (!std::isfinite(m.handoffEntryMaxJointError) || m.handoffEntryMaxJointError <= 0 ||
          m.handoffEntryMaxJointError > 1.f)
        throw std::runtime_error("invalid handoff entry gate pose threshold");
    };
    auto resolvePath = [&](const std::string &value) {
      std::filesystem::path resolved{value};
      if (resolved.is_relative())
        resolved = path.parent_path().parent_path() / resolved;
      return resolved;
    };
    if (j.contains("motions")) {
      if (j.contains("model_path") || j.contains("motion_path"))
        throw std::runtime_error(
            "motions array cannot be combined with legacy top-level model/motion paths");
      const auto &motions = j.at("motions");
      if (!motions.is_array() || motions.empty() || motions.size() > kMaxMotions)
        throw std::runtime_error("motions must be an array of 1..4 motion entries");
      std::set<std::string> names;
      for (const auto &node : motions) {
        TrackingMotion23 m = defaults;
        if (!node.is_object() || !node.at("name").is_string() ||
            node.at("name").get<std::string>().empty())
          throw std::runtime_error("every motion entry needs a non-empty string name");
        m.name = node.at("name").get<std::string>();
        if (!names.insert(m.name).second)
          throw std::runtime_error("duplicate tracking motion name: " + m.name);
        m.model = resolvePath(node.at("model_path").get<std::string>());
        m.motion = resolvePath(node.at("motion_path").get<std::string>());
        if (node.at("model_path").get<std::string>().empty() ||
            node.at("motion_path").get<std::string>().empty())
          throw std::runtime_error("empty tracking model or motion path");
        applyOverrides(m, node);
        c.motions.push_back(std::move(m));
      }
    } else {
      TrackingMotion23 m = defaults;
      m.name = "tracking";
      const auto modelPath = j.at("model_path").get<std::string>();
      const auto motionPath = j.at("motion_path").get<std::string>();
      if (modelPath.empty() || motionPath.empty())
        throw std::runtime_error("empty tracking model or motion path");
      m.model = resolvePath(modelPath);
      m.motion = resolvePath(motionPath);
      applyOverrides(m, j);
      c.motions.push_back(std::move(m));
    }
    return c;
  }
};
} // namespace g1
