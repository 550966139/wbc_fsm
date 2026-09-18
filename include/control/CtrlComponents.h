#ifndef CTRLCOMPONENTS_H
#define CTRLCOMPONENTS_H

#include "control/config23.h"
#include "interface/IOInterface.h"
#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"
#include <optional>

struct CtrlComponents {
  explicit CtrlComponents(IOInterface *io) : ioInter(io) {
    lowCmd = new LowlevelCmd();
    lowState = new LowlevelState();
  }
  ~CtrlComponents() {
    delete lowCmd;
    delete lowState;
    delete ioInter;
  }
  CtrlComponents(const CtrlComponents &) = delete;
  CtrlComponents &operator=(const CtrlComponents &) = delete;

  LowlevelCmd *lowCmd;
  LowlevelState *lowState;
  IOInterface *ioInter;
  double dt = .02;
  bool *running = nullptr;
  bool exitFlag = false;
  CtrlPlatform ctrlPlatform = CtrlPlatform::MUJOCO;
  control::SafetySupervisor safety;
  std::optional<g1::Config23> config23;

  void configure23(const g1::Config23 &config) {
    config23 = config;
    dt = config.dt;
    safety = control::SafetySupervisor(config.safety);
    setDampingCommand();
  }
  bool use23() const { return config23.has_value(); }

  std::array<float, 3> getGravity() const {
    const auto &q = lowState->imu.quaternion;
    return g1::gravityFromQuaternion({q[0], q[1], q[2], q[3]});
  }
  g1::Joints23 positions23() const {
    g1::Joints23 result{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      result[i] = lowState->motorState[g1::kPolicyToMotor[i]].q;
    return result;
  }
  g1::Joints23 velocities23() const {
    g1::Joints23 result{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      result[i] = lowState->motorState[g1::kPolicyToMotor[i]].dq;
    return result;
  }
  bool safetyCheck() {
    try {
      // Invalid joints must invalidate the recovery dwell as well as motion.
      if (use23()) {
        for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
          const auto &joint = lowState->motorState[g1::kPolicyToMotor[i]];
          if (!std::isfinite(joint.q) || !std::isfinite(joint.dq) ||
              joint.q < config23->lower[i] - .05f || joint.q > config23->upper[i] + .05f ||
              std::abs(joint.dq) > safety.limits().maxJointVelocity)
            throw std::runtime_error("invalid joint state or excessive velocity");
        }
        for (float v : lowState->imu.gyroscope)
          if (!std::isfinite(v))
            throw std::runtime_error("invalid IMU angular velocity");
        const auto &u = lowState->userValue;
        for (float v : {u.lx, u.ly, u.rx, u.ry})
          if (!std::isfinite(v) || std::abs(v) > 1.05f)
            throw std::runtime_error("invalid joystick input");
      }
      return safety.checkState(lowState->received, lowState->receivedAt, getGravity());
    } catch (const std::exception &error) {
      safety.trip(error.what());
      return false;
    }
  }
  void setDampingCommand() {
    for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
      auto &motor = lowCmd->motorCmd[slot];
      motor = MotorCmd{};
      if (!use23() || g1::isPolicyMotor(slot)) {
        motor.mode = 1;
        motor.Kd = 3.f;
      }
    }
    previousActive_ = false;
    standBlend_ = 0.f;
  }
  void setTargets23(const g1::Joints23 &targets) {
    // The six absent motor slots remain disabled, including during transitions.
    for (auto &motor : lowCmd->motorCmd)
      motor = MotorCmd{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      auto &motor = lowCmd->motorCmd[g1::kPolicyToMotor[i]];
      motor.mode = 1;
      motor.q = targets[i];
      motor.Kp = config23->kp[i];
      motor.Kd = config23->kd[i];
    }
    standBlend_ = 0.f;
  }
  // Stand form of the position command: gains and the stance gravity
  // feedforward ramp from the policy contract to the stand contract along the
  // FixedStand interpolation so the pull-up stays smooth while the posture
  // becomes load bearing.
  void setStandTargets23(const g1::Joints23 &targets, float blend) {
    if (!std::isfinite(blend) || blend < 0.f || blend > 1.f)
      throw std::runtime_error("stand blend outside [0, 1]");
    for (auto &motor : lowCmd->motorCmd)
      motor = MotorCmd{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      auto &motor = lowCmd->motorCmd[g1::kPolicyToMotor[i]];
      motor.mode = 1;
      motor.q = targets[i];
      motor.Kp = standGain23(config23->kp[i], config23->standKp[i], blend);
      motor.Kd = standGain23(config23->kd[i], config23->standKd[i], blend);
      motor.tau = standTau23(config23->standFf[i], blend);
    }
    standBlend_ = blend;
  }
  void validateCommand() {
    if (!use23())
      return;
    g1::Joints23 targets{};
    bool active = false;
    for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
      const auto &c = lowCmd->motorCmd[slot];
      if (!std::isfinite(c.q) || !std::isfinite(c.dq) || !std::isfinite(c.tau) ||
          !std::isfinite(c.Kp) || !std::isfinite(c.Kd) || c.Kp < 0 || c.Kd < 0 || c.Kp > 500.f ||
          c.Kd > 30.f)
        throw std::runtime_error("invalid motor command or gains");
      if (!g1::isPolicyMotor(slot) &&
          (c.mode != 0 || c.q != 0 || c.dq != 0 || c.tau != 0 || c.Kp != 0 || c.Kd != 0))
        throw std::runtime_error("command addressed an absent 23DoF joint");
      if (g1::isPolicyMotor(slot) && (c.mode != 1 || c.dq != 0))
        throw std::runtime_error("unsupported 23DoF motor command");
      if (c.tau != 0 && c.Kp == 0)
        throw std::runtime_error("torque feedforward requires an active position command");
      active = active || c.Kp > 0;
    }
    if (active) {
      for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
        const auto &c = lowCmd->motorCmd[g1::kPolicyToMotor[i]];
        // Legal forms are the policy contract (blend 0) and the stand ramp
        // toward the stand contract; the validator recomputes the same blend
        // the active setter recorded.
        if (c.Kp != standGain23(config23->kp[i], config23->standKp[i], standBlend_) ||
            c.Kd != standGain23(config23->kd[i], config23->standKd[i], standBlend_) ||
            c.tau != standTau23(config23->standFf[i], standBlend_))
          throw std::runtime_error("23DoF gains or feedforward differ from policy or stand contract");
        targets[i] = c.q;
      }
      const auto measured = positions23();
      g1::validateTargets23(targets, measured, previousActive_ ? previousTargets_ : measured,
                            config23->lower, config23->upper, safety.limits().maxTrackingError,
                            safety.limits().maxTargetStep);
      previousTargets_ = targets;
    }
    previousActive_ = active;
  }
  void sendRecv() { ioInter->sendRecv(lowCmd, lowState); }

private:
  // Single interpolation point shared by the stand setter and the validator so
  // the exact-match gain check stays bit identical to the command produced.
  static float standGain23(float policyGain, float standGain, float blend) {
    return (1.f - blend) * policyGain + blend * standGain;
  }
  static float standTau23(float standFeedforward, float blend) {
    return blend * standFeedforward;
  }
  g1::Joints23 previousTargets_{};
  bool previousActive_ = false;
  float standBlend_ = 0.f;
};
#endif
