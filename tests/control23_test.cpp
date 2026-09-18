#include "control/CtrlComponents.h"
#include "test_support.h"
#include <limits>

namespace {
class MockIO final : public IOInterface {
public:
  void receive(LowlevelState *) override {}
  void send(const LowlevelCmd *) override {}
  void sendRecv(const LowlevelCmd *, LowlevelState *) override {}
};
} // namespace

int main() {
  TestSuite suite;
  const auto config =
      g1::Config23::load(std::filesystem::path(PROJECT_ROOT_DIR) / "config/g1_23dof.json");
  CtrlComponents control(new MockIO());
  control.configure23(config);
  suite.check(control.use23(), "23DoF configuration selected");
  for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
    const auto &motor = control.lowCmd->motorCmd[slot];
    const bool present = g1::isPolicyMotor(slot);
    suite.check(motor.mode == (present ? 1u : 0u), "damping enables only existing motors");
    suite.near(motor.Kd, present ? 3.f : 0.f, "absent motors have no damping output");
    suite.check(motor.q == 0.f && motor.dq == 0.f && motor.tau == 0.f && motor.Kp == 0.f,
                "damping has no position or torque feedforward");
  }
  auto setMeasured = [&](const g1::Joints23 &positions) {
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      auto &motor = control.lowState->motorState[g1::kPolicyToMotor[i]];
      motor.q = positions[i];
      motor.dq = 0.f;
    }
    control.lowState->imu.quaternion[0] = 1.f;
    control.lowState->received = true;
    control.lowState->receivedAt = control::SafetyClock::now();
  };
  setMeasured(config.defaults);
  control.lowState->motorState[13].q = std::numeric_limits<float>::quiet_NaN();
  suite.check(control.safetyCheck(), "healthy active joints accepted; absent joint state ignored");
  suite.check(control.positions23() == config.defaults, "motor slots mapped to policy positions");
  control.setTargets23(config.defaults);
  control.validateCommand();
  suite.check(true, "default position command accepted");
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    const auto &motor = control.lowCmd->motorCmd[g1::kPolicyToMotor[i]];
    suite.near(motor.q, config.defaults[i], "position target maps to existing motor");
    suite.near(motor.Kp, config.kp[i], "trained stiffness maps to existing motor");
    suite.near(motor.Kd, config.kd[i], "trained damping maps to existing motor");
  }
  for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
    if (g1::isPolicyMotor(slot))
      continue;
    control.lowCmd->motorCmd[slot].Kp = 1.f;
    suite.throws([&] { control.validateCommand(); }, "command to an absent motor rejected");
    control.setTargets23(config.defaults);
  }
  control.lowCmd->motorCmd[0].Kp = 501.f;
  suite.throws([&] { control.validateCommand(); }, "excessive gain rejected");
  control.setTargets23(config.defaults);
  control.lowCmd->motorCmd[0].Kp = config.kp[0] + 1.f;
  suite.throws([&] { control.validateCommand(); }, "gain differing from trained contract rejected");
  control.setTargets23(config.defaults);
  control.lowCmd->motorCmd[0].tau = 1.f;
  suite.throws([&] { control.validateCommand(); }, "torque feedforward outside contract rejected");
  control.setTargets23(config.defaults);
  control.lowCmd->motorCmd[0].dq = 1.f;
  suite.throws([&] { control.validateCommand(); },
               "velocity feedforward outside contract rejected");
  control.setTargets23(config.defaults);
  control.lowCmd->motorCmd[0].mode = 0;
  suite.throws([&] { control.validateCommand(); }, "disabled active motor command rejected");

  auto changed = config.defaults;
  changed[0] += config.safety.maxTargetStep + .01f;
  setMeasured(changed);
  control.setTargets23(changed);
  suite.throws([&] { control.validateCommand(); }, "command history rejects discontinuous targets");
  control.setDampingCommand();
  control.validateCommand();
  control.setTargets23(changed);
  control.validateCommand();
  suite.check(true, "damping resets target history to measured posture for re-entry");

  setMeasured(config.defaults);
  control.setDampingCommand();
  control.validateCommand();
  control.setStandTargets23(config.defaults, .5f);
  control.validateCommand();
  suite.check(true, "blended stand command accepted");
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    const auto &motor = control.lowCmd->motorCmd[g1::kPolicyToMotor[i]];
    suite.near(motor.Kp, .5f * (config.kp[i] + config.standKp[i]), "stand stiffness blends");
    suite.near(motor.Kd, .5f * (config.kd[i] + config.standKd[i]), "stand damping blends");
    suite.near(motor.tau, .5f * config.standFf[i], "stand gravity feedforward blends");
  }
  control.setStandTargets23(config.defaults, 1.f);
  control.validateCommand();
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
    suite.near(control.lowCmd->motorCmd[g1::kPolicyToMotor[i]].Kp, config.standKp[i],
               "full stand stiffness reached");
  control.lowCmd->motorCmd[0].Kp = config.standKp[0] + 1.f;
  suite.throws([&] { control.validateCommand(); }, "tampered stand gain rejected");
  control.setStandTargets23(config.defaults, 1.f);
  control.lowCmd->motorCmd[0].tau = config.standFf[0] + .01f;
  suite.throws([&] { control.validateCommand(); }, "tampered stand feedforward rejected");
  control.setStandTargets23(config.defaults, 1.f);
  control.lowCmd->motorCmd[0].Kp = config.kp[0] + 1.f;
  suite.throws([&] { control.validateCommand(); },
               "policy contract gains rejected while stand ramp is active");
  suite.throws([&] { control.setStandTargets23(config.defaults, 1.01f); },
               "stand blend above one rejected");
  suite.throws([&] { control.setStandTargets23(config.defaults, -.01f); },
               "negative stand blend rejected");
  control.setTargets23(config.defaults);
  control.validateCommand();
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
    suite.near(control.lowCmd->motorCmd[g1::kPolicyToMotor[i]].Kp, config.kp[i],
               "policy contract gains restored after stand");

  setMeasured(config.defaults);
  control.lowState->motorState[26].dq = config.safety.maxJointVelocity + .1f;
  suite.check(!control.safetyCheck() && control.safety.latched(),
              "excessive final joint velocity latches");
  control.configure23(config);
  setMeasured(config.defaults);
  control.lowState->motorState[26].q = config.lower[22] - .1f;
  suite.check(!control.safetyCheck(), "out-of-limit measured joint latches");
  control.configure23(config);
  setMeasured(config.defaults);
  control.lowState->imu.gyroscope[2] = std::numeric_limits<float>::quiet_NaN();
  suite.check(!control.safetyCheck(), "invalid IMU angular velocity latches");
  control.lowState->imu.gyroscope[2] = 0.f;
  control.configure23(config);
  setMeasured(config.defaults);
  control.lowState->userValue.lx = 1.1f;
  suite.check(!control.safetyCheck(), "out-of-range joystick latches");
  return suite.result();
}
