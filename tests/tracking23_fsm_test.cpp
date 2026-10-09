#include "FSM/FSM.h"
#include "test_support.h"
#include <thread>

// Offline FSM integration for the 23DoF tracking state, driven by
// tests/onnx_preflight_test.py with a synthetic 124->23 ONNX graph and a small
// motion CSV in a temp directory. argv[1]: velocity config, argv[2]: tracking
// config (motion_end_target "policy23"; its safety guard values differ from
// the velocity config so the per-state swap is observable), argv[3]: tracking
// config with motion_end_target "stand", argv[4]: two-motion "motions[]"
// config (R2+B -> motion one, R2+Y -> motion two with its own guards and a
// "stand" end target).
class FakeIO final : public IOInterface {
public:
  LowlevelState state;
  LowlevelCmd sent;
  bool fault = false;
  int resets = 0;
  void receive(LowlevelState *out) override { *out = state; }
  void send(const LowlevelCmd *command) override { sent = *command; }
  void sendRecv(const LowlevelCmd *command, LowlevelState *out) override {
    receive(out);
    send(command);
  }
  bool watchdogFault() const override { return fault; }
  void resetWatchdog() override {
    fault = false;
    ++resets;
  }
};

namespace {
struct Rig {
  explicit Rig(const g1::Config23 &cfg) : io(new FakeIO()), components(io) {
    components.configure23(cfg);
    io->state.received = true;
    io->state.imu.quaternion[0] = 1.f;
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      io->state.motorState[g1::kPolicyToMotor[i]].q = cfg.defaults[i];
  }
  void step(FSM &fsm, UserCommand command = UserCommand::NONE) {
    io->state.userCmd = command;
    io->state.receivedAt = control::SafetyClock::now();
    fsm.run();
  }
  void stand(FSM &fsm) {
    step(fsm, UserCommand::START);
    for (int i = 0; i < 155; ++i)
      step(fsm);
  }
  void setJoint(std::size_t policyIndex, float q) {
    io->state.motorState[g1::kPolicyToMotor[policyIndex]].q = q;
  }
  FakeIO *io;
  CtrlComponents components;
};
} // namespace

int main(int argc, char **argv) {
  TestSuite suite;
#define CHECK(expression) suite.check((expression), #expression)
  if (argc != 5)
    return 2;
  auto cfg = g1::Config23::load(argv[1]);
  auto tcfg = g1::ConfigTracking23::load(argv[2]);
  auto tcfgStand = g1::ConfigTracking23::load(argv[3]);
  auto tcfgDual = g1::ConfigTracking23::load(argv[4]);
  CHECK(tcfg.motions[0].motionEndTarget == g1::MotionEndTarget::Policy23);
  CHECK(tcfgStand.motions[0].motionEndTarget == g1::MotionEndTarget::Stand);
  CHECK(tcfg.motions[0].maxTargetStep != cfg.safety.maxTargetStep ||
        tcfg.motions[0].maxTrackingError != cfg.safety.maxTrackingError);
  CHECK(tcfgDual.motions.size() == 2);
  CHECK(tcfgDual.motions[1].maxTargetStep != tcfgDual.motions[0].maxTargetStep ||
        tcfgDual.motions[1].maxTrackingError != tcfgDual.motions[0].maxTrackingError);
  CHECK(tcfgDual.motions[1].motionEndTarget == g1::MotionEndTarget::Stand);
  Rig rig(cfg);
  rig.components.configureTracking23(tcfg);
  FSM fsm(&rig.components);
  auto damping = [&]() {
    CHECK(fsm.stateName() == FSMStateName::PASSIVE);
    for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
      const auto &motor = rig.io->sent.motorCmd[slot];
      CHECK(motor.Kp == 0.f && motor.tau == 0.f);
      CHECK(motor.Kd == (g1::isPolicyMotor(slot) ? 3.f : 0.f));
    }
  };

  rig.step(fsm);
  damping();
  rig.stand(fsm);
  CHECK(fsm.stateName() == FSMStateName::FIXEDSTAND);
  rig.step(fsm, UserCommand::R2_B);
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  CHECK(!rig.components.safety.latched());
  suite.near(rig.components.safety.limits().maxTargetStep, tcfg.motions[0].maxTargetStep,
             "tracking swaps in its own target-step guard");
  suite.near(rig.components.safety.limits().maxTrackingError, tcfg.motions[0].maxTrackingError,
             "tracking swaps in its own tracking-error guard");
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    suite.near(rig.io->sent.motorCmd[g1::kPolicyToMotor[i]].Kp, cfg.kp[i],
               "tracking commands use the shared policy stiffness");
    suite.near(rig.io->sent.motorCmd[g1::kPolicyToMotor[i]].Kd, cfg.kd[i],
               "tracking commands use the shared policy damping");
    suite.near(rig.io->sent.motorCmd[g1::kPolicyToMotor[i]].q, tcfg.defaults[i],
               "constant-zero synthetic policy holds the default posture");
  }

  // The fixture motion has 8 frames; the transition cycle consumed frame 0.
  // Motion end with entry-gate jitter must NOT fail outright: the settle-hold
  // freezes the outro frame and re-checks the gate every cycle.
  for (int i = 0; i < 6; ++i)
    rig.step(fsm);
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  rig.setJoint(3, cfg.defaults[3] + .2f);                    // inside the 1.4 rad test guard
  rig.io->state.motorState[g1::kPolicyToMotor[3]].dq = 1.9f; // post-dance jitter
  rig.step(fsm); // frame 7 consumed; settle-hold starts during this cycle
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  rig.step(fsm); // gate fails under jitter: still settling, not fixed stand
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  CHECK(!rig.components.safety.latched());
  rig.setJoint(3, cfg.defaults[3]); // jitter settles
  rig.io->state.motorState[g1::kPolicyToMotor[3]].dq = 0.f;
  rig.step(fsm); // gate passes inside the settle window: handoff
  CHECK(fsm.stateName() == FSMStateName::LOCO);
  CHECK(!rig.components.safety.latched());
  suite.near(rig.components.safety.limits().maxTargetStep, cfg.safety.maxTargetStep,
             "target-step guard restored on tracking exit");
  suite.near(rig.components.safety.limits().maxTrackingError, cfg.safety.maxTrackingError,
             "tracking-error guard restored on tracking exit");
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    suite.near(rig.io->sent.motorCmd[g1::kPolicyToMotor[i]].Kp, cfg.kp[i],
               "policy form after the handoff uses policy stiffness");
    suite.near(rig.io->sent.motorCmd[g1::kPolicyToMotor[i]].q, cfg.defaults[i],
               "zero command after the handoff holds the default posture");
  }

  // Walk -> dance: R2+B inside the velocity policy, accepted at frame 0 pose.
  rig.step(fsm);
  rig.step(fsm, UserCommand::R2_B);
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  suite.near(rig.components.safety.limits().maxTargetStep, tcfg.motions[0].maxTargetStep,
             "tracking guards swapped back in after walk->dance entry");

  // Operator abort now also hands off to the velocity policy.
  rig.step(fsm);
  rig.step(fsm, UserCommand::R2_B);
  CHECK(fsm.stateName() == FSMStateName::LOCO);
  CHECK(!rig.components.safety.latched());
  suite.near(rig.components.safety.limits().maxTargetStep, cfg.safety.maxTargetStep,
             "target-step guard restored after operator abort");

  // Walk -> dance rejected while a joint is far from motion frame 0.
  rig.step(fsm); // fresh command edge after the abort's R2+B
  rig.setJoint(3, cfg.defaults[3] + .3f);
  rig.step(fsm, UserCommand::R2_B);
  CHECK(fsm.stateName() == FSMStateName::LOCO);
  CHECK(!rig.components.safety.latched());
  rig.setJoint(3, cfg.defaults[3]);

  // Operator abort DURING the settle-hold gets no window: the one-shot gate is
  // evaluated immediately and fails under held jitter -> fixed stand.
  rig.step(fsm);
  rig.step(fsm, UserCommand::R2_B);
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  for (int i = 0; i < 6; ++i)
    rig.step(fsm);
  rig.setJoint(3, cfg.defaults[3] + .2f);
  rig.io->state.motorState[g1::kPolicyToMotor[3]].dq = 1.9f;
  rig.step(fsm); // settle-hold starts
  rig.step(fsm); // settling under jitter
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  rig.step(fsm, UserCommand::R2_B); // abort: one-shot gate fails, no settle window
  CHECK(fsm.stateName() == FSMStateName::FIXEDSTAND);
  CHECK(!rig.components.safety.latched());
  rig.setJoint(3, cfg.defaults[3]);
  rig.io->state.motorState[g1::kPolicyToMotor[3]].dq = 0.f;

  // Settle timeout: jitter held past handoff_settle_timeout (0.2 s in the
  // fixture config) falls back to fixed stand with the reason logged.
  rig.stand(fsm);
  rig.step(fsm, UserCommand::R2_B);
  CHECK(fsm.stateName() == FSMStateName::TRACKING);
  for (int i = 0; i < 6; ++i)
    rig.step(fsm);
  rig.setJoint(3, cfg.defaults[3] + .2f);
  rig.io->state.motorState[g1::kPolicyToMotor[3]].dq = 1.9f;
  rig.step(fsm); // settle-hold starts
  for (int i = 0; i < 15; ++i)
    rig.step(fsm); // 0.3 s of held jitter > 0.2 s timeout
  CHECK(fsm.stateName() == FSMStateName::FIXEDSTAND);
  CHECK(!rig.components.safety.latched());
  rig.setJoint(3, cfg.defaults[3]);
  rig.io->state.motorState[g1::kPolicyToMotor[3]].dq = 0.f;

  rig.step(fsm, UserCommand::L2_B);
  damping();
  CHECK(rig.components.safety.latched());

  // Fallback switch: motion_end_target "stand" keeps the phase-1 behaviour —
  // motion end and operator abort return to fixed stand.
  Rig standRig(cfg);
  standRig.components.configureTracking23(tcfgStand);
  FSM standFsm(&standRig.components);
  standRig.step(standFsm);
  standRig.stand(standFsm);
  standRig.step(standFsm, UserCommand::R2_B);
  CHECK(standFsm.stateName() == FSMStateName::TRACKING);
  for (int i = 0; i < 7; ++i)
    standRig.step(standFsm);
  standRig.step(standFsm);
  CHECK(standFsm.stateName() == FSMStateName::FIXEDSTAND);
  CHECK(!standRig.components.safety.latched());
  // The fresh stand has not interpolated yet: early entry is rejected.
  standRig.step(standFsm, UserCommand::R2_B);
  CHECK(standFsm.stateName() == FSMStateName::FIXEDSTAND);
  // Abort in stand mode returns to fixed stand as well.
  standRig.stand(standFsm);
  standRig.step(standFsm, UserCommand::R2_B);
  CHECK(standFsm.stateName() == FSMStateName::TRACKING);
  standRig.step(standFsm);
  standRig.step(standFsm, UserCommand::R2_B);
  CHECK(standFsm.stateName() == FSMStateName::FIXEDSTAND);
  CHECK(!standRig.components.safety.latched());

  // Single-motion config: R2+Y (dance #2) is inert from stand and from walk.
  standRig.step(standFsm, UserCommand::R2_Y);
  CHECK(standFsm.stateName() == FSMStateName::FIXEDSTAND);
  CHECK(!standRig.components.safety.latched());

  // Dual-motion library: R2+Y enters motion two with ITS guards swapped in;
  // the per-motion "stand" end target returns to fixed stand on completion.
  Rig dualRig(cfg);
  dualRig.components.configureTracking23(tcfgDual);
  FSM dualFsm(&dualRig.components);
  dualRig.step(dualFsm);
  dualRig.stand(dualFsm);
  dualRig.step(dualFsm, UserCommand::R2_Y);
  CHECK(dualFsm.stateName() == FSMStateName::TRACKING);
  CHECK(!dualRig.components.safety.latched());
  suite.near(dualRig.components.safety.limits().maxTargetStep, tcfgDual.motions[1].maxTargetStep,
             "motion two swaps in its own target-step guard");
  suite.near(dualRig.components.safety.limits().maxTrackingError,
             tcfgDual.motions[1].maxTrackingError,
             "motion two swaps in its own tracking-error guard");
  // R2+Y mid-dance does not switch references: abort with R2+B first.
  dualRig.step(dualFsm);
  dualRig.step(dualFsm, UserCommand::R2_Y);
  CHECK(dualFsm.stateName() == FSMStateName::TRACKING);
  CHECK(!dualRig.components.safety.latched());
  for (int i = 0; i < 7; ++i)
    dualRig.step(dualFsm);
  dualRig.step(dualFsm);
  CHECK(dualFsm.stateName() == FSMStateName::FIXEDSTAND);
  CHECK(!dualRig.components.safety.latched());
  suite.near(dualRig.components.safety.limits().maxTargetStep, cfg.safety.maxTargetStep,
             "guards restored after motion two ends");
  // R2+B still selects motion one in the dual config.
  dualRig.stand(dualFsm);
  dualRig.step(dualFsm, UserCommand::R2_B);
  CHECK(dualFsm.stateName() == FSMStateName::TRACKING);
  suite.near(dualRig.components.safety.limits().maxTargetStep, tcfgDual.motions[0].maxTargetStep,
             "R2+B keeps selecting motion one's guards");
  dualRig.step(dualFsm);
  dualRig.step(dualFsm, UserCommand::R2_B);
  // Motion one ends with the policy23 handoff, so the abort goes to LOCO
  // (dance-two's "stand" end target above covered the FIXEDSTAND path).
  CHECK(dualFsm.stateName() == FSMStateName::LOCO);
  CHECK(!dualRig.components.safety.latched());
  suite.near(dualRig.components.safety.limits().maxTargetStep, cfg.safety.maxTargetStep,
             "guards restored after motion one abort");
  return suite.result();
}
