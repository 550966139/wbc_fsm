#include "FSM/FSM.h"
#include "test_support.h"
#include <thread>

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

int main(int argc, char **argv) {
  TestSuite suite;
#define CHECK(expression) suite.check((expression), #expression)
  if (argc != 2)
    return 2;
  auto cfg = g1::Config23::load(argv[1]);
  auto *io = new FakeIO();
  CtrlComponents components(io);
  components.configure23(cfg);
  io->state.received = true;
  io->state.imu.quaternion[0] = 1.f;
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
    io->state.motorState[g1::kPolicyToMotor[i]].q = cfg.defaults[i];
  FSM fsm(&components);
  auto step = [&](UserCommand command = UserCommand::NONE, bool fresh = true) {
    io->state.userCmd = command;
    if (fresh)
      io->state.receivedAt = control::SafetyClock::now();
    fsm.run();
  };
  auto damping = [&]() {
    CHECK(fsm.stateName() == FSMStateName::PASSIVE);
    for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
      const auto &motor = io->sent.motorCmd[slot];
      CHECK(motor.Kp == 0.f && motor.tau == 0.f);
      CHECK(motor.Kd == (g1::isPolicyMotor(slot) ? 3.f : 0.f));
    }
  };

  step();
  damping();
  step(UserCommand::R2_A);
  damping();
  step(UserCommand::START);
  CHECK(fsm.stateName() == FSMStateName::FIXEDSTAND);
  step(UserCommand::R2_A);
  CHECK(fsm.stateName() == FSMStateName::FIXEDSTAND); // early policy entry rejected
  for (int i = 0; i < 155; ++i)
    step();
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    suite.near(io->sent.motorCmd[g1::kPolicyToMotor[i]].Kp, cfg.standKp[i],
               "stand interpolation ends at stand stiffness");
    suite.near(io->sent.motorCmd[g1::kPolicyToMotor[i]].tau, cfg.standFf[i],
               "stand interpolation ends at full gravity feedforward");
  }
  step(UserCommand::R2_A);
  CHECK(fsm.stateName() == FSMStateName::LOCO);
  CHECK(!components.safety.latched());

  // Stops replace the action in the same send, then require distinct acknowledgements.
  step(UserCommand::L2_B);
  damping();
  CHECK(components.safety.latched());
  step(UserCommand::START);
  damping();
  for (int i = 0; i < 28; ++i)
    step();
  step(UserCommand::START);
  CHECK(!components.safety.latched());
  damping();
  step(UserCommand::START); // held START must not re-arm
  damping();
  step();
  step(UserCommand::START);
  CHECK(fsm.stateName() == FSMStateName::FIXEDSTAND);

  io->fault = true;
  step();
  CHECK(components.safety.latched());
  damping();
  for (int i = 0; i < 28; ++i)
    step();
  step(UserCommand::START);
  CHECK(!components.safety.latched() && !io->fault);
  CHECK(io->resets >= 2);
  damping();

  io->state.receivedAt = control::SafetyClock::now() - std::chrono::seconds(1);
  step(UserCommand::NONE, false);
  CHECK(components.safety.latched());
  damping();
  step(UserCommand::SELECT, false);
  CHECK(components.exitFlag);
  damping();
  return suite.result();
}
