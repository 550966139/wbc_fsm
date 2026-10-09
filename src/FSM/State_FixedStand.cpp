
#include "FSM/State_FixedStand.h"
#include <algorithm>
#include <fstream>
#include <iostream>

State_FixedStand::State_FixedStand(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::FIXEDSTAND, "fixed stand") {}

void State_FixedStand::enter() {
  if (_ctrlComp->use23()) {
    const auto &config = _ctrlComp->config23.value();
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      const auto slot = g1::kPolicyToMotor[i];
      _startPos[slot] = _lowState->motorState[slot].q;
      _targetPos[slot] = config.defaults[i];
    }
    _duration = config.standDuration;
    _phase = 0.f;
    _fixedstand_complete_flag = false;
    std::cout << "23DoF stand interpolation; when complete and stable, press R2+A for locomotion."
              << std::endl;
    return;
  }
  for (int i = 0; i < NUM_DOF; i++) {
    _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
    _startPos[i] = _lowState->motorState[i].q;
  }
  _phase = 0;
  _duration = 2.0;
  _fixedstand_complete_flag = false;
  std::string config_path = std::string(PROJECT_ROOT_DIR) + "/config/fixedpose.json";
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    std::cerr << "[ERROR] Failed to open config file: " << config_path << std::endl;
    throw std::runtime_error("Cannot open config file");
  }
  try {
    json config = json::parse(config_file);
    _duration = config["duration"].get<float>();
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to parse config file: " << e.what() << std::endl;
    throw;
  }
  config_file.close();
  std::cout
      << "Please make the robot stand first, stabilize it, then press **R2+A** to enter Locomode"
      << std::endl;
}

void State_FixedStand::run() {

  _phase += _ctrlComp->dt / _duration;
  if (_phase >= 1) {
    _fixedstand_complete_flag = true;
  }
  _phase = _fixedstand_complete_flag ? 1 : _phase;

  if (_ctrlComp->use23()) {
    const auto &config = _ctrlComp->config23.value();
    const float blend = _phase * _phase * (3.f - 2.f * _phase);
    g1::Joints23 target{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      const auto slot = g1::kPolicyToMotor[i];
      target[i] = (1.f - blend) * _startPos[slot] + blend * _targetPos[slot];
    }
    _ctrlComp->setStandTargets23(g1::clampTargets23(target, config.lower, config.upper), blend);
    return;
  }

  for (int j = 0; j < NUM_DOF; j++) {
    _lowCmd->motorCmd[j].tau = 0;
    _lowCmd->motorCmd[j].q = (1 - _phase) * _startPos[j] + _phase * _targetPos[j];
    _lowCmd->motorCmd[j].Kp = Kps[j];
    _lowCmd->motorCmd[j].Kd = Kds[j];
  }
}

void State_FixedStand::exit() {
  _phase = 0;
  _fixedstand_complete_flag = false;
}

FSMStateName State_FixedStand::checkChange() {
  if (_lowState->userCmd == UserCommand::L2_B) {
    return FSMStateName::PASSIVE;
  } else if (_lowState->userCmd == UserCommand::R2_A) {
    if (_ctrlComp->use23()) {
      if (!_fixedstand_complete_flag) {
        std::cout << "[FSM] R2+A held: stand interpolation not complete yet" << std::endl;
        return FSMStateName::FIXEDSTAND;
      }
      const auto gate = g1::entryGate23(_ctrlComp->positions23(), _ctrlComp->velocities23(),
                                        _ctrlComp->config23->defaults);
      if (!gate.passes(g1::kPolicyEntryMaxJointError, g1::kPolicyEntryMaxJointVel)) {
        std::cout << "[FSM] R2+A rejected: worst joint " << gate.worstJoint << " error "
                  << gate.worstError << " rad, max |dq| " << gate.worstVelocity << std::endl;
        return FSMStateName::FIXEDSTAND;
      }
      return FSMStateName::LOCO;
    }
    // return FSMStateName::AMP;
    return FSMStateName::MJAMP;
  } else if (_lowState->userCmd == UserCommand::R2_B || _lowState->userCmd == UserCommand::R2_Y) {
    if (_ctrlComp->use23())
      // Dance entry (R2+B motion 1, R2+Y motion 2): shared availability,
      // stand-readiness and per-motion frame-0 entry gate.
      return _ctrlComp->requestTrackingMotion(_lowState->userCmd, _fixedstand_complete_flag,
                                              FSMStateName::FIXEDSTAND);
    return FSMStateName::FIXEDSTAND;
  } else if (_lowState->userCmd == UserCommand::SELECT) {
    throw std::runtime_error("exit..");
    return FSMStateName::PASSIVE;
  } else {
    return FSMStateName::FIXEDSTAND;
  }
}
