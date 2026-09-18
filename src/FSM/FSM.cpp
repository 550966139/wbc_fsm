#include "FSM/FSM.h"
#include <iostream>
#include <thread>

FSM::FSM(CtrlComponents *ctrlComp) : _ctrlComp(ctrlComp) {
  _stateList.passive = std::make_unique<State_Passive>(_ctrlComp);
  _stateList.fixedStand = std::make_unique<State_FixedStand>(_ctrlComp);
  if (_ctrlComp->use23()) {
    _stateList.loco = std::make_unique<State_Policy23>(_ctrlComp);
  } else {
    _stateList.loco = std::make_unique<State_Loco>(_ctrlComp);
    _stateList.amp = std::make_unique<State_AMP>(_ctrlComp);
    _stateList.mjamp = std::make_unique<State_MJAMP>(_ctrlComp);
    _stateList.wbc = std::make_unique<State_WBC>(_ctrlComp);
  }
  initialize();
}

void FSM::initialize() {
  _currentState = _stateList.passive.get();
  _currentState->enter();
}

void FSM::forcePassive() {
  if (_currentState != _stateList.passive.get()) {
    _currentState->exit();
    _currentState = _stateList.passive.get();
    _currentState->enter();
  }
  _ctrlComp->setDampingCommand();
}

void FSM::reportFault() {
  if (_ctrlComp->safety.latched() && _reportedFault != _ctrlComp->safety.reason()) {
    _reportedFault = _ctrlComp->safety.reason();
    std::cerr << "[Safety] Damping latched: " << _reportedFault
              << ". After recovery dwell, press START to acknowledge;"
                 " press START again to stand."
              << std::endl;
  }
}

void FSM::run() {
  const auto controlStarted = control::SafetyClock::now();
  const auto deadline = std::chrono::duration<double>(2.0 * _ctrlComp->dt);
  try {
    if (_ctrlComp->use23() && _lastRunAt != control::SafetyClock::time_point{} &&
        controlStarted - _lastRunAt > deadline)
      _ctrlComp->safety.trip("control loop missed two policy periods");
    _lastRunAt = controlStarted;
    _ctrlComp->ioInter->receive(_ctrlComp->lowState);
    auto &command = _ctrlComp->lowState->userCmd;
    const auto rawCommand = command;
    if (command == _lastUserCommand)
      command = UserCommand::NONE;
    _lastUserCommand = rawCommand;

    // Handle stops before computing any state command, including while faulted.
    if (rawCommand == UserCommand::SELECT) {
      _ctrlComp->exitFlag = true;
      forcePassive();
      _ctrlComp->ioInter->send(_ctrlComp->lowCmd);
    } else {
      if (rawCommand == UserCommand::L2_B)
        _ctrlComp->safety.trip("operator requested damping");
      if (_ctrlComp->ioInter->watchdogFault() && !_ctrlComp->safety.latched())
        _ctrlComp->safety.trip("DDS state/command watchdog or robot mode fault");

      _ctrlComp->safetyCheck();
      if (_ctrlComp->safety.latched()) {
        forcePassive();
        reportFault();
        // Acknowledgement never resumes the prior policy or initiates stand.
        if (_ctrlComp->safety.recover(command == UserCommand::START)) {
          _ctrlComp->ioInter->resetWatchdog();
          if (_ctrlComp->ioInter->watchdogFault()) {
            _ctrlComp->safety.trip("DDS watchdog cannot be reset");
          } else {
            _reportedFault.clear();
            std::cout << "[Safety] Acknowledged; passive damping. Release and press START to stand."
                      << std::endl;
          }
        }
      } else {
        const auto requested = _currentState->checkChange();
        if (requested != _currentState->_stateName) {
          auto *next = getNextState(requested);
          if (!next)
            throw std::runtime_error("unsupported FSM state for this robot");
          _currentState->exit();
          _currentState = next;
          _currentState->enter();
          std::cout << "[FSM] " << _currentState->_stateNameString << std::endl;
        }
        _currentState->run();
        // Inference latency must not publish a command based on expired state.
        if (_ctrlComp->use23() && control::SafetyClock::now() - controlStarted > deadline) {
          _ctrlComp->safety.trip("control computation exceeded two policy periods");
          forcePassive();
        } else if (!_ctrlComp->safetyCheck()) {
          forcePassive();
        } else if (_ctrlComp->ioInter->watchdogFault()) {
          _ctrlComp->safety.trip("DDS watchdog fault during control computation");
          forcePassive();
        } else {
          _ctrlComp->validateCommand();
        }
      }
      reportFault();
      _ctrlComp->ioInter->send(_ctrlComp->lowCmd);
    }
  } catch (const std::exception &error) {
    _ctrlComp->safety.trip(error.what());
    forcePassive();
    reportFault();
    try {
      _ctrlComp->ioInter->send(_ctrlComp->lowCmd);
    } catch (...) {
      // The independent publisher watchdog also rejects a stalled command loop.
      _ctrlComp->exitFlag = true;
    }
  } catch (...) {
    _ctrlComp->safety.trip("unknown controller failure");
    forcePassive();
    reportFault();
    try {
      _ctrlComp->ioInter->send(_ctrlComp->lowCmd);
    } catch (...) {
    }
    _ctrlComp->exitFlag = true;
  }
  // Robot timing must not follow NTP / wall-clock adjustments.
  std::this_thread::sleep_until(controlStarted +
                                std::chrono::duration_cast<control::SafetyClock::duration>(
                                    std::chrono::duration<double>(_ctrlComp->dt)));
}

FSMState *FSM::getNextState(FSMStateName stateName) {
  switch (stateName) {
  case FSMStateName::PASSIVE:
    return _stateList.passive.get();
  case FSMStateName::FIXEDSTAND:
    return _stateList.fixedStand.get();
  case FSMStateName::LOCO:
    return _stateList.loco.get();
  case FSMStateName::WBC:
    return _stateList.wbc.get();
  case FSMStateName::AMP:
    return _stateList.amp.get();
  case FSMStateName::MJAMP:
    return _stateList.mjamp.get();
  default:
    return nullptr;
  }
}
