#ifndef FSM_H
#define FSM_H

#include "FSM/FSMState.h"
#include "FSM/State_Amp.h"
#include "FSM/State_FixedStand.h"
#include "FSM/State_Loco.h"
#include "FSM/State_MJAmp.h"
#include "FSM/State_Passive.h"
#include "FSM/State_Policy23.h"
#include "FSM/State_WBC.h"
#include <memory>

struct FSMStateList {
  std::unique_ptr<FSMState> passive, fixedStand, loco, wbc, amp, mjamp;
};

class FSM {
public:
  explicit FSM(CtrlComponents *ctrlComp);
  ~FSM() = default;
  void initialize();
  void run();
  FSMStateName stateName() const { return _currentState->_stateName; }

private:
  FSMState *getNextState(FSMStateName stateName);
  void forcePassive();
  void reportFault();
  CtrlComponents *_ctrlComp;
  FSMState *_currentState = nullptr;
  FSMStateList _stateList;
  UserCommand _lastUserCommand = UserCommand::NONE;
  std::string _reportedFault;
  control::SafetyClock::time_point _lastRunAt{};
};
#endif
