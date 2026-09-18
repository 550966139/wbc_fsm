#pragma once

#include "FSM/FSMState.h"
#include "common/policy23.h"
#include <memory>
#include <onnxruntime_cxx_api.h>

// A separate adapter prevents the legacy 29DoF observation/action path from
// being reached on the 23DoF robot.
class State_Policy23 final : public FSMState {
public:
  explicit State_Policy23(CtrlComponents *components);
  void enter() override;
  void run() override;
  void exit() override;
  FSMStateName checkChange() override;

private:
  g1::Joints23 infer(g1::Observation23 &observation);
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "g1-policy23"};
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  std::string inputName_, outputName_;
  g1::Joints23 lastAction_{};
  float phase_ = 0.f;
};
