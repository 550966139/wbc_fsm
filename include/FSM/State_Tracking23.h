#pragma once

#include "FSM/FSMState.h"
#include "common/tracking23.h"
#include <memory>
#include <onnxruntime_cxx_api.h>

// 23DoF whole-body motion tracking (dance). A separate state keeps the
// velocity-policy path untouched; it exists only when a tracking config was
// supplied and its motion CSV loaded (see FSM.cpp).
class State_Tracking23 final : public FSMState {
public:
  explicit State_Tracking23(CtrlComponents *components);
  void enter() override;
  void run() override;
  void exit() override;
  FSMStateName checkChange() override;

private:
  g1::Joints23 infer(g1::ObservationTracking23 &observation);
  g1::Quat23 currentRobotAnchor() const;
  // Operator-abort handoff: one-shot gate evaluation (no settle window —
  // a mid-dance pose cannot settle into the policy entry gate anyway).
  FSMStateName handoff(const char *reason);
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "g1-tracking23"};
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  std::string inputName_, outputName_;
  g1::Joints23 lastAction_{};
  std::size_t cursor_ = 0;
  // Entry target ramp-in: commands blend from the measured pose at enter()
  // to the policy targets over kTrackingEntryRampCycles cycles, removing the
  // controller-handoff command jump (the visible first-second wobble).
  g1::Joints23 entryPose_{};
  std::size_t rampCycle_ = 0;
  bool finished_ = false;
  // Motion-end settle-hold: cursor frozen on the outro frame, tracking keeps
  // running, and checkChange re-evaluates the policy entry gate every cycle
  // until it passes or handoffSettleTimeout expires.
  bool settling_ = false, settleLogged_ = false;
  float settleElapsed_ = 0.f;
  g1::Quat23 entryAlign_{1.f, 0.f, 0.f, 0.f};
};
