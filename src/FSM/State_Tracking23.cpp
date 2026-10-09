#include "FSM/State_Tracking23.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {
void requireShape(const Ort::ConstTensorTypeAndShapeInfo &tensor, int64_t width,
                  const char *label) {
  const auto shape = tensor.GetShape();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 2 ||
      (shape[0] != 1 && shape[0] != -1) || shape[1] != width)
    throw std::runtime_error(std::string("23DoF tracking policy requires float ") + label + " [1," +
                             std::to_string(width) + "]");
}
} // namespace

State_Tracking23::State_Tracking23(CtrlComponents *components)
    : FSMState(components, FSMStateName::TRACKING, "23DoF motion tracking") {
  if (!components->configTracking23 || components->trackingMotions.empty())
    throw std::runtime_error("23DoF tracking requires a tracking config and loaded motions");
  const auto &config = components->configTracking23.value();
  options_.SetIntraOpNumThreads(1);
  options_.SetInterOpNumThreads(1);
  options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::AllocatorWithDefaultOptions allocator;
  for (std::size_t slot = 0; slot < config.motions.size(); ++slot) {
    const auto &entry = config.motions[slot];
    if (!std::filesystem::is_regular_file(entry.model))
      throw std::runtime_error("23DoF tracking policy missing for motion \"" + entry.name +
                               "\": " + entry.model.string() +
                               "; provide a matching trained 124-input / 23-output model");
    TrackingSession s;
    s.session = std::make_unique<Ort::Session>(env_, entry.model.c_str(), options_);
    if (s.session->GetInputCount() != 1 || s.session->GetOutputCount() != 1)
      throw std::runtime_error("23DoF tracking policy must have one observation input and one "
                               "action output (motion \"" +
                               entry.name + "\")");
    requireShape(s.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo(), 124, "input");
    requireShape(s.session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo(), 23, "output");
    s.inputName = s.session->GetInputNameAllocated(0, allocator).get();
    s.outputName = s.session->GetOutputNameAllocated(0, allocator).get();
    sessions_.push_back(std::move(s));
  }

  // Validate executable output on motion frame 0 before the DDS publisher can
  // be started; the anchor terms use identity robot orientation.
  for (std::size_t slot = 0; slot < config.motions.size(); ++slot) {
    const auto *motion = components->trackingMotions[slot].get();
    const float *quat0 = motion->anchorQuat(0);
    auto observation = g1::observationTracking23(
        motion->jointPos(0), motion->jointVel(0), {1.f, 0.f, 0.f, 0.f},
        {quat0[0], quat0[1], quat0[2], quat0[3]}, {}, config.defaults, {}, config.defaults, {});
    (void)infer(observation, slot);
  }
}

g1::Joints23 State_Tracking23::infer(g1::ObservationTracking23 &observation,
                                     std::size_t motionIndex) {
  const auto &s = sessions_[motionIndex];
  const std::array<int64_t, 2> shape{1, 124};
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(memory, observation.data(), observation.size(),
                                               shape.data(), shape.size());
  const char *inputs[] = {s.inputName.c_str()};
  const char *outputs[] = {s.outputName.c_str()};
  auto result = s.session->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
  if (result.size() != 1 || !result[0].IsTensor())
    throw std::runtime_error("23DoF tracking policy produced a non-tensor action");
  const auto info = result[0].GetTensorTypeAndShapeInfo();
  if (info.GetShape() != std::vector<int64_t>{1, 23} ||
      info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("23DoF tracking policy output shape or type changed during inference");
  g1::Joints23 actions{};
  std::copy_n(result[0].GetTensorData<float>(), actions.size(), actions.begin());
  for (float value : actions)
    if (!std::isfinite(value))
      throw std::runtime_error("23DoF tracking policy produced non-finite action");
  return actions;
}

void State_Tracking23::enter() {
  const auto &config = _ctrlComp->configTracking23.value();
  motionIndex_ = std::min(_ctrlComp->nextTrackingMotion, config.motions.size() - 1);
  const auto &entry = config.motions[motionIndex_];
  cursor_ = 0;
  finished_ = false;
  settling_ = false;
  settleLogged_ = false;
  settleElapsed_ = 0.f;
  lastAction_.fill(0.f);
  logCycle_ = 50;
  entryPose_ = _ctrlComp->positions23();
  rampCycle_ = 0;
  // Entry yaw alignment: the robot world frame is gravity-aligned with
  // arbitrary yaw, the reference is MuJoCo z-up; rotate the whole reference by
  // the yaw offset measured at this entry, applied per frame in run().
  const auto robotAnchor = currentRobotAnchor();
  const float *ref0 = _ctrlComp->trackingMotions[motionIndex_]->anchorQuat(0);
  entryAlign_ = g1::quatAboutZ23(g1::yawOf23(robotAnchor) -
                                 g1::yawOf23({ref0[0], ref0[1], ref0[2], ref0[3]}));
  _ctrlComp->safety.overrideCommandGuards(entry.maxTrackingError, entry.maxTargetStep);
  std::cout << "[FSM] tracking \"" << entry.name << "\" "
            << _ctrlComp->trackingMotions[motionIndex_]->frameCount() << " motion frames"
            << (entry.loopMotion ? " (looping)" : "") << "; R2+B aborts to "
            << (entry.motionEndTarget == g1::MotionEndTarget::Stand ? "fixed stand"
                                                                    : "the velocity policy")
            << std::endl;
}

g1::Quat23 State_Tracking23::currentRobotAnchor() const {
  const auto &imu = _lowState->imu.quaternion;
  const float waistYaw = _lowState->motorState[g1::kPolicyToMotor[g1::kWaistYawJoint]].q;
  return g1::robotAnchorQuat23({imu[0], imu[1], imu[2], imu[3]}, waistYaw);
}

void State_Tracking23::run() {
  const auto &config = _ctrlComp->configTracking23.value();
  const auto &entry = config.motions[motionIndex_];
  const auto &motion = *_ctrlComp->trackingMotions[motionIndex_];
  const float *refQuat = motion.anchorQuat(cursor_);
  const auto refAligned =
      g1::quatMul23(entryAlign_, {refQuat[0], refQuat[1], refQuat[2], refQuat[3]});
  const auto &gyro = _lowState->imu.gyroscope;
  auto observation = g1::observationTracking23(
      motion.jointPos(cursor_), motion.jointVel(cursor_), currentRobotAnchor(), refAligned,
      {gyro[0], gyro[1], gyro[2]}, _ctrlComp->positions23(), _ctrlComp->velocities23(),
      config.defaults, lastAction_);
  const auto actions = infer(observation, motionIndex_);
  auto target = g1::clampTargets23(g1::actionTargets23(actions, config.scale, config.defaults),
                                   config.lower, config.upper);
  if (rampCycle_ < g1::kTrackingEntryRampCycles) {
    const float t =
        static_cast<float>(++rampCycle_) / static_cast<float>(g1::kTrackingEntryRampCycles);
    const float w = t * t * (3.f - 2.f * t);
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      target[i] = (1.f - w) * entryPose_[i] + w * target[i];
  }
  _ctrlComp->setTargets23(target);
  lastAction_ = actions;
  if (++logCycle_ >= 50) {
    // 1 Hz joint snapshot: frame cursor, worst reference error and all 23
    // measured positions — makes every hardware session analyzable from the
    // controller log alone (hardware gap analysis previously impossible).
    logCycle_ = 0;
    const auto &measured = _ctrlComp->positions23();
    const auto &ref = motion.jointPos(cursor_);
    std::size_t worst = 0;
    float worstErr = 0.f;
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      const float e = std::fabs(measured[i] - ref[i]);
      if (e > worstErr) {
        worstErr = e;
        worst = i;
      }
    }
    std::cout << "[track:" << entry.name << "] f=" << cursor_ << "/" << motion.frameCount()
              << " worst j" << worst << " e" << worstErr << " q=";
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      std::cout << (i ? " " : "") << measured[i];
    std::cout << "\n";
  }
  if (settling_) {
    // Cursor frozen on the outro stance frame; the policy keeps balancing at
    // the stance pose while checkChange re-evaluates the entry gate.
    settleElapsed_ += config.dt;
  } else if (++cursor_ >= motion.frameCount()) {
    if (entry.loopMotion) {
      cursor_ = 0;
    } else {
      finished_ = true;
      settling_ = true;
      settleElapsed_ = 0.f;
      settleLogged_ = false;
      cursor_ = motion.frameCount() - 1; // hold the outro stance frame
    }
  }
}

void State_Tracking23::exit() {
  _ctrlComp->safety.restoreCommandGuards();
  lastAction_.fill(0.f);
  rampCycle_ = 0;
  cursor_ = 0;
  finished_ = false;
  settling_ = false;
  settleLogged_ = false;
  settleElapsed_ = 0.f;
  entryAlign_ = {1.f, 0.f, 0.f, 0.f};
}

FSMStateName State_Tracking23::checkChange() {
  if (_lowState->userCmd == UserCommand::R2_B)
    return handoff("R2+B: aborting tracking"); // one-shot gate, no settle window
  if (_lowState->userCmd == UserCommand::R2_Y) {
    // Second-motion select mid-dance: switching references mid-flight is not
    // a supported transition; abort with R2+B first.
    std::cout << "[FSM] R2+Y ignored while tracking motion \""
              << _ctrlComp->configTracking23->motions[motionIndex_].name
              << "\": abort with R2+B before selecting another motion" << std::endl;
    return FSMStateName::TRACKING;
  }
  if (!finished_)
    return FSMStateName::TRACKING;
  const auto &entry = _ctrlComp->configTracking23.value().motions[motionIndex_];
  if (entry.motionEndTarget == g1::MotionEndTarget::Stand) {
    std::cout << "[FSM] tracking motion complete, returning to fixed stand" << std::endl;
    return FSMStateName::FIXEDSTAND;
  }
  // Motion-end settle-hold: the tracking policy is still running on the frozen
  // outro frame; hand off the first cycle the policy entry gate passes, or fall
  // back to fixed stand once the settle timeout expires.
  const auto gate = g1::entryGate23(_ctrlComp->positions23(), _ctrlComp->velocities23(),
                                    _ctrlComp->config23->defaults);
  if (gate.passes(g1::kPolicyEntryMaxJointError, g1::kPolicyEntryMaxJointVel)) {
    std::cout << "[FSM] tracking motion complete, handing off to the velocity policy" << std::endl;
    return FSMStateName::LOCO;
  }
  if (settleElapsed_ >= entry.handoffSettleTimeout) {
    std::cout << "[FSM] policy entry gate still failing after " << settleElapsed_
              << " s: worst joint " << gate.worstJoint << " error " << gate.worstError
              << " rad, max |dq| " << gate.worstVelocity << "; returning to fixed stand"
              << std::endl;
    return FSMStateName::FIXEDSTAND;
  }
  if (!settleLogged_) {
    settleLogged_ = true;
    std::cout << "[FSM] tracking motion complete; holding the outro frame, waiting for the "
                 "policy entry gate (worst joint "
              << gate.worstJoint << " error " << gate.worstError << " rad, max |dq| "
              << gate.worstVelocity << ")" << std::endl;
  }
  return FSMStateName::TRACKING;
}

FSMStateName State_Tracking23::handoff(const char *reason) {
  const auto &entry = _ctrlComp->configTracking23->motions[motionIndex_];
  if (entry.motionEndTarget == g1::MotionEndTarget::Stand) {
    std::cout << "[FSM] " << reason << ", returning to fixed stand" << std::endl;
    return FSMStateName::FIXEDSTAND;
  }
  // Default chain: hand off to the velocity policy at zero command (verified
  // active balance, unlike FixedStand which needs external support). The
  // dance outro ends at the stance pose, but the policy entry gate is still
  // enforced exactly like the FixedStand R2+A gate; on rejection fall back to
  // fixed stand with the reason logged.
  const auto gate = g1::entryGate23(_ctrlComp->positions23(), _ctrlComp->velocities23(),
                                    _ctrlComp->config23->defaults);
  if (!gate.passes(g1::kPolicyEntryMaxJointError, g1::kPolicyEntryMaxJointVel)) {
    std::cout << "[FSM] " << reason << "; policy entry rejected: worst joint " << gate.worstJoint
              << " error " << gate.worstError << " rad, max |dq| " << gate.worstVelocity
              << "; returning to fixed stand" << std::endl;
    return FSMStateName::FIXEDSTAND;
  }
  std::cout << "[FSM] " << reason << ", handing off to the velocity policy" << std::endl;
  return FSMStateName::LOCO;
}
