#include "FSM/State_Policy23.h"
#include <algorithm>
#include <filesystem>

namespace {
void requireShape(const Ort::ConstTensorTypeAndShapeInfo &tensor, int64_t width,
                  const char *label) {
  const auto shape = tensor.GetShape();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 2 ||
      (shape[0] != 1 && shape[0] != -1) || shape[1] != width)
    throw std::runtime_error(std::string("23DoF policy requires float ") + label + " [1," +
                             std::to_string(width) + "]");
}
} // namespace

State_Policy23::State_Policy23(CtrlComponents *components)
    : FSMState(components, FSMStateName::LOCO, "23DoF locomotion") {
  const auto &config = components->config23.value();
  if (!std::filesystem::is_regular_file(config.model))
    throw std::runtime_error("23DoF policy missing: " + config.model.string() +
                             "; provide a matching trained 81-input / 23-output model");
  options_.SetIntraOpNumThreads(1);
  options_.SetInterOpNumThreads(1);
  options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ = std::make_unique<Ort::Session>(env_, config.model.c_str(), options_);
  if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
    throw std::runtime_error("23DoF policy must have one observation input and one action output");
  requireShape(session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo(), 81, "input");
  requireShape(session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo(), 23, "output");
  Ort::AllocatorWithDefaultOptions allocator;
  inputName_ = session_->GetInputNameAllocated(0, allocator).get();
  outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

  // Validate executable output before the DDS publisher can be started.
  float phase = 0.f;
  auto observation = g1::observation23({}, {0.f, 0.f, -1.f}, {}, config.defaults, {},
                                       config.defaults, {}, phase, config.dt, config.gaitPeriod);
  (void)infer(observation);
}

g1::Joints23 State_Policy23::infer(g1::Observation23 &observation) {
  const std::array<int64_t, 2> shape{1, 81};
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(memory, observation.data(), observation.size(),
                                               shape.data(), shape.size());
  const char *inputs[] = {inputName_.c_str()};
  const char *outputs[] = {outputName_.c_str()};
  auto result = session_->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
  if (result.size() != 1 || !result[0].IsTensor())
    throw std::runtime_error("23DoF policy produced a non-tensor action");
  const auto info = result[0].GetTensorTypeAndShapeInfo();
  if (info.GetShape() != std::vector<int64_t>{1, 23} ||
      info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("23DoF policy output shape or type changed during inference");
  g1::Joints23 actions{};
  std::copy_n(result[0].GetTensorData<float>(), actions.size(), actions.begin());
  for (float value : actions)
    if (!std::isfinite(value))
      throw std::runtime_error("23DoF policy produced non-finite action");
  return actions;
}

void State_Policy23::enter() {
  phase_ = 0.f;
  lastAction_.fill(0.f);
}

void State_Policy23::run() {
  const auto &config = _ctrlComp->config23.value();
  const auto &user = _lowState->userValue;
  // Both vertical stick axes command forward/backward; lateral and yaw are
  // unchanged. The pitch dimension is held at zero, see commandFromSticks23.
  const auto command =
      g1::commandFromSticks23(user.ly, user.lx, user.rx, user.ry, config.commandLimit);
  const auto &gyro = _lowState->imu.gyroscope;
  auto observation =
      g1::observation23({gyro[0], gyro[1], gyro[2]}, _ctrlComp->getGravity(), command,
                        _ctrlComp->positions23(), _ctrlComp->velocities23(), config.defaults,
                        lastAction_, phase_, config.dt, config.gaitPeriod);
  const auto actions = infer(observation);
  _ctrlComp->setTargets23(g1::clampTargets23(
      g1::actionTargets23(actions, config.scale, config.defaults), config.lower, config.upper));
  lastAction_ = actions;
}

void State_Policy23::exit() {
  lastAction_.fill(0.f);
  phase_ = 0.f;
}

FSMStateName State_Policy23::checkChange() { return FSMStateName::LOCO; }
