#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <string>

namespace control {
using SafetyClock = std::chrono::steady_clock;
struct SafetyLimits {
  float projectedGravityThreshold = 0.5f; // abs(gravity.z + 1), not sin(tilt)
  float recoveryGravityThreshold = 0.15f;
  float maxJointVelocity = 30.0f;
  float maxTrackingError = 0.8f;
  float maxTargetStep = 0.35f;
  std::chrono::milliseconds stateTimeout{100};
  std::chrono::milliseconds tiltConfirmation{40};
  std::chrono::milliseconds recoveryDuration{500};
};

class SafetySupervisor {
public:
  explicit SafetySupervisor(SafetyLimits limits = {}) : limits_(limits) {}
  void trip(const std::string &reason) {
    if (!latched_) reason_ = reason;
    latched_ = true;
    recoverySince_ = {};
  }
  bool checkState(bool received, SafetyClock::time_point stamp,
                  const std::array<float, 3> &gravity,
                  SafetyClock::time_point now = SafetyClock::now()) {
    if (!received || stamp > now || now - stamp > limits_.stateTimeout) {
      trip("LowState missing or stale");
      return false;
    }
    float norm = 0;
    for (float v : gravity) norm += v * v;
    if (!std::isfinite(norm) || std::abs(norm - 1.f) > 0.05f) {
      trip("invalid IMU gravity");
      return false;
    }
    const float error = std::abs(gravity[2] + 1.f);
    if (error > limits_.projectedGravityThreshold) {
      recoverySince_ = {};
      if (tiltSince_ == SafetyClock::time_point{}) tiltSince_ = now;
      if (now - tiltSince_ >= limits_.tiltConfirmation) trip("excessive tilt");
    } else {
      tiltSince_ = {};
      if (error <= limits_.recoveryGravityThreshold) {
        if (recoverySince_ == SafetyClock::time_point{}) recoverySince_ = now;
      } else recoverySince_ = {};
    }
    return !latched_;
  }
  bool recover(bool newStartPress, SafetyClock::time_point now = SafetyClock::now()) {
    if (newStartPress && recoverySince_ != SafetyClock::time_point{} &&
        now - recoverySince_ >= limits_.recoveryDuration) {
      latched_ = false;
      reason_.clear();
    }
    return !latched_;
  }
  bool latched() const { return latched_; }
  const std::string &reason() const { return reason_; }
  const SafetyLimits &limits() const { return limits_; }
private:
  SafetyLimits limits_;
  bool latched_ = false;
  std::string reason_;
  SafetyClock::time_point tiltSince_{}, recoverySince_{};
};
}
