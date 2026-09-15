#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>

namespace control {
struct SafetyLimits {
    float projectedGravityThreshold = 0.5f;
    float maxJointVelocity = 30.0f;
    float maxTargetStep = 0.35f;
    std::chrono::milliseconds stateTimeout{100};
};

class SafetySupervisor {
public:
    explicit SafetySupervisor(SafetyLimits limits = {}) : limits_(limits) {}

    void stateReceived() { lastState_ = Clock::now(); hasState_ = true; }
    bool stateTimedOut() const {
        return !hasState_ || Clock::now() - lastState_ > limits_.stateTimeout;
    }
    bool orientationUnsafe(const std::array<float, 3>& projectedGravity) const {
        const float error = std::sqrt((projectedGravity[0]) * projectedGravity[0] +
                                      (projectedGravity[1]) * projectedGravity[1]);
        return error > limits_.projectedGravityThreshold;
    }
    bool jointsUnsafe(const float* q, const float* dq, const float* target,
                      std::size_t count) const {
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(q[i]) || !std::isfinite(dq[i]) || !std::isfinite(target[i])) return true;
            if (std::fabs(dq[i]) > limits_.maxJointVelocity ||
                std::fabs(target[i] - q[i]) > limits_.maxTargetStep) return true;
        }
        return false;
    }
private:
    using Clock = std::chrono::steady_clock;
    SafetyLimits limits_;
    Clock::time_point lastState_{};
    bool hasState_ = false;
};
}  // namespace control
