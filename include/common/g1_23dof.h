#pragma once

#include <array>
#include <cstddef>

namespace g1 {
constexpr std::size_t kMotorCount = 29;
constexpr std::size_t kPolicyDof = 23;

// Official G1-23DoF policy order -> LowCmd motor slot.
constexpr std::array<int, kPolicyDof> kPolicyToMotor = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 17, 18, 19, 22, 23, 24, 25, 26};

constexpr bool validPolicyMapping() {
    for (int i : kPolicyToMotor)
        if (i < 0 || i >= static_cast<int>(kMotorCount)) return false;
    return true;
}
static_assert(validPolicyMapping(), "invalid G1-23DoF motor mapping");

template <typename T>
std::array<T, kPolicyDof> selectPolicyJoints(const std::array<T, kMotorCount>& motors) {
    std::array<T, kPolicyDof> result{};
    for (std::size_t i = 0; i < kPolicyDof; ++i) result[i] = motors[kPolicyToMotor[i]];
    return result;
}
}  // namespace g1
