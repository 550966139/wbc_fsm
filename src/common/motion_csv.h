#pragma once
#include "common/g1_23dof.h"
#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace g1 {
// Deployment-side motion reference for the 23DoF tracking policy: plain CSV,
// one row per 50 Hz frame, no header, 50 columns per row:
// ref_joint_pos[23], ref_joint_vel[23], ref_anchor_quat_wxyz[4].
// Generated from the training NPZ so values stay bit-identical to training;
// the file is already at control rate, so no interpolation is applied.
class MotionCsv23 {
public:
  static constexpr std::size_t kColumns = 2 * kPolicyDof + 4;

  // Throws std::runtime_error with the offending line on any violation:
  // unreadable file, fewer than 2 frames, wrong column count, non-finite
  // values, or anchor quaternions off unit norm by more than 1e-3.
  static MotionCsv23 load(const std::filesystem::path &path);

  std::size_t frameCount() const { return frames_.size(); }
  // Pointers to the 23 position / 23 velocity / 4 quaternion (wxyz) values of
  // a frame; std::out_of_range when frame >= frameCount().
  const float *jointPos(std::size_t frame) const { return at(frame).pos.data(); }
  const float *jointVel(std::size_t frame) const { return at(frame).vel.data(); }
  const float *anchorQuat(std::size_t frame) const { return at(frame).quat.data(); }

private:
  struct Frame {
    std::array<float, kPolicyDof> pos{}, vel{};
    std::array<float, 4> quat{};
  };
  const Frame &at(std::size_t frame) const;
  std::vector<Frame> frames_;
};
} // namespace g1
