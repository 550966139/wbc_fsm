#include "common/motion_csv.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace g1 {

MotionCsv23 MotionCsv23::load(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot open tracking motion CSV: " + path.string());
  const auto invalid = [&](std::size_t line, const std::string &reason) {
    return std::runtime_error("invalid tracking motion CSV " + path.string() + " line " +
                              std::to_string(line) + ": " + reason);
  };
  MotionCsv23 motion;
  std::string text;
  std::size_t line = 0;
  while (std::getline(file, text)) {
    ++line;
    if (!text.empty() && text.back() == '\r')
      text.pop_back();
    if (text.find_first_not_of(" \t") == std::string::npos)
      continue; // tolerate blank lines, including a trailing newline
    std::vector<float> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
      std::size_t parsed = 0;
      float value;
      try {
        value = std::stof(token, &parsed);
      } catch (const std::exception &) {
        throw invalid(line, "unparseable value '" + token + "'");
      }
      if (parsed != token.size())
        throw invalid(line, "trailing characters in value '" + token + "'");
      if (!std::isfinite(value))
        throw invalid(line, "non-finite value");
      values.push_back(value);
    }
    if (values.size() != kColumns)
      throw invalid(line, "expected " + std::to_string(kColumns) + " columns, got " +
                              std::to_string(values.size()));
    Frame frame;
    std::copy_n(values.begin(), kPolicyDof, frame.pos.begin());
    std::copy_n(values.begin() + kPolicyDof, kPolicyDof, frame.vel.begin());
    std::copy_n(values.begin() + 2 * kPolicyDof, 4, frame.quat.begin());
    float norm = 0;
    for (float v : frame.quat)
      norm += v * v;
    if (std::abs(std::sqrt(norm) - 1.f) > 1e-3f)
      throw invalid(line, "anchor quaternion not unit norm");
    motion.frames_.push_back(frame);
  }
  if (motion.frames_.size() < 2)
    throw std::runtime_error("tracking motion CSV " + path.string() +
                             " must contain at least 2 frames");
  return motion;
}

const MotionCsv23::Frame &MotionCsv23::at(std::size_t frame) const {
  if (frame >= frames_.size())
    throw std::out_of_range("tracking motion frame " + std::to_string(frame) + " outside [0, " +
                            std::to_string(frames_.size()) + ")");
  return frames_[frame];
}

} // namespace g1
