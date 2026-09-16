#include "common/policy23.h"
#include "control/config23.h"
#include <cassert>
#include <cmath>

int main() {
  const auto cfg = g1::Config23::load("config/g1_23dof.json");
  g1::Joints23 q = cfg.defaults, dq{}, last{};
  std::array<float, 3> gyro{}, gravity{0.f, 0.f, -1.f}, command{.2f, 0.f, 0.f};
  float phase = 0.f;
  const auto obs = g1::observation23(gyro, gravity, command, q, dq, cfg.defaults,
                                    last, phase, cfg.dt, cfg.gaitPeriod);
  assert(phase > 0.f && phase < 1.f);
  assert(std::all_of(obs.begin(), obs.end(), [](float x) { return std::isfinite(x); }));
  assert(std::abs(obs[9]) > 0.f || std::abs(obs[10]) > 0.f);
  const auto targets = g1::actionTargets23(last, cfg.scale, cfg.defaults);
  assert(targets == cfg.defaults);
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
    assert(g1::kPolicyToMotor[i] >= 0 && g1::kPolicyToMotor[i] < 29);
  return 0;
}
