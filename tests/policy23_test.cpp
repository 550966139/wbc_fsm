#include "common/policy23.h"
#include "test_support.h"
#include <limits>

int main() {
  TestSuite suite;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::array<int, 23> expectedMapping{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                            12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
  suite.check(g1::kPolicyToMotor == expectedMapping, "exact G1 23DoF motor order");
  std::array<int, g1::kMotorCount> motorState{};
  for (std::size_t slot = 0; slot < motorState.size(); ++slot)
    motorState[slot] = static_cast<int>(100 + slot);
  const auto selected = g1::selectPolicyJoints(motorState);
  for (std::size_t joint = 0; joint < selected.size(); ++joint)
    suite.check(selected[joint] == 100 + expectedMapping[joint], "motor state mapped to policy");
  for (std::size_t slot = 0; slot < g1::kMotorCount; ++slot) {
    const bool absent =
        slot == 13 || slot == 14 || slot == 20 || slot == 21 || slot == 27 || slot == 28;
    suite.check(g1::isPolicyMotor(slot) == !absent, "absent joints excluded from commands");
  }

  const auto upright = g1::gravityFromQuaternion({1.f, 0.f, 0.f, 0.f});
  suite.near(upright[0], 0.f, "identity IMU gravity x");
  suite.near(upright[1], 0.f, "identity IMU gravity y");
  suite.near(upright[2], -1.f, "identity IMU gravity z");
  const float halfRoot = std::sqrt(.5f);
  const auto rolled = g1::gravityFromQuaternion({halfRoot, halfRoot, 0.f, 0.f});
  suite.near(rolled[1], -1.f, "wxyz quaternion roll direction");
  suite.near(rolled[2], 0.f, "90 degree roll removes vertical gravity");
  const auto pitched = g1::gravityFromQuaternion({halfRoot, 0.f, halfRoot, 0.f});
  suite.near(pitched[0], 1.f, "wxyz quaternion pitch direction");
  suite.near(g1::gravityFromQuaternion({1.01f, 0.f, 0.f, 0.f})[2], -1.f,
             "small quaternion norm error normalized");
  suite.throws([] { g1::gravityFromQuaternion({0.f, 0.f, 0.f, 0.f}); }, "zero quaternion rejected");
  suite.throws([] { g1::gravityFromQuaternion({2.f, 0.f, 0.f, 0.f}); },
               "unnormalized IMU rejected");
  suite.throws([&] { g1::gravityFromQuaternion({nan, 0.f, 0.f, 0.f}); }, "NaN IMU rejected");

  {
    const std::array<float, 4> limits{.3f, .3f, .4f, .25f};
    suite.check(g1::commandFromSticks23(0.f, 0.f, 0.f, 0.f, limits) == std::array<float, 4>{},
                "centered sticks give zero command");
    suite.near(g1::commandFromSticks23(1.f, 0.f, 0.f, 0.f, limits)[0], .3f,
               "left stick up commands forward velocity");
    suite.near(g1::commandFromSticks23(0.f, 0.f, 0.f, 1.f, limits)[0], .3f,
               "right stick up commands forward velocity");
    suite.near(g1::commandFromSticks23(-.5f, 0.f, 0.f, -.5f, limits)[0], -.3f,
               "backward stick sum clamps to the vx limit");
    suite.near(g1::commandFromSticks23(.6f, 0.f, 0.f, .6f, limits)[0], .3f,
               "forward stick sum saturates at the vx limit");
    suite.near(g1::commandFromSticks23(0.f, 1.f, 0.f, 0.f, limits)[1], .3f,
               "left stick right commands lateral velocity");
    suite.near(g1::commandFromSticks23(0.f, 0.f, 1.f, 0.f, limits)[2], .4f,
               "right stick right commands yaw rate");
    suite.near(g1::commandFromSticks23(0.f, 0.f, 0.f, 1.f, limits)[3], 0.f,
               "pitch dimension stays zero for any stick input");
    suite.check(g1::commandFromSticks23(.05f, .09f, .05f, 0.f, limits) == std::array<float, 4>{},
                "below-deadzone sticks give zero command");
    suite.near(g1::commandFromSticks23(.5f, 0.f, 0.f, -.5f, limits)[0], 0.f,
               "opposed vertical sticks cancel to zero");
    suite.throws([&] { g1::commandFromSticks23(nan, 0.f, 0.f, 0.f, limits); },
                 "NaN stick input rejected");
    suite.throws([&] { g1::commandFromSticks23(0.f, inf, 0.f, 0.f, limits); },
                 "infinite stick input rejected");
  }

  g1::Joints23 defaults{}, q{}, dq{}, last{}, scale{};
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    defaults[i] = static_cast<float>(i) * .01f;
    q[i] = defaults[i] + static_cast<float>(i + 1) * .02f;
    dq[i] = static_cast<float>(i) * .03f;
    last[i] = static_cast<float>(i) * -.04f;
    scale[i] = .25f + static_cast<float>(i) * .01f;
  }
  const std::array<float, 3> gyro{.1f, .2f, .3f};
  const std::array<float, 4> command{.2f, -.1f, .3f, -.15f};
  float phase = .99f;
  const auto obs =
      g1::observation23(gyro, upright, command, q, dq, defaults, last, phase, .02f, .6f);
  suite.check(obs.size() == 81, "policy observation contains exactly 81 scalars");
  suite.near(phase, .99f + .02f / .6f - 1.f, "phase wraps after one period");
  for (std::size_t i = 0; i < 3; ++i) {
    suite.near(obs[i], gyro[i], "observation angular velocity order");
    suite.near(obs[3 + i], upright[i], "observation projected gravity order");
  }
  for (std::size_t i = 0; i < 4; ++i)
    suite.near(obs[6 + i], command[i], "observation command order including pitch");
  suite.near(obs[10], std::sin(phase * 6.28318530718f), "gait sine position");
  suite.near(obs[11], std::cos(phase * 6.28318530718f), "gait cosine position");
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
    suite.near(obs[12 + i], q[i] - defaults[i], "relative joint positions in observation");
    suite.near(obs[35 + i], dq[i], "joint velocities in observation");
    suite.near(obs[58 + i], last[i], "previous actions in observation");
  }
  const float oldPhase = phase;
  auto masked = g1::observation23(gyro, upright, {}, q, dq, defaults, last, phase, .02f, .6f);
  suite.near(masked[10], 0.f, "zero command masks gait sine");
  suite.near(masked[11], 0.f, "zero command masks gait cosine");
  suite.near(phase, oldPhase + .02f / .6f, "masked gait still advances its phase");
  masked = g1::observation23(gyro, upright, {.099f, 0.f, 0.f, 0.f}, q, dq, defaults, last, phase,
                             .02f, .6f);
  suite.near(masked[10], 0.f, "below-threshold command masks phase");
  masked = g1::observation23(gyro, upright, {0.f, 0.f, 0.f, .099f}, q, dq, defaults, last, phase,
                             .02f, .6f);
  suite.near(masked[10], 0.f, "pitch-only below-threshold command masks phase");
  const auto active = g1::observation23(gyro, upright, {.1f, 0.f, 0.f, 0.f}, q, dq, defaults, last,
                                        phase, .02f, .6f);
  suite.check(std::abs(active[10]) + std::abs(active[11]) > .1f, "threshold command enables phase");
  const auto pitchActive = g1::observation23(gyro, upright, {0.f, 0.f, 0.f, .1f}, q, dq, defaults,
                                             last, phase, .02f, .6f);
  suite.check(std::abs(pitchActive[10]) + std::abs(pitchActive[11]) > .1f,
              "pitch-only threshold command enables phase");
  for (float invalid : {0.f, -.02f, nan, inf}) {
    suite.throws(
        [&] { g1::observation23(gyro, upright, {}, q, dq, defaults, last, phase, invalid, .6f); },
        "invalid timestep rejected even when gait is masked");
    suite.throws(
        [&] { g1::observation23(gyro, upright, {}, q, dq, defaults, last, phase, .02f, invalid); },
        "invalid gait period rejected even when gait is masked");
  }
  float invalidPhase = nan;
  suite.throws(
      [&] { g1::observation23(gyro, upright, {}, q, dq, defaults, last, invalidPhase, .02f, .6f); },
      "NaN phase rejected even when gait is masked");
  auto invalidQ = q;
  invalidQ[22] = nan;
  suite.throws(
      [&] {
        g1::observation23(gyro, upright, command, invalidQ, dq, defaults, last, phase, .02f, .6f);
      },
      "non-finite last joint observation rejected");

  const auto targets = g1::actionTargets23(last, scale, defaults);
  for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
    suite.near(targets[i], defaults[i] + last[i] * scale[i], "per-joint action scale and offset");
  suite.check(g1::actionTargets23({}, scale, defaults) == defaults,
              "zero action gives default posture");
  auto invalidAction = last;
  invalidAction[22] = nan;
  suite.throws([&] { g1::actionTargets23(invalidAction, scale, defaults); }, "NaN action rejected");
  invalidAction[22] = inf;
  suite.throws([&] { g1::actionTargets23(invalidAction, scale, defaults); },
               "infinite action rejected");
  auto invalidScale = scale;
  invalidScale[22] = nan;
  suite.throws([&] { g1::actionTargets23(last, invalidScale, defaults); }, "NaN scale rejected");

  {
    g1::Joints23 lo{}, hi{};
    lo.fill(-1.f);
    hi.fill(1.f);
    auto saturated = targets;
    saturated[22] = 4.f;
    suite.near(g1::clampTargets23(saturated, lo, hi)[22], 1.f, "target above upper limit clamped");
    saturated[22] = -4.f;
    suite.near(g1::clampTargets23(saturated, lo, hi)[22], -1.f, "target below lower limit clamped");
    saturated[22] = targets[22];
    const auto passthrough = g1::clampTargets23(saturated, lo, hi);
    suite.near(passthrough[22], targets[22], "in-range target passes through clamping");
    auto inverted = lo;
    inverted[22] = 2.f;
    suite.throws([&] { g1::clampTargets23(saturated, inverted, hi); },
                 "inverted limits rejected by clamping");
    auto nonFinite = targets;
    nonFinite[22] = nan;
    suite.throws([&] { g1::clampTargets23(nonFinite, lo, hi); },
                 "non-finite target rejected by clamping");
  }

  g1::Joints23 lower{}, upper{}, target{}, measured{}, previous{};
  lower.fill(-1.f);
  upper.fill(1.f);
  target[22] = .25f;
  g1::validateTargets23(target, measured, previous, lower, upper, .5f, .25f);
  suite.check(true, "target at step boundary accepted");
  target[22] = .2501f;
  suite.throws([&] { g1::validateTargets23(target, measured, previous, lower, upper, .5f, .25f); },
               "last joint target step rejected");
  target[22] = .501f;
  previous[22] = .5f;
  suite.throws([&] { g1::validateTargets23(target, measured, previous, lower, upper, .5f, .25f); },
               "excessive tracking error rejected");
  target[22] = 1.001f;
  measured[22] = previous[22] = 1.f;
  suite.throws([&] { g1::validateTargets23(target, measured, previous, lower, upper, .5f, .25f); },
               "upper joint limit rejected");
  target[22] = -1.001f;
  measured[22] = previous[22] = -1.f;
  suite.throws([&] { g1::validateTargets23(target, measured, previous, lower, upper, .5f, .25f); },
               "lower joint limit rejected");
  target.fill(0.f);
  measured.fill(0.f);
  previous.fill(0.f);
  for (auto *values : {&target, &measured, &previous, &lower, &upper}) {
    const float valid = (*values)[22];
    (*values)[22] = nan;
    suite.throws(
        [&] { g1::validateTargets23(target, measured, previous, lower, upper, .5f, .25f); },
        "non-finite target validation input rejected");
    (*values)[22] = valid;
  }
  for (float limit : {0.f, -.5f, nan, inf}) {
    suite.throws(
        [&] { g1::validateTargets23(target, measured, previous, lower, upper, limit, .25f); },
        "invalid tracking error limit rejected");
    suite.throws(
        [&] { g1::validateTargets23(target, measured, previous, lower, upper, .5f, limit); },
        "invalid target step limit rejected");
  }
  return suite.result();
}
