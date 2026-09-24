#include "common/motion_csv.h"
#include "common/tracking23.h"
#include "test_support.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace {
const float kPi = 3.14159265358979f;

g1::Quat23 rotZ(float angle) { return g1::quatAboutZ23(angle); }
g1::Quat23 rotX(float angle) {
  const float half = .5f * angle;
  return {std::cos(half), std::sin(half), 0.f, 0.f};
}

class CsvFixture {
public:
  CsvFixture() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    directory =
        std::filesystem::temp_directory_path() / ("g1-motion-csv-test-" + std::to_string(unique));
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("cannot create unique test directory");
    path = directory / "motion.csv";
  }
  ~CsvFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
  // rows: each entry is 50 values (pos 23, vel 23, quat wxyz 4)
  void write(const std::vector<std::vector<float>> &rows, const std::string &suffix = "") const {
    std::ofstream output(path);
    for (const auto &row : rows) {
      for (std::size_t i = 0; i < row.size(); ++i)
        output << (i ? "," : "") << row[i];
      output << "\n";
    }
    output << suffix;
  }
  std::vector<float> row(float posOffset, float velValue, g1::Quat23 quat) const {
    std::vector<float> values;
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      values.push_back(posOffset + static_cast<float>(i) * .01f);
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      values.push_back(velValue);
    for (float v : quat)
      values.push_back(v);
    return values;
  }
  std::filesystem::path directory, path;
};
} // namespace

int main() {
  TestSuite suite;
#define CHECK(expression) suite.check((expression), #expression)
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float halfRoot = std::sqrt(.5f);

  static_assert(g1::kTrackingObsSize == 124, "tracking contract is 124 observations");
  CHECK(g1::kTrackingRefPos == 0 && g1::kTrackingRefVel == 23 && g1::kTrackingAnchor == 46 &&
        g1::kTrackingGyro == 52 && g1::kTrackingJointPos == 55 && g1::kTrackingJointVel == 78 &&
        g1::kTrackingLastAction == 101);
  CHECK(g1::kWaistYawJoint == 12 && g1::kPolicyToMotor[g1::kWaistYawJoint] == 12);
  CHECK(g1::MotionCsv23::kColumns == 50);

  // Quaternion multiplication: wxyz Hamilton, non-commutative.
  const auto identity = g1::Quat23{1.f, 0.f, 0.f, 0.f};
  const auto z90 = rotZ(kPi / 2), x90 = rotX(kPi / 2);
  const auto zz = g1::quatMul23(z90, z90); // Rz(90) (x) Rz(90) = Rz(180)
  suite.near(zz[0], 0.f, "z90 squared w");
  suite.near(zz[3], 1.f, "z90 squared z");
  const auto zx = g1::quatMul23(z90, x90);
  for (std::size_t i = 0; i < 4; ++i)
    suite.near(zx[i], .5f, "Rz90 (x) Rx90 component");
  const auto xz = g1::quatMul23(x90, z90);
  suite.near(xz[0], .5f, "Rx90 (x) Rz90 w");
  suite.near(xz[1], .5f, "Rx90 (x) Rz90 x");
  suite.near(xz[2], -.5f, "Rx90 (x) Rz90 y differs: multiplication order matters");
  suite.near(xz[3], .5f, "Rx90 (x) Rz90 z");
  const auto identityProduct = g1::quatMul23(identity, z90);
  for (std::size_t i = 0; i < 4; ++i)
    suite.near(identityProduct[i], z90[i], "identity multiplication is a pass-through");

  // Robot anchor: torso_link world quat = imu (x) Rz(waist_yaw).
  const auto anchorA = g1::robotAnchorQuat23(identity, kPi / 2);
  for (std::size_t i = 0; i < 4; ++i)
    suite.near(anchorA[i], z90[i], "identity IMU with 90 deg waist yaw");
  const auto anchorB = g1::robotAnchorQuat23(z90, kPi / 2);
  suite.near(std::abs(anchorB[3]), 1.f, "IMU yaw and waist yaw compose to 180 deg");
  suite.throws([&] { g1::robotAnchorQuat23({2.f, 0.f, 0.f, 0.f}, 0.f); },
               "unnormalized IMU rejected for anchor");
  suite.throws([&] { g1::robotAnchorQuat23(identity, nan); }, "NaN waist yaw rejected");

  // Yaw extraction and entry alignment.
  suite.near(g1::yawOf23(identity), 0.f, "identity yaw is zero");
  suite.near(g1::yawOf23(rotZ(.3f)), .3f, "yaw of pure z rotation");
  suite.near(g1::yawOf23(rotX(.4f)), 0.f, "pure roll has no yaw");
  {
    // Entry offset: robot anchor Rz(0.5), reference frame-0 anchor Rz(0.2);
    // aligning Rz(0.5 - 0.2) (x) ref must yaw-shift every reference frame.
    const auto align = g1::quatAboutZ23(g1::yawOf23(rotZ(.5f)) - g1::yawOf23(rotZ(.2f)));
    const auto aligned = g1::quatMul23(align, rotZ(.7f));
    suite.near(g1::yawOf23(aligned), 1.f, "aligned reference carries the entry yaw offset");
    const auto aligned0 = g1::quatMul23(align, rotZ(.2f));
    suite.near(g1::yawOf23(aligned0), .5f, "reference frame 0 aligns with the robot yaw");
  }

  // 6D anchor error: quat_rel = robot^-1 (x) ref, matrix first two columns
  // row-major {R00, R01, R10, R11, R20, R21}.
  {
    const auto same = g1::anchorError6(identity, identity);
    const std::array<float, 6> expectIdentity{1.f, 0.f, 0.f, 1.f, 0.f, 0.f};
    for (std::size_t i = 0; i < 6; ++i)
      suite.near(same[i], expectIdentity[i], "identity anchor error is identity matrix");
    // robot = Rz(90), ref = identity: rel = Rz(-90) = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    const auto turned = g1::anchorError6(z90, identity);
    const std::array<float, 6> expectTurned{0.f, 1.f, -1.f, 0.f, 0.f, 0.f};
    for (std::size_t i = 0; i < 6; ++i)
      suite.near(turned[i], expectTurned[i], "robot yawed 90 deg relative anchor");
    // robot = identity, ref = Rx(90): rel = Rx(90) = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    const auto rolled = g1::anchorError6(identity, x90);
    const std::array<float, 6> expectRolled{1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    for (std::size_t i = 0; i < 6; ++i)
      suite.near(rolled[i], expectRolled[i], "reference rolled 90 deg relative anchor");
    suite.throws([&] { g1::anchorError6({0.f, 0.f, 0.f, 0.f}, identity); },
                 "zero robot anchor rejected");
    suite.throws([&] { g1::anchorError6(identity, {nan, 0.f, 0.f, 0.f}); },
                 "NaN reference anchor rejected");
  }

  // Full 124-observation layout with hand-computed segments.
  {
    g1::Joints23 refPos{}, refVel{}, q{}, dq{}, defaults{}, last{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      refPos[i] = 1.f + static_cast<float>(i) * .1f;
      refVel[i] = -2.f + static_cast<float>(i) * .05f;
      defaults[i] = static_cast<float>(i) * .01f;
      q[i] = defaults[i] + static_cast<float>(i) * .02f;
      dq[i] = static_cast<float>(i) * -.03f;
      last[i] = static_cast<float>(i) * .04f;
    }
    const auto robot = rotZ(.25f), ref = rotZ(.1f); // rel = Rz(-0.15)
    const std::array<float, 3> gyro{.1f, .2f, .3f};
    const auto obs = g1::observationTracking23(refPos.data(), refVel.data(), robot, ref, gyro, q,
                                               dq, defaults, last);
    CHECK(obs.size() == 124);
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      suite.near(obs[g1::kTrackingRefPos + i], refPos[i], "reference joint positions first");
      suite.near(obs[g1::kTrackingRefVel + i], refVel[i], "reference joint velocities second");
      suite.near(obs[g1::kTrackingJointPos + i], q[i] - defaults[i],
                 "relative joint positions segment");
      suite.near(obs[g1::kTrackingJointVel + i], dq[i], "joint velocities segment");
      suite.near(obs[g1::kTrackingLastAction + i], last[i], "last action segment");
    }
    const float c = std::cos(.15f), s = std::sin(.15f);
    const std::array<float, 6> expectAnchor{c, s, -s, c, 0.f, 0.f};
    for (std::size_t i = 0; i < 6; ++i)
      suite.near(obs[g1::kTrackingAnchor + i], expectAnchor[i], "anchor error segment Rz(-0.15)");
    for (std::size_t i = 0; i < 3; ++i)
      suite.near(obs[g1::kTrackingGyro + i], gyro[i], "gyro segment");
    suite.throws(
        [&] {
          g1::observationTracking23(nullptr, refVel.data(), robot, ref, gyro, q, dq, defaults,
                                    last);
        },
        "missing reference positions rejected");
    auto badQ = q;
    badQ[12] = nan;
    suite.throws(
        [&] {
          g1::observationTracking23(refPos.data(), refVel.data(), robot, ref, gyro, badQ, dq,
                                    defaults, last);
        },
        "non-finite joint observation rejected");
    auto badRef = refPos;
    badRef[0] = std::numeric_limits<float>::infinity();
    suite.throws(
        [&] {
          g1::observationTracking23(badRef.data(), refVel.data(), robot, ref, gyro, q, dq, defaults,
                                    last);
        },
        "non-finite reference rejected");
  }

  // Action targets reuse the velocity-policy transform and clamping.
  {
    g1::Joints23 defaults{}, scale{}, action{}, lower{}, upper{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i) {
      defaults[i] = .1f * static_cast<float>(i) - .5f;
      scale[i] = .44f;
      action[i] = i % 2 ? .5f : -.5f;
      lower[i] = -2.f;
      upper[i] = 2.f;
    }
    const auto targets = g1::actionTargets23(action, scale, defaults);
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      suite.near(targets[i], defaults[i] + action[i] * scale[i], "tracking action transform");
    action[3] = 100.f;
    suite.near(g1::clampTargets23(g1::actionTargets23(action, scale, defaults), lower, upper)[3],
               2.f, "tracking targets saturate at joint limits like training");
  }

  // Shared entry-gate helper behind the FixedStand R2+A/R2+B gates and the
  // tracking <-> policy handoff gates.
  {
    g1::Joints23 q{}, dq{}, reference{};
    for (std::size_t i = 0; i < g1::kPolicyDof; ++i)
      reference[i] = static_cast<float>(i) * .01f;
    q = reference;
    auto gate = g1::entryGate23(q, dq, reference);
    CHECK(gate.passes(g1::kPolicyEntryMaxJointError, g1::kPolicyEntryMaxJointVel));
    suite.near(gate.worstError, 0.f, "gate zero error at the reference pose");
    q[7] = reference[7] + .14f;
    q[3] = reference[3] - .16f;
    dq[9] = .5f;
    gate = g1::entryGate23(q, dq, reference);
    CHECK(!gate.passes(g1::kPolicyEntryMaxJointError, g1::kPolicyEntryMaxJointVel));
    suite.check(gate.worstJoint == 3, "gate identifies the worst joint");
    suite.near(gate.worstError, .16f, "gate reports the worst joint error");
    suite.near(gate.worstVelocity, .5f, "gate reports the worst joint velocity");
    CHECK(gate.passes(.2f, .6f));
    dq[9] = 1.01f;
    q = reference;
    CHECK(!g1::entryGate23(q, dq, reference)
               .passes(g1::kPolicyEntryMaxJointError, g1::kPolicyEntryMaxJointVel));
    dq[9] = nan;
    suite.throws([&] { g1::entryGate23(q, dq, reference); }, "gate rejects non-finite velocity");
  }

  // Motion CSV loader.
  {
    CsvFixture fixture;
    suite.throws([&] { g1::MotionCsv23::load(fixture.directory / "missing.csv"); },
                 "missing motion file rejected");
    fixture.write({fixture.row(0.f, 0.f, identity), fixture.row(.2f, .1f, z90),
                   fixture.row(.4f, -.1f, identity)});
    const auto motion = g1::MotionCsv23::load(fixture.path);
    CHECK(motion.frameCount() == 3);
    suite.near(motion.jointPos(0)[0], 0.f, "frame 0 position column 0");
    suite.near(motion.jointPos(1)[5], .25f, "frame 1 position column 5");
    suite.near(motion.jointVel(1)[22], .1f, "frame 1 velocity column 22");
    suite.near(motion.jointVel(2)[0], -.1f, "frame 2 velocity column 0");
    suite.near(motion.anchorQuat(1)[0], halfRoot, "frame 1 anchor quat w");
    suite.near(motion.anchorQuat(1)[3], halfRoot, "frame 1 anchor quat z");
    suite.near(motion.anchorQuat(2)[0], 1.f, "frame 2 anchor quat w");
    suite.throws([&] { motion.jointPos(3); }, "frame beyond the motion rejected");
    suite.throws([&] { motion.anchorQuat(99); }, "far frame rejected");

    fixture.write({fixture.row(0.f, 0.f, identity)}, "\n\n");
    suite.throws([&] { g1::MotionCsv23::load(fixture.path); }, "single-frame motion rejected");

    auto shortRow = fixture.row(0.f, 0.f, identity);
    shortRow.pop_back();
    fixture.write({shortRow, fixture.row(0.f, 0.f, identity)});
    suite.throws([&] { g1::MotionCsv23::load(fixture.path); }, "49-column row rejected");

    auto longRow = fixture.row(0.f, 0.f, identity);
    longRow.push_back(0.f);
    fixture.write({longRow, fixture.row(0.f, 0.f, identity)});
    suite.throws([&] { g1::MotionCsv23::load(fixture.path); }, "51-column row rejected");

    fixture.write({fixture.row(0.f, 0.f, {1.01f, 0.f, 0.f, 0.f}), fixture.row(0.f, 0.f, identity)});
    suite.throws([&] { g1::MotionCsv23::load(fixture.path); },
                 "non-unit anchor quaternion rejected");
    fixture.write(
        {fixture.row(0.f, 0.f, {1.0005f, 0.f, 0.f, 0.f}), fixture.row(0.f, 0.f, identity)});
    CHECK(g1::MotionCsv23::load(fixture.path).frameCount() == 2);

    auto nanRow = fixture.row(0.f, 0.f, identity);
    nanRow[7] = nan;
    fixture.write({nanRow, fixture.row(0.f, 0.f, identity)});
    suite.throws([&] { g1::MotionCsv23::load(fixture.path); }, "NaN column rejected");

    {
      std::ofstream output(fixture.path);
      output << "1,2,three\n";
    }
    suite.throws([&] { g1::MotionCsv23::load(fixture.path); }, "non-numeric column rejected");

    // Trailing blank line and CRLF endings are tolerated.
    fixture.write({fixture.row(0.f, 0.f, identity), fixture.row(.1f, 0.f, identity)}, "\n");
    CHECK(g1::MotionCsv23::load(fixture.path).frameCount() == 2);
    {
      std::ofstream output(fixture.path);
      const auto a = fixture.row(0.f, 0.f, identity), b = fixture.row(.1f, 0.f, identity);
      for (const auto *row : {&a, &b}) {
        for (std::size_t i = 0; i < row->size(); ++i)
          output << (i ? "," : "") << (*row)[i];
        output << "\r\n";
      }
    }
    const auto crlf = g1::MotionCsv23::load(fixture.path);
    CHECK(crlf.frameCount() == 2);
    suite.near(crlf.jointPos(1)[0], .1f, "CRLF motion parses");
  }
  return suite.result();
}
