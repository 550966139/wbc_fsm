#include "common/base_state_estimator.h"
#include <cmath>

namespace g1 {
namespace {
constexpr float kDt = 0.02f;          // 50 Hz control cycle
constexpr float kAlphaQ = 0.46651f;   // fc 5 Hz, one pole
constexpr float kAlphaTau = 0.39725f; // fc 4 Hz
constexpr float kAlphaAcc = 0.84800f; // fc 15 Hz
constexpr float kBeta = 0.98f;        // complementary weight of accel prediction
constexpr float kTauFrac = 0.28f;     // stance: load fraction threshold
constexpr float kSwingGate = 1.8f;    // stance: |dq_hip_p| + |dq_knee| <= gate
constexpr float kGz = 9.81f;

using Vec3 = std::array<float, 3>;
using Mat3 = std::array<std::array<float, 3>, 3>;

Vec3 cross3(const Vec3 &a, const Vec3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 operator*(float s, const Vec3 &a) { return {s * a[0], s * a[1], s * a[2]}; }

Mat3 quatToMat3(const std::array<float, 4> &q) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
           {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
           {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}
Mat3 mul3(const Mat3 &a, const Mat3 &b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        r[i][j] += a[i][k] * b[k][j];
  return r;
}
Vec3 mul3v(const Mat3 &a, const Vec3 &v) {
  return {a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
          a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
          a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2]};
}
Mat3 axisRot(const Vec3 &axis, float ang) {
  const float n = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  const float a = axis[0] / n, b = axis[1] / n, c = axis[2] / n;
  const float s = std::sin(ang), t = 1 - std::cos(ang);
  return {{{t * a * a + std::cos(ang), t * a * b - s * c, t * a * c + s * b},
           {t * a * b + s * c, t * b * b + std::cos(ang), t * b * c - s * a},
           {t * a * c - s * b, t * b * c + s * a, t * c * c + std::cos(ang)}}};
}

// Pelvis -> ankle_roll chain from the training XML (g1_23dof.xml). Each entry:
// joint axis, offset applied in the pre-joint frame, fixed rotation before the
// joint (the +/-10.02 deg y-rotations on hip_roll and knee links).
struct ChainLink {
  Vec3 axis, offset;
  Mat3 fix;
};
using Chain = std::array<ChainLink, 6>;
Chain legChain(std::size_t side) { // side 0 = left, 1 = right
  const float s = side == 0 ? 1.f : -1.f;
  const Mat3 I = quatToMat3({1, 0, 0, 0});
  const Mat3 hipRollFix = quatToMat3({0.996179f, 0.f, -0.0873386f, 0.f});
  const Mat3 kneeFix = quatToMat3({0.996179f, 0.f, 0.0873386f, 0.f});
  return {{
      {{0, 1, 0}, {0, s * 0.064452f, -0.1027f}, I},
      {{1, 0, 0}, {0, s * 0.052f, -0.030465f}, hipRollFix},
      {{0, 0, 1}, {0.025001f, 0, -0.12412f}, I},
      {{0, 1, 0}, {-0.078273f, s * 0.0021489f, -0.17734f}, kneeFix},
      {{0, 1, 0}, {0, -s * 9.4445e-05f, -0.30001f}, I},
      {{1, 0, 0}, {0, 0, -0.017558f}, I},
  }};
}
constexpr std::array<float, 4> kSoleX = {-0.04f, 0.f, 0.04f, 0.10f}; // heel -> toe
constexpr Vec3 kSoleZ = {0.f, 0.f, -0.037f};
constexpr Vec3 kImuOffset = {0.04525f, 0.f, -0.08339f}; // imu_in_pelvis

// Forward kinematics of one leg; fills joint axes/origins (pelvis frame) and
// the ankle_roll frame orientation, plus the origin position.
struct LegFK {
  Vec3 axes[6], points[6];
  Vec3 ankleOrigin;
  Mat3 ankleRot;
};
LegFK legFK(const float *qLeg, std::size_t side) {
  static const Chain left = legChain(0), right = legChain(1);
  const Chain &chain = side == 0 ? left : right;
  Mat3 R = quatToMat3({1, 0, 0, 0});
  Vec3 p = {0, 0, 0};
  LegFK fk{};
  for (std::size_t i = 0; i < 6; ++i) {
    const Mat3 Rpre = mul3(R, chain[i].fix);
    fk.axes[i] = mul3v(Rpre, chain[i].axis);
    p = p + mul3v(R, chain[i].offset);
    R = mul3(Rpre, axisRot(chain[i].axis, qLeg[i]));
    fk.points[i] = p;
  }
  fk.ankleOrigin = p;
  fk.ankleRot = R;
  return fk;
}
} // namespace

void BaseStateEstimator23::filter(const Inputs &in) {
  for (std::size_t i = 0; i < kPolicyDof; ++i) {
    q_[1][i] = q_[1][i] + kAlphaQ * (in.q[i] - q_[1][i]);
    q_[0][i] = q_[0][i] + kAlphaQ * (q_[1][i] - q_[0][i]);
    qf_[i] = q_[0][i];
    dq_[1][i] = dq_[1][i] + kAlphaQ * (in.dq[i] - dq_[1][i]);
    dq_[0][i] = dq_[0][i] + kAlphaQ * (dq_[1][i] - dq_[0][i]);
    dqf_[i] = dq_[0][i];
    tau_[1][i] = tau_[1][i] + kAlphaTau * (in.tau[i] - tau_[1][i]);
    tau_[0][i] = tau_[0][i] + kAlphaTau * (tau_[1][i] - tau_[0][i]);
    tauf_[i] = tau_[0][i];
  }
  for (std::size_t i = 0; i < 3; ++i) {
    acc_[0][i] = acc_[0][i] + kAlphaAcc * (in.accel[i] - acc_[0][i]);
    accf_[i] = acc_[0][i];
  }
}

std::array<float, 3> BaseStateEstimator23::update(const Inputs &in) {
  if (!in.valid)
    return {v_[0] + cross3(in.gyro, kImuOffset)[0], v_[1] + cross3(in.gyro, kImuOffset)[1],
            v_[2] + cross3(in.gyro, kImuOffset)[2]};
  for (float x : in.q)
    if (!std::isfinite(x))
      return v_;
  filter(in);
  const Mat3 R = quatToMat3(in.quat);
  Vec3 aLin = accf_;
  for (std::size_t i = 0; i < 3; ++i)
    aLin[i] += -kGz * R[2][i]; // R^T * (0,0,-9.81): gravity removal
  const Vec3 vPred = {v_[0] + aLin[0] * kDt, v_[1] + aLin[1] * kDt, v_[2] + aLin[2] * kDt};

  // Stance gating: torque-load fraction + swing-speed gate.
  const float loadL = std::abs(tauf_[0]) + 0.5f * std::abs(tauf_[3]) + 0.5f * std::abs(tauf_[4]);
  const float loadR = std::abs(tauf_[6]) + 0.5f * std::abs(tauf_[9]) + 0.5f * std::abs(tauf_[10]);
  const float swingL = std::abs(dqf_[0]) + std::abs(dqf_[3]);
  const float swingR = std::abs(dqf_[6]) + std::abs(dqf_[9]);
  const float tot = loadL + loadR + 1e-9f;
  const float fracL = loadL / tot, fracR = loadR / tot;
  stance_[0] = fracL >= kTauFrac && swingL <= kSwingGate;
  stance_[1] = fracR >= kTauFrac && swingR <= kSwingGate;

  Vec3 vLeg{};
  float wSum = 0.f;
  for (std::size_t side = 0; side < 2; ++side) {
    if (!stance_[side])
      continue;
    const float frac = side == 0 ? fracL : fracR;
    const std::size_t off = side * 6;
    const float qLeg[6] = {qf_[off],     qf_[off + 1], qf_[off + 2],
                           qf_[off + 3], qf_[off + 4], qf_[off + 5]};
    const LegFK fk = legFK(qLeg, side);
    // Sole candidate with implied pelvis velocity closest to the prediction.
    Vec3 best{};
    float bestD = 1e30f;
    for (float sx : kSoleX) {
      const Vec3 candLocal = {sx, kSoleZ[1], kSoleZ[2]};
      const Vec3 foot = fk.ankleOrigin + mul3v(fk.ankleRot, candLocal);
      Vec3 vImp{};
      for (std::size_t j = 0; j < 6; ++j) {
        const Vec3 arm = foot - fk.points[j];
        const Vec3 col = cross3(fk.axes[j], arm);
        for (std::size_t c = 0; c < 3; ++c)
          vImp[c] += -col[c] * dqf_[off + j];
      }
      const Vec3 d = vImp - vPred;
      const float dist = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (dist < bestD) {
        bestD = dist;
        best = vImp;
      }
    }
    vLeg = vLeg + frac * best;
    wSum += frac;
  }
  if (wSum > 0.f) {
    vLeg = (1.f / wSum) * vLeg;
    for (std::size_t c = 0; c < 3; ++c)
      v_[c] = kBeta * vPred[c] + (1.f - kBeta) * vLeg[c];
  } else {
    v_ = vPred;
  }
  const Vec3 lever = cross3(in.gyro, kImuOffset);
  return {v_[0] + lever[0], v_[1] + lever[1], v_[2] + lever[2]};
}
} // namespace g1
