#include "control/config23.h"
#include "test_support.h"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace {
class ConfigFixture {
public:
  ConfigFixture() {
    std::ifstream input(std::filesystem::path(PROJECT_ROOT_DIR) / "config/g1_23dof.json");
    if (!input)
      throw std::runtime_error("cannot read checked-in test configuration");
    input >> valid;
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    directory =
        std::filesystem::temp_directory_path() / ("g1-config23-test-" + std::to_string(unique));
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("cannot create unique test directory");
    std::filesystem::create_directory(directory / "config");
    path = directory / "config/g1_23dof.json";
  }
  ~ConfigFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
  g1::Config23 load(const nlohmann::json &value) const {
    {
      std::ofstream output(path);
      if (!(output << value.dump(2)))
        throw std::runtime_error("cannot write test configuration");
    }
    return g1::Config23::load(path);
  }
  nlohmann::json valid;
  std::filesystem::path directory, path;
};
} // namespace

int main() {
  TestSuite suite;
  ConfigFixture fixture;
  const auto config = fixture.load(fixture.valid);
  suite.check(config.model == fixture.directory / "model/g1_23dof/policy.onnx",
              "model path resolves relative to the configuration project");
  suite.near(config.dt, .02f, "configuration uses 50Hz policy");
  suite.near(config.gaitPeriod, .6f, "configuration uses trained gait period");
  suite.check(config.defaults.size() == 23, "configuration uses all 23 joints");
  auto reject = [&](const std::string &description, auto mutate) {
    auto invalid = fixture.valid;
    mutate(invalid);
    suite.throws([&] { fixture.load(invalid); }, description);
  };
  reject("29DoF policy cannot load", [](auto &j) { j["policy_dof"] = 29; });
  reject("motor slot count cannot change", [](auto &j) { j["motor_count"] = 23; });
  reject("observation width cannot change", [](auto &j) { j["observation_size"] = 96; });
  reject("unrecognized training contract cannot load",
         [](auto &j) { j["policy_contract"] = "other"; });
  reject("duplicate joint mapping rejected", [](auto &j) { j["joint_ids_map"][22] = 25; });
  reject("fractional joint mapping rejected", [](auto &j) { j["joint_ids_map"][22] = 26.5; });
  reject("incorrect joint order rejected",
         [](auto &j) { std::swap(j["joint_ids_map"][0], j["joint_ids_map"][1]); });
  reject("missing contract field rejected", [](auto &j) { j.erase("policy_contract"); });
  for (const std::string key : {"default_joint_pos", "stiffness", "damping", "action_scale",
                                "joint_lower", "joint_upper"}) {
    reject("short " + key + " rejected", [&](auto &j) { j[key].erase(22); });
    reject("non-numeric " + key + " rejected", [&](auto &j) { j[key][22] = "NaN"; });
  }
  reject("zero stiffness rejected", [](auto &j) { j["stiffness"][0] = 0; });
  reject("excessive stiffness rejected", [](auto &j) { j["stiffness"][0] = 501; });
  reject("negative damping rejected", [](auto &j) { j["damping"][0] = -1; });
  reject("excessive damping rejected", [](auto &j) { j["damping"][0] = 31; });
  reject("short stand stiffness rejected", [](auto &j) { j["stand_stiffness"].erase(22); });
  reject("zero stand stiffness rejected", [](auto &j) { j["stand_stiffness"][0] = 0; });
  reject("excessive stand stiffness rejected", [](auto &j) { j["stand_stiffness"][0] = 501; });
  reject("negative stand damping rejected", [](auto &j) { j["stand_damping"][0] = -1; });
  reject("excessive stand damping rejected", [](auto &j) { j["stand_damping"][0] = 31; });
  reject("half-specified stand gains rejected", [](auto &j) { j.erase("stand_damping"); });
  reject("short stand feedforward rejected", [](auto &j) { j["stand_gravity_ff"].erase(22); });
  reject("excessive stand feedforward rejected", [](auto &j) { j["stand_gravity_ff"][0] = 89; });
  {
    auto legacy = fixture.valid;
    legacy.erase("stand_stiffness");
    legacy.erase("stand_damping");
    legacy.erase("stand_gravity_ff");
    const auto withoutStand = fixture.load(legacy);
    suite.check(withoutStand.standKp == withoutStand.kp && withoutStand.standKd == withoutStand.kd,
                "absent stand gains fall back to the policy contract");
    bool ffZero = true;
    for (float v : withoutStand.standFf)
      ffZero = ffZero && v == 0.f;
    suite.check(ffZero, "absent stand feedforward defaults to zero");
  }
  reject("zero action scale rejected", [](auto &j) { j["action_scale"][0] = 0; });
  reject("inverted limits rejected", [](auto &j) { j["joint_lower"][0] = 4; });
  reject("out-of-limit default posture rejected",
         [](auto &j) { j["default_joint_pos"][22] = 100; });
  reject("mismatched trained timestep rejected", [](auto &j) { j["step_dt"] = .01; });
  reject("mismatched gait period rejected", [](auto &j) { j["gait_period"] = .8; });
  reject("too-fast stand transition rejected", [](auto &j) { j["stand_duration"] = 1; });
  reject("excessive stand duration rejected", [](auto &j) { j["stand_duration"] = 31; });
  reject("excessive deployment command rejected",
         [](auto &j) { j["deployment_command_limits"][0] = .6; });
  reject("negative command bound rejected",
         [](auto &j) { j["deployment_command_limits"][1] = -.1; });
  reject("excessive pitch command bound rejected",
         [](auto &j) { j["deployment_command_limits"][3] = .4; });
  reject("short command limits rejected", [](auto &j) { j["deployment_command_limits"].erase(3); });
  reject("unsafe tilt threshold rejected", [](auto &j) { j["orientation_threshold"] = 1; });
  reject("recovery hysteresis enforced",
         [](auto &j) { j["safety"]["recovery_gravity_threshold"] = .5; });
  reject("disabled velocity protection rejected",
         [](auto &j) { j["safety"]["max_joint_velocity"] = 0; });
  reject("disabled tracking protection rejected",
         [](auto &j) { j["safety"]["max_tracking_error"] = 0; });
  reject("disabled step protection rejected", [](auto &j) { j["safety"]["max_target_step"] = 0; });
  reject("excessive state timeout rejected",
         [](auto &j) { j["safety"]["state_timeout_ms"] = 101; });
  reject("too-short state timeout rejected", [](auto &j) { j["safety"]["state_timeout_ms"] = 19; });
  reject("fractional state timeout rejected",
         [](auto &j) { j["safety"]["state_timeout_ms"] = 20.5; });
  reject("overflowing state timeout rejected",
         [](auto &j) { j["safety"]["state_timeout_ms"] = 4294967396LL; });
  reject("excessive tilt confirmation rejected",
         [](auto &j) { j["safety"]["tilt_confirmation_ms"] = 101; });
  reject("too-short recovery duration rejected",
         [](auto &j) { j["safety"]["recovery_duration_ms"] = 99; });
  reject("empty model path rejected", [](auto &j) { j["model_path"] = ""; });
  suite.throws([&] { g1::Config23::load(fixture.directory / "missing.json"); },
               "missing config rejected");
  {
    std::ofstream output(fixture.path);
    output << "{invalid json}";
  }
  suite.throws([&] { g1::Config23::load(fixture.path); }, "malformed config rejected");
  return suite.result();
}
