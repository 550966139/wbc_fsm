#include "control/config_tracking23.h"
#include "test_support.h"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace {
class ConfigFixture {
public:
  ConfigFixture() {
    std::ifstream input(std::filesystem::path(PROJECT_ROOT_DIR) / "config/g1_23dof_tracking.json");
    if (!input)
      throw std::runtime_error("cannot read checked-in test configuration");
    input >> valid;
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
                ("g1-config-tracking23-test-" + std::to_string(unique));
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("cannot create unique test directory");
    std::filesystem::create_directory(directory / "config");
    path = directory / "config/g1_23dof_tracking.json";
  }
  ~ConfigFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
  g1::ConfigTracking23 load(const nlohmann::json &value) const {
    {
      std::ofstream output(path);
      if (!(output << value.dump(2)))
        throw std::runtime_error("cannot write test configuration");
    }
    return g1::ConfigTracking23::load(path);
  }
  nlohmann::json valid;
  std::filesystem::path directory, path;
};
} // namespace

int main() {
  TestSuite suite;
  ConfigFixture fixture;
  const auto config = fixture.load(fixture.valid);
  suite.check(config.model == fixture.directory / "model/g1_23dof_tracking/policy.onnx",
              "tracking model path resolves relative to the configuration project");
  suite.check(config.motion == fixture.directory / "model/g1_23dof_tracking/motion.csv",
              "motion path resolves relative to the configuration project");
  suite.near(config.dt, .02f, "tracking configuration uses 50Hz policy");
  suite.near(config.entryMaxJointError, .15f, "entry gate joint error threshold");
  suite.near(config.entryMaxJointVel, 1.f, "entry gate joint velocity threshold");
  suite.check(config.motionEndTarget == g1::MotionEndTarget::Stand,
              "shipped configuration hands motion end off to fixed stand (2026-09-22 decision)");
  suite.near(config.handoffSettleTimeout, 3.f, "settle-hold window defaults to three seconds");
  suite.check(!config.loopMotion, "loop_motion defaults to false");
  {
    auto toStand = fixture.valid;
    toStand["motion_end_target"] = "stand";
    suite.check(fixture.load(toStand).motionEndTarget == g1::MotionEndTarget::Stand,
                "motion_end_target stand fallback loads");
    auto absent = fixture.valid;
    absent.erase("motion_end_target");
    absent.erase("handoff_settle_timeout");
    const auto defaultsOnly = fixture.load(absent);
    suite.check(defaultsOnly.motionEndTarget == g1::MotionEndTarget::Policy23,
                "absent motion_end_target keeps the policy23 default");
    suite.near(defaultsOnly.handoffSettleTimeout, 3.f,
               "absent handoff_settle_timeout keeps the default");
    auto custom = fixture.valid;
    custom["handoff_settle_timeout"] = 5.5;
    suite.near(fixture.load(custom).handoffSettleTimeout, 5.5f, "custom settle timeout loads");
  }
  suite.check(config.defaults.size() == 23, "tracking configuration uses all 23 joints");
  {
    auto looping = fixture.valid;
    looping["loop_motion"] = true;
    suite.check(fixture.load(looping).loopMotion, "loop_motion opt-in loads");
  }
  auto reject = [&](const std::string &description, auto mutate) {
    auto invalid = fixture.valid;
    mutate(invalid);
    suite.throws([&] { fixture.load(invalid); }, description);
  };
  reject("velocity contract cannot load as tracking",
         [](auto &j) { j["policy_contract"] = "unitree_rl_mjlab_g1_23dof_velocity_v1"; });
  reject("29DoF tracking policy cannot load", [](auto &j) { j["policy_dof"] = 29; });
  reject("motor slot count cannot change", [](auto &j) { j["motor_count"] = 23; });
  reject("velocity observation width cannot load", [](auto &j) { j["observation_size"] = 81; });
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
  reject("zero action scale rejected", [](auto &j) { j["action_scale"][0] = 0; });
  reject("inverted limits rejected", [](auto &j) { j["joint_lower"][0] = 4; });
  reject("out-of-limit default posture rejected",
         [](auto &j) { j["default_joint_pos"][22] = 100; });
  reject("mismatched trained timestep rejected", [](auto &j) { j["step_dt"] = .01; });
  reject("disabled tracking protection rejected",
         [](auto &j) { j["safety"]["max_tracking_error"] = 0; });
  reject("disabled step protection rejected", [](auto &j) { j["safety"]["max_target_step"] = 0; });
  reject("negative tracking protection rejected",
         [](auto &j) { j["safety"]["max_tracking_error"] = -.5; });
  reject("missing safety object rejected", [](auto &j) { j.erase("safety"); });
  reject("zero entry error gate rejected",
         [](auto &j) { j["entry_gate"]["entry_max_joint_error"] = 0; });
  reject("excessive entry error gate rejected",
         [](auto &j) { j["entry_gate"]["entry_max_joint_error"] = 1.5; });
  reject("negative entry velocity gate rejected",
         [](auto &j) { j["entry_gate"]["entry_max_joint_vel"] = -.1; });
  reject("missing entry gate rejected", [](auto &j) { j.erase("entry_gate"); });
  reject("non-boolean loop_motion rejected", [](auto &j) { j["loop_motion"] = "yes"; });
  reject("unknown motion end target rejected", [](auto &j) { j["motion_end_target"] = "loco"; });
  reject("non-string motion end target rejected", [](auto &j) { j["motion_end_target"] = 1; });
  reject("zero settle timeout rejected", [](auto &j) { j["handoff_settle_timeout"] = 0; });
  reject("negative settle timeout rejected", [](auto &j) { j["handoff_settle_timeout"] = -1; });
  reject("excessive settle timeout rejected", [](auto &j) { j["handoff_settle_timeout"] = 31; });
  reject("non-numeric settle timeout rejected", [](auto &j) { j["handoff_settle_timeout"] = "3"; });
  reject("empty model path rejected", [](auto &j) { j["model_path"] = ""; });
  reject("empty motion path rejected", [](auto &j) { j["motion_path"] = ""; });
  reject("missing motion path rejected", [](auto &j) { j.erase("motion_path"); });
  suite.throws([&] { g1::ConfigTracking23::load(fixture.directory / "missing.json"); },
               "missing tracking config rejected");
  {
    std::ofstream output(fixture.path);
    output << "{invalid json}";
  }
  suite.throws([&] { g1::ConfigTracking23::load(fixture.path); }, "malformed config rejected");
  return suite.result();
}
