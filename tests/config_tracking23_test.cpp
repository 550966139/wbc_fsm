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
    // The checked-in configuration ships the dual-motion library; the legacy
    // single-motion form is reconstructed from slot 0 so the historical
    // single-motion cases keep testing what they claim.
    legacy = valid;
    legacy["model_path"] = valid.at("motions").at(0).at("model_path");
    legacy["motion_path"] = valid.at("motions").at(0).at("motion_path");
    legacy.erase("motions");
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
  // The legacy single-motion form rewritten as an equivalent two-motion
  // array: motion one keeps the legacy paths, motion two carries its own
  // model/motion and per-motion overrides.
  nlohmann::json motionsValid() const {
    auto multi = legacy;
    const auto model = multi.at("model_path").get<std::string>();
    const auto motion = multi.at("motion_path").get<std::string>();
    multi.erase("model_path");
    multi.erase("motion_path");
    multi["motions"] = nlohmann::json::array(
        {{{"name", "dance-one"}, {"model_path", model}, {"motion_path", motion}},
         {{"name", "dance-two"},
          {"model_path", "model/g1_23dof_tracking/policy2.onnx"},
          {"motion_path", "model/g1_23dof_tracking/motion2.csv"},
          {"safety", {{"max_tracking_error", 6.1}, {"max_target_step", 1.7}}},
          {"entry_gate", {{"entry_max_joint_error", 0.12}, {"entry_max_joint_vel", 0.8}}},
          {"motion_end_target", "policy23"},
          {"handoff_settle_timeout", 2.5},
          {"handoff_entry_max_joint_error", 0.4},
          {"loop_motion", true}}});
    return multi;
  }
  nlohmann::json valid, legacy;
  std::filesystem::path directory, path;
};
} // namespace

int main() {
  TestSuite suite;
  ConfigFixture fixture;
  const auto config = fixture.load(fixture.legacy);
  suite.check(config.motions.size() == 1, "legacy single-motion config loads as one motion");
  suite.check(config.motions[0].name == "tracking", "legacy motion carries the default name");
  suite.check(config.motions[0].model == fixture.directory / "model/g1_23dof_tracking/policy.onnx",
              "tracking model path resolves relative to the configuration project");
  suite.check(config.motions[0].motion == fixture.directory / "model/g1_23dof_tracking/motion.csv",
              "motion path resolves relative to the configuration project");
  suite.near(config.dt, .02f, "tracking configuration uses 50Hz policy");
  suite.near(config.motions[0].entryMaxJointError, .15f, "entry gate joint error threshold");
  suite.near(config.motions[0].entryMaxJointVel, 1.f, "entry gate joint velocity threshold");
  suite.check(config.motions[0].motionEndTarget == g1::MotionEndTarget::Stand,
              "shipped configuration hands motion end off to fixed stand (2026-09-22 decision)");
  suite.near(config.motions[0].handoffSettleTimeout, 3.f,
             "settle-hold window defaults to three seconds");
  suite.near(config.motions[0].handoffEntryMaxJointError, .15f,
             "handoff pose gate defaults to the R2+A entry threshold");
  suite.check(!config.motions[0].loopMotion, "loop_motion defaults to false");
  {
    auto toStand = fixture.legacy;
    toStand["motion_end_target"] = "stand";
    suite.check(fixture.load(toStand).motions[0].motionEndTarget == g1::MotionEndTarget::Stand,
                "motion_end_target stand fallback loads");
    auto absent = fixture.legacy;
    absent.erase("motion_end_target");
    absent.erase("handoff_settle_timeout");
    absent.erase("handoff_entry_max_joint_error");
    const auto defaultsOnly = fixture.load(absent);
    suite.check(defaultsOnly.motions[0].motionEndTarget == g1::MotionEndTarget::Policy23,
                "absent motion_end_target keeps the policy23 default");
    suite.near(defaultsOnly.motions[0].handoffSettleTimeout, 3.f,
               "absent handoff_settle_timeout keeps the default");
    suite.near(defaultsOnly.motions[0].handoffEntryMaxJointError, .15f,
               "absent handoff_entry_max_joint_error keeps the default");
    auto custom = fixture.legacy;
    custom["handoff_settle_timeout"] = 5.5;
    suite.near(fixture.load(custom).motions[0].handoffSettleTimeout, 5.5f,
               "custom settle timeout loads");
    custom["handoff_entry_max_joint_error"] = 0.5;
    suite.near(fixture.load(custom).motions[0].handoffEntryMaxJointError, .5f,
               "custom handoff pose gate loads");
  }
  suite.check(config.defaults.size() == 23, "tracking configuration uses all 23 joints");
  {
    auto looping = fixture.legacy;
    looping["loop_motion"] = true;
    suite.check(fixture.load(looping).motions[0].loopMotion, "loop_motion opt-in loads");
  }
  {
    // The shipped dual-motion baseline (2026-09-30): take6-F10 keeps slot 0
    // with inherited guards, take2 overrides with its own calibration. Slot 1
    // moved take2-F2 -> take2-F17c on 2026-10-12 (user decision: accept the
    // revised bars; guards 1.9/4.3 from F17c replay envelopes, handoff pose
    // gate 0.8 for its 0.672 rad r_shoulder_yaw outro offset).
    const auto shipped = fixture.load(fixture.valid);
    suite.check(shipped.motions.size() == 2,
                "shipped configuration carries the dual-motion library");
    suite.check(shipped.motions[0].name == "take6-F10", "shipped slot 0 is take6-F10 (R2+B)");
    suite.check(shipped.motions[0].model ==
                    fixture.directory / "model/g1_23dof_tracking/policy.onnx",
                "shipped slot 0 keeps the F10 policy");
    suite.near(shipped.motions[0].maxTargetStep, config.motions[0].maxTargetStep,
               "shipped slot 0 inherits the F10 step guard (R2+B behaviour unchanged)");
    suite.near(shipped.motions[0].maxTrackingError, config.motions[0].maxTrackingError,
               "shipped slot 0 inherits the F10 tracking guard");
    suite.check(shipped.motions[0].motionEndTarget == g1::MotionEndTarget::Stand,
                "shipped slot 0 keeps the stand end target");
    suite.check(shipped.motions[1].name == "take2-F17c", "shipped slot 1 is take2-F17c (R2+Y)");
    suite.check(shipped.motions[1].model ==
                    fixture.directory / "model/g1_23dof_tracking/policy_take2f17c_200000.onnx",
                "shipped slot 1 keeps the F17c policy");
    suite.check(shipped.motions[1].motion ==
                    fixture.directory / "model/g1_23dof_tracking/motion_take2f2_115000.csv",
                "shipped slot 1 shares the take2 motion chain with its F2 predecessor");
    suite.near(shipped.motions[1].maxTargetStep, 1.9f,
               "shipped slot 1 carries its replay-calibrated step guard");
    suite.near(shipped.motions[1].maxTrackingError, 4.3f,
               "shipped slot 1 carries its replay-calibrated tracking guard");
    suite.check(shipped.motions[1].motionEndTarget == g1::MotionEndTarget::Policy23,
                "shipped slot 1 hands motion end to the velocity policy (2026-10-10 stand-stable decision)");
    suite.near(shipped.motions[1].handoffEntryMaxJointError, .8f,
               "shipped slot 1 widens the handoff pose gate for the tracking policy's steady-state offset");
  }
  {
    const auto multi = fixture.load(fixture.motionsValid());
    suite.check(multi.motions.size() == 2, "motions array loads both entries");
    suite.check(multi.motions[0].name == "dance-one", "first motion name loads");
    suite.check(multi.motions[0].model == fixture.directory / "model/g1_23dof_tracking/policy.onnx",
                "array entry paths resolve relative to the configuration project");
    suite.check(multi.motions[0].maxTargetStep == config.motions[0].maxTargetStep &&
                    multi.motions[0].maxTrackingError == config.motions[0].maxTrackingError,
                "entry without overrides inherits the top-level guards");
    suite.check(multi.motions[1].name == "dance-two", "second motion name loads");
    suite.near(multi.motions[1].maxTrackingError, 6.1f, "per-motion tracking guard override loads");
    suite.near(multi.motions[1].maxTargetStep, 1.7f, "per-motion step guard override loads");
    suite.near(multi.motions[1].entryMaxJointError, .12f,
               "per-motion entry gate error override loads");
    suite.near(multi.motions[1].entryMaxJointVel, .8f,
               "per-motion entry gate velocity override loads");
    suite.check(multi.motions[1].motionEndTarget == g1::MotionEndTarget::Policy23,
                "per-motion motion_end_target override loads");
    suite.near(multi.motions[1].handoffSettleTimeout, 2.5f,
               "per-motion settle timeout override loads");
    suite.near(multi.motions[1].handoffEntryMaxJointError, .4f,
               "per-motion handoff pose gate override loads");
    suite.near(multi.motions[0].handoffEntryMaxJointError, .15f,
               "entry without overrides inherits the handoff pose gate default");
    suite.check(multi.motions[1].loopMotion, "per-motion loop override loads");
    suite.check(!multi.motions[0].loopMotion, "first entry keeps the top-level loop default");
  }
  auto reject = [&](const std::string &description, auto mutate) {
    auto invalid = fixture.legacy;
    mutate(invalid);
    suite.throws([&] { fixture.load(invalid); }, description);
  };
  auto rejectMulti = [&](const std::string &description, auto mutate) {
    auto invalid = fixture.motionsValid();
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
  reject("zero handoff pose gate rejected",
         [](auto &j) { j["handoff_entry_max_joint_error"] = 0; });
  reject("excessive handoff pose gate rejected",
         [](auto &j) { j["handoff_entry_max_joint_error"] = 1.5; });
  reject("non-numeric handoff pose gate rejected",
         [](auto &j) { j["handoff_entry_max_joint_error"] = "0.5"; });
  reject("empty model path rejected", [](auto &j) { j["model_path"] = ""; });
  reject("empty motion path rejected", [](auto &j) { j["motion_path"] = ""; });
  reject("missing motion path rejected", [](auto &j) { j.erase("motion_path"); });
  rejectMulti("motions array cannot combine with legacy model_path",
              [](auto &j) { j["model_path"] = "model/g1_23dof_tracking/policy.onnx"; });
  rejectMulti("motions array cannot combine with legacy motion_path",
              [](auto &j) { j["motion_path"] = "model/g1_23dof_tracking/motion.csv"; });
  rejectMulti("empty motions array rejected",
              [](auto &j) { j["motions"] = nlohmann::json::array(); });
  rejectMulti("non-array motions rejected", [](auto &j) { j["motions"] = 7; });
  rejectMulti("more than four motions rejected", [&](auto &j) {
    auto extra = j["motions"][0];
    for (int i = 0; i < 4; ++i) {
      extra["name"] = "dance-extra" + std::to_string(i);
      j["motions"].push_back(extra);
    }
  });
  rejectMulti("duplicate motion names rejected",
              [](auto &j) { j["motions"][1]["name"] = j["motions"][0]["name"]; });
  rejectMulti("missing motion name rejected", [](auto &j) { j["motions"][1].erase("name"); });
  rejectMulti("empty motion name rejected", [](auto &j) { j["motions"][1]["name"] = ""; });
  rejectMulti("non-string motion name rejected", [](auto &j) { j["motions"][1]["name"] = 5; });
  rejectMulti("missing array entry model path rejected",
              [](auto &j) { j["motions"][1].erase("model_path"); });
  rejectMulti("empty array entry motion path rejected",
              [](auto &j) { j["motions"][1]["motion_path"] = ""; });
  rejectMulti("per-motion disabled guard rejected",
              [](auto &j) { j["motions"][1]["safety"]["max_target_step"] = 0; });
  rejectMulti("per-motion excessive entry gate rejected",
              [](auto &j) { j["motions"][1]["entry_gate"]["entry_max_joint_error"] = 2; });
  rejectMulti("per-motion unknown end target rejected",
              [](auto &j) { j["motions"][1]["motion_end_target"] = "loco"; });
  rejectMulti("per-motion zero settle timeout rejected",
              [](auto &j) { j["motions"][1]["handoff_settle_timeout"] = 0; });
  rejectMulti("per-motion non-boolean loop rejected",
              [](auto &j) { j["motions"][1]["loop_motion"] = 1; });
  suite.throws([&] { g1::ConfigTracking23::load(fixture.directory / "missing.json"); },
               "missing tracking config rejected");
  {
    std::ofstream output(fixture.path);
    output << "{invalid json}";
  }
  suite.throws([&] { g1::ConfigTracking23::load(fixture.path); }, "malformed config rejected");
  return suite.result();
}
