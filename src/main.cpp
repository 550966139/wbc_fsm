#include "control/ControlFrame.h"
#include "control/CtrlComponents.h"
#include "control/config23.h"
#include "interface/IOSDK.h"
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
volatile std::sig_atomic_t keepRunning = 1;
void shutDown(int) { keepRunning = 0; }

void usage() {
  std::cout << "G1 23DoF controller\n"
               "  wbc_fsm [--check] [--config PATH]\n"
               "  wbc_fsm --run [--config PATH] [--interface IFACE] [--domain ID]\n\n"
               "Default: validate config and ONNX without connecting DDS or publishing commands.\n"
               "--run starts DDS in damping; START enters stand, R2+A enters the velocity policy.\n"
               "L2+B latches damping; SELECT or SIGINT exits with a brief damping handoff.\n"
               "After a fault: recover, press START to acknowledge, release, then START to stand.\n"
               "DDS defaults: UNITREE_DDS_IFACE=lo, UNITREE_DDS_DOMAIN=1 (simulation).\n"
               "Hardware requires domain 0 and an explicit robot network interface.\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    bool run = false;
    bool selectedMode = false;
    std::filesystem::path configPath =
        std::filesystem::path(PROJECT_ROOT_DIR) / "config/g1_23dof.json";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        usage();
        return 0;
      }
      if (arg == "--check" || arg == "--run") {
        if (selectedMode)
          throw std::runtime_error("select exactly one of --check or --run");
        run = arg == "--run";
        selectedMode = true;
      } else if (arg == "--config" || arg == "--interface" || arg == "--domain") {
        if (++i >= argc)
          throw std::runtime_error("missing value for " + arg);
        if (arg == "--config")
          configPath = std::filesystem::absolute(argv[i]);
        else if (setenv(arg == "--interface" ? "UNITREE_DDS_IFACE" : "UNITREE_DDS_DOMAIN", argv[i],
                        1) != 0)
          throw std::runtime_error("cannot set DDS option");
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    const auto config = g1::Config23::load(configPath);
    if (!std::filesystem::is_regular_file(config.model))
      throw std::runtime_error("23DoF ONNX model not found: " + config.model.string() +
                               ". Export a matching trained 81-input / 23-output velocity policy; "
                               "the bundled legacy models are incompatible.");

    // Construct and validate the entire policy before any network activity.
    auto ioOwner = std::make_unique<IOSDK>();
    auto *io = ioOwner.get();
    CtrlComponents components(ioOwner.release());
    components.ctrlPlatform = CtrlPlatform::REALROBOT;
    bool running = true;
    components.running = &running;
    components.configure23(config);
    ControlFrame controller(&components);
    if (!run) {
      std::cout << "PASS: 23DoF config and ONNX contract checked; DDS was not started.\n"
                   "This does not establish trained-model provenance or sim-to-real validity.\n";
      return 0;
    }

    std::signal(SIGINT, shutDown);
    std::signal(SIGTERM, shutDown);
    io->start();
    std::cout << "Controller started in damping. Await fresh state before pressing START.\n";
    while (keepRunning && !components.exitFlag)
      controller.run();
    io->stop();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Controller refused to start / stopped: " << error.what() << '\n';
    return 1;
  }
}
