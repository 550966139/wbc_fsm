#include "interface/IOSDK.h"
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <net/if.h>
#include <stdexcept>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
uint32_t crc32_core(uint32_t *ptr, uint32_t len) {
  unsigned int xbit = 0;
  unsigned int data = 0;
  unsigned int CRC32 = 0xFFFFFFFF;
  const unsigned int dwPolynomial = 0x04c11db7;

  for (unsigned int i = 0; i < len; i++) {
    xbit = 1u << 31;
    data = ptr[i];
    for (unsigned int bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }

      if (data & xbit)
        CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }

  return CRC32;
}

IOSDK::IOSDK(bool use23) : use23_(use23) {
  const char *iface = std::getenv("UNITREE_DDS_IFACE");
  if (!iface || !*iface)
    iface = "lo";
  const char *raw = std::getenv("UNITREE_DDS_DOMAIN");
  unsigned domain = 1;
  if (raw) {
    auto result = std::from_chars(raw, raw + std::strlen(raw), domain);
    if (result.ec != std::errc{} || result.ptr != raw + std::strlen(raw) || domain > 232)
      throw std::runtime_error("UNITREE_DDS_DOMAIN must be an integer in [0,232]");
  }
  if (domain == 0 && std::string(iface) == "lo")
    throw std::runtime_error("real domain 0 requires an explicit robot DDS interface");
  interface_ = iface;
  domain_ = domain;
}
void IOSDK::start() {
  if (started_)
    throw std::runtime_error("DDS already started");
  if (if_nametoindex(interface_.c_str()) == 0)
    throw std::runtime_error("DDS interface does not exist: " + interface_);
  std::cout << "DDS interface " << interface_ << ", domain " << domain_ << std::endl;
  ChannelFactory::Instance()->Init(domain_, interface_);
  if (domain_ == 0) {
    unitree::robot::b2::MotionSwitcherClient switcher;
    switcher.SetTimeout(5.f);
    switcher.Init();
    std::string form, name;
    if (switcher.CheckMode(form, name) != 0)
      throw std::runtime_error("cannot verify robot motion service; no LowCmd publisher started");
    if (!name.empty())
      throw std::runtime_error("robot motion service is active (" + name +
                               "); enter development mode before running this controller");
  }

  // Listen before creating a publisher. This catches active competing controllers;
  // it is not a DDS-wide exclusive lock, so operators must prevent later starts.
  auto observedCommands = std::make_shared<std::atomic<bool>>(false);
  auto commandMonitor = std::make_unique<ChannelSubscriber<LowCmd_>>("rt/lowcmd");
  commandMonitor->InitChannel([observedCommands](const void *) { *observedCommands = true; }, 1);
  subscriber_.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
  subscriber_->InitChannel(std::bind(&IOSDK::LowStateHandler, this, std::placeholders::_1), 1);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  bool ready = false;
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lock(mutex_);
    if (machineMismatch_)
      throw std::runtime_error(
          "LowState mode_machine is not G1 23DoF (expected 4); no commands sent");
    ready = state_.received &&
            std::chrono::steady_clock::now() - state_.receivedAt < std::chrono::milliseconds(100);
  } while (!ready && std::chrono::steady_clock::now() < deadline);
  if (!ready)
    throw std::runtime_error("no fresh valid LowState within 3 seconds; no commands sent");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  if (*observedCommands)
    throw std::runtime_error("another rt/lowcmd controller is active; no commands sent");
  commandMonitor->CloseChannel();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (machineMismatch_ ||
        std::chrono::steady_clock::now() - state_.receivedAt > std::chrono::milliseconds(100))
      throw std::runtime_error("LowState lost or machine changed during startup; no commands sent");
    state_.userCmd = UserCommand::NONE;
  }
  publisher_.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
  publisher_->InitChannel();
  stopping_ = false;
  publisherThread_ = std::thread(&IOSDK::publishLoop, this);
  started_ = true;
}
IOSDK::~IOSDK() { stop(); }
void IOSDK::stop() {
  if (started_) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      watchdogLatched_ = true;
    }
    // Best-effort damping handoff; transport/process loss still needs robot-side protection.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  stopping_ = true;
  if (publisherThread_.joinable())
    publisherThread_.join();
  if (subscriber_)
    subscriber_->CloseChannel();
  subscriber_.reset();
  if (publisher_)
    publisher_->CloseChannel();
  publisher_.reset();
  started_ = false;
}
void IOSDK::receive(LowlevelState *state) {
  std::lock_guard<std::mutex> lock(mutex_);
  *state = state_;
  state_.userCmd = UserCommand::NONE; // events consumed once, not sticky commands
}
void IOSDK::send(const LowlevelCmd *cmd) {
  std::lock_guard<std::mutex> lock(mutex_);
  command_ = *cmd;
  commandAt_ = std::chrono::steady_clock::now();
  haveCommand_ = true;
}
bool IOSDK::watchdogFault() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return watchdogLatched_;
}
void IOSDK::resetWatchdog() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (machineMismatch_ || !state_.received ||
      std::chrono::steady_clock::now() - state_.receivedAt > std::chrono::milliseconds(100))
    return;
  // Recovery cannot replay an old active command.
  command_ = LowlevelCmd{};
  for (std::size_t i = 0; i < g1::kMotorCount; ++i)
    if (activeMotor(i))
      command_.motorCmd[i].Kd = 3.f;
  commandAt_ = std::chrono::steady_clock::now();
  haveCommand_ = true;
  watchdogLatched_ = false;
}
void IOSDK::publishLoop() {
  using Clock = std::chrono::steady_clock;
  while (!stopping_) {
    const auto cycleStarted = Clock::now();
    LowCmd_ message{};
    bool publish = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = Clock::now();
      publish = state_.received && !machineMismatch_;
      if (publish) {
        if (now - state_.receivedAt > std::chrono::milliseconds(100) ||
            (haveCommand_ && now - commandAt_ > std::chrono::milliseconds(100)))
          watchdogLatched_ = true;
        bool damping = watchdogLatched_ || !haveCommand_;
        for (std::size_t i = 0; i < g1::kMotorCount; ++i)
          if (activeMotor(i)) {
            const auto &c = command_.motorCmd[i];
            if (!std::isfinite(c.q) || !std::isfinite(c.dq) || !std::isfinite(c.tau) ||
                !std::isfinite(c.Kp) || !std::isfinite(c.Kd) || c.Kp < 0 || c.Kd < 0) {
              watchdogLatched_ = true;
              damping = true;
            }
          }
        message.mode_pr() = 0;
        message.mode_machine() = modeMachine_;
        for (std::size_t i = 0; i < g1::kMotorCount; ++i) {
          auto &out = message.motor_cmd().at(i);
          if (!activeMotor(i)) {
            out.mode() = 0;
            continue;
          }
          out.mode() = 1;
          const auto &c = command_.motorCmd[i];
          out.q() = damping ? 0 : c.q;
          out.dq() = damping ? 0 : c.dq;
          out.tau() = damping ? 0 : c.tau;
          out.kp() = damping ? 0 : c.Kp;
          out.kd() = damping ? 3.f : c.Kd;
        }
      }
    }
    if (publish) {
      message.crc() =
          crc32_core(reinterpret_cast<uint32_t *>(&message), (sizeof(message) >> 2) - 1);
      try {
        if (!publisher_->Write(message)) {
          std::lock_guard<std::mutex> lock(mutex_);
          watchdogLatched_ = true;
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        watchdogLatched_ = true;
      }
    }
    std::this_thread::sleep_until(cycleStarted + std::chrono::milliseconds(2));
  }
}
void IOSDK::LowStateHandler(const void *message) {
  LowState_ raw = *static_cast<const LowState_ *>(message);
  if (raw.crc() != crc32_core(reinterpret_cast<uint32_t *>(&raw), (sizeof(raw) >> 2) - 1))
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (use23_ && raw.mode_machine() != 4) {
    machineMismatch_ = true;
    watchdogLatched_ = true;
    return;
  }
  if (state_.received) {
    // Ignore duplicates and delayed/reordered packets, including across uint32 wrap.
    // A robot reboot requires restarting this controller instead of reusing old state.
    const std::uint32_t advance = raw.tick() - lastTick_;
    if (advance == 0 || advance > 0x7fffffffu)
      return;
  }
  if (state_.received && modeMachine_ != raw.mode_machine())
    watchdogLatched_ = true;
  lastTick_ = raw.tick();
  modeMachine_ = raw.mode_machine();
  for (std::size_t i = 0; i < g1::kMotorCount; ++i) {
    state_.motorState[i].q = raw.motor_state()[i].q();
    state_.motorState[i].dq = raw.motor_state()[i].dq();
  }
  for (int i = 0; i < 4; ++i)
    state_.imu.quaternion[i] = raw.imu_state().quaternion()[i];
  for (int i = 0; i < 3; ++i) {
    state_.imu.gyroscope[i] = raw.imu_state().gyroscope()[i];
    state_.imu.accelerometer[i] = raw.imu_state().accelerometer()[i];
  }
  std::memcpy(rx_.buff, &raw.wireless_remote()[0], 40);
  gamepad_.update(rx_.RF_RX);
  UserCommand cmd = UserCommand::NONE;
  if (gamepad_.start.pressed)
    cmd = UserCommand::START;
  if (gamepad_.R2.pressed && gamepad_.A.pressed)
    cmd = UserCommand::R2_A;
  if (!use23_) {
    if (gamepad_.R1.pressed)
      cmd = UserCommand::R1;
    if (gamepad_.R2.pressed)
      cmd = UserCommand::R2;
    if (gamepad_.L2.pressed)
      cmd = UserCommand::L2;
    if (gamepad_.R2.pressed && gamepad_.A.pressed)
      cmd = UserCommand::R2_A;
    if (gamepad_.R2.pressed && gamepad_.B.pressed)
      cmd = UserCommand::R2_B;
    if (gamepad_.R2.pressed && gamepad_.up.pressed)
      cmd = UserCommand::R2_UP;
    if (gamepad_.R2.pressed && gamepad_.down.pressed)
      cmd = UserCommand::R2_DOWN;
    if (gamepad_.R1.pressed && gamepad_.up.pressed)
      cmd = UserCommand::R1_UP;
    if (gamepad_.R1.pressed && gamepad_.left.pressed)
      cmd = UserCommand::R1_LEFT;
    if (gamepad_.R1.pressed && gamepad_.right.pressed)
      cmd = UserCommand::R1_RIGHT;
  }
  // Stop commands have priority and also gate the independent publisher.
  if (gamepad_.L2.pressed && gamepad_.B.pressed)
    cmd = UserCommand::L2_B;
  if (gamepad_.select.pressed)
    cmd = UserCommand::SELECT;
  if (cmd == UserCommand::SELECT || cmd == UserCommand::L2_B)
    watchdogLatched_ = true;
  if (cmd != heldCommand_ && cmd != UserCommand::NONE) {
    if (state_.userCmd != UserCommand::SELECT && state_.userCmd != UserCommand::L2_B)
      state_.userCmd = cmd;
    if (cmd == UserCommand::SELECT)
      state_.userCmd = cmd;
  }
  heldCommand_ = cmd;
  state_.userValue.lx = -gamepad_.lx;
  state_.userValue.ly = gamepad_.ly;
  state_.userValue.rx = -gamepad_.rx;
  state_.userValue.ry = gamepad_.ry;
  state_.received = true;
  state_.receivedAt = std::chrono::steady_clock::now();
  ++state_.sequence;
}
