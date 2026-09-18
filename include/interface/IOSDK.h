#pragma once
#include "common/g1_23dof.h"
#include "common/gamepad.hpp"
#include "interface/IOInterface.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

class IOSDK : public IOInterface {
public:
  explicit IOSDK(bool use23 = true);
  ~IOSDK() override;
  // Construction is offline; start is the explicit boundary for DDS activity.
  void start();
  void stop();
  void receive(LowlevelState *state) override;
  void send(const LowlevelCmd *cmd) override;
  void sendRecv(const LowlevelCmd *cmd, LowlevelState *state) override {
    receive(state);
    send(cmd);
  }
  bool watchdogFault() const override;
  void resetWatchdog() override;

private:
  void LowStateHandler(const void *message);
  void publishLoop();
  bool activeMotor(std::size_t i) const { return !use23_ || g1::isPolicyMotor(i); }
  bool use23_;
  std::string interface_;
  unsigned domain_ = 1;
  bool started_ = false;
  bool machineMismatch_ = false;
  mutable std::mutex mutex_;
  LowlevelState state_;
  LowlevelCmd command_;
  std::chrono::steady_clock::time_point commandAt_{};
  bool haveCommand_ = false, watchdogLatched_ = false;
  std::uint32_t lastTick_ = 0;
  std::uint8_t modeMachine_ = 0;
  UserCommand heldCommand_ = UserCommand::NONE;
  unitree::common::REMOTE_DATA_RX rx_{};
  unitree::common::Gamepad gamepad_;
  std::atomic<bool> stopping_{false};
  std::thread publisherThread_;
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> subscriber_;
};
