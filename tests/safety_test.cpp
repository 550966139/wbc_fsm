#include "common/safety_supervisor.h"
#include "test_support.h"
#include <limits>

using namespace std::chrono_literals;

int main() {
  TestSuite suite;
  const auto start = control::SafetyClock::time_point{10s};
  const std::array<float, 3> upright{0.f, 0.f, -1.f}, tilted{1.f, 0.f, 0.f};
  control::SafetySupervisor missing;
  suite.check(!missing.checkState(false, start, upright, start), "missing first LowState trips");
  suite.check(missing.latched(), "missing state fault is latched");
  missing.trip("later error");
  suite.check(missing.reason() == "LowState missing or stale", "first fault reason retained");

  control::SafetySupervisor stale;
  suite.check(stale.checkState(true, start, upright, start + 100ms),
              "state at timeout boundary accepted");
  suite.check(!stale.checkState(true, start, upright, start + 101ms), "stale LowState trips");
  control::SafetySupervisor future;
  suite.check(!future.checkState(true, start + 1ms, upright, start),
              "future LowState timestamp trips");
  for (const auto &gravity :
       {std::array<float, 3>{0.f, 0.f, 0.f}, std::array<float, 3>{0.f, 0.f, -2.f},
        std::array<float, 3>{std::numeric_limits<float>::quiet_NaN(), 0.f, -1.f}}) {
    control::SafetySupervisor invalidImu;
    suite.check(!invalidImu.checkState(true, start, gravity, start),
                "invalid projected gravity trips");
  }

  control::SafetySupervisor tilt;
  suite.check(tilt.checkState(true, start, tilted, start), "initial tilt waits for confirmation");
  suite.check(tilt.checkState(true, start + 39ms, tilted, start + 39ms),
              "short tilt does not latch");
  suite.check(tilt.checkState(true, start + 40ms, upright, start + 40ms),
              "upright clears tilt timer");
  suite.check(tilt.checkState(true, start + 41ms, tilted, start + 41ms),
              "second tilt restarts timer");
  suite.check(!tilt.checkState(true, start + 81ms, tilted, start + 81ms), "confirmed tilt latches");
  suite.check(tilt.reason() == "excessive tilt", "tilt diagnostic retained");
  suite.check(!tilt.recover(true, start + 100ms), "start press while tilted cannot recover");
  suite.check(!tilt.checkState(true, start + 101ms, upright, start + 101ms),
              "upright does not auto-unlatch");
  suite.check(!tilt.recover(true, start + 150ms), "short upright interval cannot recover");
  for (auto elapsed = 151ms; elapsed < 601ms; elapsed += 50ms)
    tilt.checkState(true, start + elapsed, upright, start + elapsed);
  suite.check(!tilt.checkState(true, start + 601ms, upright, start + 601ms),
              "healthy state retains latch");
  suite.check(!tilt.recover(false, start + 601ms), "recovery requires a new start press");
  suite.check(tilt.recover(true, start + 601ms), "fresh upright state and new start recover");
  suite.check(!tilt.latched() && tilt.reason().empty(),
              "successful recovery clears reason and latch");

  control::SafetySupervisor staleRecovery;
  staleRecovery.trip("test fault");
  staleRecovery.checkState(true, start, upright, start);
  suite.check(!staleRecovery.recover(true, start + 501ms),
              "stale once-healthy state cannot recover");
  staleRecovery.checkState(true, start + 501ms, upright, start + 501ms);
  suite.check(!staleRecovery.recover(true, start + 501ms),
              "a fresh state after a telemetry gap restarts recovery dwell");

  control::SafetySupervisor interruptedRecovery;
  interruptedRecovery.trip("test fault");
  interruptedRecovery.checkState(true, start, upright, start);
  interruptedRecovery.checkState(false, start, upright, start + 400ms);
  interruptedRecovery.checkState(true, start + 500ms, upright, start + 500ms);
  suite.check(!interruptedRecovery.recover(true, start + 500ms),
              "state loss resets recovery interval");
  return suite.result();
}
