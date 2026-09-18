#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <string>

// Checks remain active in Release builds, unlike the C assert macro.
class TestSuite {
public:
  void check(bool passed, const std::string &description) {
    ++checks_;
    if (!passed) {
      ++failures_;
      std::cerr << "FAIL: " << description << '\n';
    }
  }

  void near(float actual, float expected, const std::string &description,
            float tolerance = 1.e-5f) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, description);
  }

  template <typename Callable> void throws(Callable action, const std::string &description) {
    bool rejected = false;
    try {
      action();
    } catch (const std::exception &) {
      rejected = true;
    }
    check(rejected, description);
  }

  int result() const {
    std::cout << checks_ << " checks, " << failures_ << " failures\n";
    return failures_ == 0 ? 0 : 1;
  }

private:
  unsigned checks_ = 0;
  unsigned failures_ = 0;
};
