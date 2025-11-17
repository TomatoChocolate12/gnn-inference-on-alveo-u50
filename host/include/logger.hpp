#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

enum class LogLevel { kInfo, kWarn, kError, kDebug };

class Logger {
 public:
  static Logger &Instance() {
    static Logger instance;
    return instance;
  }

  void set_level(LogLevel lvl) { level_ = lvl; }

  void log(LogLevel lvl, const std::string &msg) {
    if (lvl < level_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    std::ostringstream oss;
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&tt);
    oss << std::put_time(&tm, "%F %T") << " [" << level_to_string(lvl)
        << "] " << msg << '\n';
    std::cerr << oss.str();
  }

 private:
  Logger() = default;
  std::mutex mu_;
  LogLevel level_ = LogLevel::kInfo;

  static const char *level_to_string(LogLevel lvl) {
    switch (lvl) {
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kError:
      return "ERROR";
    case LogLevel::kDebug:
      return "DEBUG";
    }
    return "INFO";
  }
};

inline void LogInfo(const std::string &msg) { Logger::Instance().log(LogLevel::kInfo, msg); }
inline void LogWarn(const std::string &msg) { Logger::Instance().log(LogLevel::kWarn, msg); }
inline void LogError(const std::string &msg) { Logger::Instance().log(LogLevel::kError, msg); }
inline void LogDebug(const std::string &msg) { Logger::Instance().log(LogLevel::kDebug, msg); }
