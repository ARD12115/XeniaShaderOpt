#pragma once
// =============================================================================
//  frame_pacer.h
//  Frame Pacing Stabilizer — Header
//  Integrates into: src/xenia/gpu/
//
//  What this solves:
//    - Shader compile spikes cause uneven frame delivery → judder even at
//      average 60 FPS. The 60FPS patch enables deltatime but doesn't enforce
//      even 16.67ms cadence at the presentation layer.
//
//  Improvements over v1:
//    - KalmanFilter class: predicts next frame delivery time, feeds adaptive
//      sleep/spin split to eliminate both oversleep and busy-spin waste.
//    - FrameStats struct: P50/P95/P99 percentile computation from history.
//    - StutterEvent log: timestamped record of every frame exceeding 2×target.
//    - VRR hooks: SetVRRRange() / IsVRRActive() for future G-Sync / FreeSync.
//    - AdaptivePacer: dynamically adjusts coarse-sleep / spin-residual ratio
//      based on observed OS timer granularity.
//    - ScopedFrame RAII guard: auto-calls BeginFrame/EndFrame.
// =============================================================================

#ifndef XENIA_GPU_FRAME_PACER_H_
#define XENIA_GPU_FRAME_PACER_H_

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace xe {
namespace gpu {

// =============================================================================
//  Constants
// =============================================================================

inline constexpr int kFrameHistorySize = 128; // rolling history depth
inline constexpr int kStutterLogSize = 32;    // last N stutter events
inline constexpr double kDefaultTargetFPS = 60.0;
inline constexpr double kStutterThreshMult = 2.0;    // >2× target = stutter
inline constexpr double kSpinResidualNs = 100'000.0; // 100µs default spin

// =============================================================================
//  KalmanFilter  — 1D scalar, predicts next frame delivery time
//
//  State:    x = estimated frame interval (ns)
//  Predict:  x_pred = x  (constant model)
//  Update:   x = x_pred + K*(measurement - x_pred)
// =============================================================================

class KalmanFilter {
public:
  explicit KalmanFilter(double initial_estimate_ns, double process_noise = 1e4,
                        double measurement_noise = 1e6);

  // Feed measured frame interval; returns updated estimate (ns).
  double Update(double measured_ns);

  double Estimate() const { return x_; }
  double Uncertainty() const { return p_; }
  void Reset(double initial_ns);

private:
  double x_; // state estimate (ns)
  double p_; // estimate uncertainty (covariance)
  double q_; // process noise covariance
  double r_; // measurement noise covariance
};

// =============================================================================
//  FrameStats  — percentile analysis over rolling history
// =============================================================================

struct FrameStats {
  double avg_fps;
  double avg_frame_ms;
  double min_frame_ms;
  double max_frame_ms;
  double p50_ms; // median
  double p95_ms;
  double p99_ms;
  double stddev_ms;
  uint64_t stutter_count; // frames > kStutterThreshMult × target
  uint64_t total_frames;
};

// =============================================================================
//  StutterEvent  — timestamped record of a bad frame
// =============================================================================

struct StutterEvent {
  uint64_t frame_index;
  double frame_ms;     // actual frame time
  double target_ms;    // what it should have been
  double overshoot_ms; // frame_ms - target_ms
};

// =============================================================================
//  VRRConfig  — Variable Refresh Rate integration hooks
// =============================================================================

struct VRRConfig {
  bool enabled = false;
  double min_fps = 48.0;
  double max_fps = 165.0;
  // Callback invoked when pacer wants to signal a present to VRR runtime
  std::function<void(double frame_ms)> present_callback = nullptr;
};

// =============================================================================
//  FramePacer  — main public API
// =============================================================================

class FramePacer {
public:
  explicit FramePacer(double target_fps = kDefaultTargetFPS);

  // Non-copyable
  FramePacer(const FramePacer &) = delete;
  FramePacer &operator=(const FramePacer &) = delete;

  // ── Per-frame API (called from d3d12_command_processor.cc) ───────────────

  // Call at start of render. Returns frame budget in ms.
  double BeginFrame();

  // Call after render, before Present(). Sleeps to target cadence.
  // Returns actual frame time in ms.
  double EndFrame();

  // ── Stats ─────────────────────────────────────────────────────────────────

  FrameStats GetStats() const;
  double AverageFPS() const;
  double FrameTimeVarianceMs() const;
  bool IsStuttering() const;

  // Retrieve last N stutter events (up to kStutterLogSize)
  const std::vector<StutterEvent> &StutterLog() const { return stutter_log_; }

  void PrintStats() const;

  // ── Configuration ─────────────────────────────────────────────────────────

  void SetTargetFPS(double fps);
  void SetVRR(VRRConfig cfg) { vrr_ = cfg; }
  bool IsVRRActive() const { return vrr_.enabled; }

  // Override spin residual (ns). 0 = pure sleep, no spin.
  void SetSpinResidualNs(double ns) { spin_residual_ns_ = ns; }

  // Kalman tuning
  void SetKalmanNoise(double process_noise, double measurement_noise);

  // Reset all history and timing state
  void Reset();

  // ── RAII guard ────────────────────────────────────────────────────────────

  class ScopedFrame {
  public:
    explicit ScopedFrame(FramePacer &p) : pacer_(p) { pacer_.BeginFrame(); }
    ~ScopedFrame() { pacer_.EndFrame(); }
    ScopedFrame(const ScopedFrame &) = delete;
    ScopedFrame &operator=(const ScopedFrame &) = delete;

  private:
    FramePacer &pacer_;
  };

private:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;
  using Ns = std::chrono::nanoseconds;

  // Adaptive sleep: coarse OS sleep + precision spin
  void SleepUntil(TimePoint deadline);

  // Record frame and log stutter if needed
  void RecordFrame(int64_t actual_ns);

  // Compute percentile from sorted copy of history (ms)
  double Percentile(double p) const;

  double target_fps_;
  int64_t target_frame_ns_;
  double spin_residual_ns_ = kSpinResidualNs;

  TimePoint last_present_time_;
  TimePoint frame_start_;
  uint64_t frame_index_ = 0;

  std::array<int64_t, kFrameHistorySize> frame_history_{};

  KalmanFilter kalman_;
  VRRConfig vrr_;
  std::vector<StutterEvent> stutter_log_;

  // Adaptive OS timer granularity estimate (ns)
  double os_sleep_granularity_ns_ = 1'000'000.0; // 1ms default assumption
};

} // namespace gpu
} // namespace xe

#endif // XENIA_GPU_FRAME_PACER_H_
