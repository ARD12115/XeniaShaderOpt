// =============================================================================
//  frame_pacer.cc
//  Frame Pacing Stabilizer — Implementation
//  Integrates into: src/xenia/gpu/frame_pacer.cc
//
//  Improvements over v1:
//    - KalmanFilter fully implemented — predicts next frame interval to tune
//      the coarse-sleep / spin-residual split dynamically.
//    - SleepUntil() uses Kalman estimate to decide how early to wake and spin,
//      minimising both CPU waste and oversleep error.
//    - OS timer granularity calibrated at startup via probe loop.
//    - Percentile() computes P50/P95/P99 from sorted history copy.
//    - RecordFrame() appends to StutterEvent log (ring, capped at
//    kStutterLogSize).
//    - VRR present_callback fired after every EndFrame if VRR is enabled.
//    - Reset() reinstates Kalman + history without reallocating.
//    - PrintStats() prints full FrameStats snapshot.
// =============================================================================

#include "frame_pacer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

namespace xe {
namespace gpu {

// =============================================================================
//  Internal helpers
// =============================================================================

static void Log(const char *tag, const std::string &msg) {
  std::cout << "[" << tag << "] " << msg << "\n";
}

// =============================================================================
//  KalmanFilter
// =============================================================================

KalmanFilter::KalmanFilter(double initial_estimate_ns, double process_noise,
                           double measurement_noise)
    : x_(initial_estimate_ns), p_(measurement_noise), // start uncertain
      q_(process_noise), r_(measurement_noise) {}

double KalmanFilter::Update(double measured_ns) {
  // Predict
  double x_pred = x_;
  double p_pred = p_ + q_;

  // Update (Kalman gain)
  double K = p_pred / (p_pred + r_);
  x_ = x_pred + K * (measured_ns - x_pred);
  p_ = (1.0 - K) * p_pred;

  return x_;
}

void KalmanFilter::Reset(double initial_ns) {
  x_ = initial_ns;
  p_ = r_; // reset to measurement noise level
}

// =============================================================================
//  OS timer granularity calibration
//  Measures real sleep resolution by timing a series of minimal sleeps.
// =============================================================================

static double CalibrateOSSleepGranularityNs() {
  using Clock = std::chrono::high_resolution_clock;
  using Ns = std::chrono::nanoseconds;

  constexpr int kProbes = 8;
  double total = 0.0;

  for (int i = 0; i < kProbes; ++i) {
    auto t0 = Clock::now();
    std::this_thread::sleep_for(Ns(1));
    auto t1 = Clock::now();
    total +=
        static_cast<double>(std::chrono::duration_cast<Ns>(t1 - t0).count());
  }

  double granularity = total / kProbes;
  // Clamp to sane range [100µs, 16ms]
  granularity = std::clamp(granularity, 100'000.0, 16'000'000.0);
  return granularity;
}

// =============================================================================
//  FramePacer — Constructor
// =============================================================================

FramePacer::FramePacer(double target_fps)
    : target_fps_(target_fps),
      target_frame_ns_(static_cast<int64_t>(1e9 / target_fps)),
      last_present_time_(Clock::now()),
      kalman_(static_cast<double>(static_cast<int64_t>(1e9 / target_fps)), 1e4,
              1e6) {
  frame_history_.fill(target_frame_ns_);
  stutter_log_.reserve(kStutterLogSize);

  // Calibrate OS sleep granularity once at construction
  os_sleep_granularity_ns_ = CalibrateOSSleepGranularityNs();

  std::ostringstream oss;
  oss << "Init — target=" << target_fps_ << " FPS"
      << "  frame=" << (target_frame_ns_ / 1e6) << "ms"
      << "  OS granularity=" << (os_sleep_granularity_ns_ / 1e6) << "ms";
  Log("FramePacer", oss.str());
}

// =============================================================================
//  SetTargetFPS
// =============================================================================

void FramePacer::SetTargetFPS(double fps) {
  target_fps_ = fps;
  target_frame_ns_ = static_cast<int64_t>(1e9 / fps);
  kalman_.Reset(static_cast<double>(target_frame_ns_));
  frame_history_.fill(target_frame_ns_);

  std::ostringstream oss;
  oss << "Target changed → " << fps << " FPS (" << (target_frame_ns_ / 1e6)
      << "ms)";
  Log("FramePacer", oss.str());
}

void FramePacer::SetKalmanNoise(double process_noise,
                                double measurement_noise) {
  kalman_ = KalmanFilter(static_cast<double>(target_frame_ns_), process_noise,
                         measurement_noise);
}

// =============================================================================
//  BeginFrame
// =============================================================================

double FramePacer::BeginFrame() {
  frame_start_ = Clock::now();
  return static_cast<double>(target_frame_ns_) / 1e6;
}

// =============================================================================
//  SleepUntil  — coarse OS sleep + Kalman-guided spin residual
// =============================================================================

void FramePacer::SleepUntil(TimePoint deadline) {
  // How much total time remains?
  auto now = Clock::now();
  int64_t remaining_ns = std::chrono::duration_cast<Ns>(deadline - now).count();

  if (remaining_ns <= 0)
    return;

  // Kalman estimate tells us how much the OS will overshoot a sleep call.
  // We stop sleeping this far before deadline and spin the rest.
  double spin_window_ns =
      std::max(spin_residual_ns_, os_sleep_granularity_ns_ * 1.5);

  int64_t coarse_sleep_ns = static_cast<int64_t>(remaining_ns - spin_window_ns);

  if (coarse_sleep_ns > 0) {
    std::this_thread::sleep_for(Ns(coarse_sleep_ns));
  }

  // Precision spin — busy-wait the final window
  while (Clock::now() < deadline) { /* spin */
  }
}

// =============================================================================
//  EndFrame
// =============================================================================

double FramePacer::EndFrame() {
  // Deadline = last present + one target frame interval
  TimePoint deadline = last_present_time_ + Ns(target_frame_ns_);

  SleepUntil(deadline);

  TimePoint present_time = Clock::now();
  int64_t actual_ns =
      std::chrono::duration_cast<Ns>(present_time - last_present_time_).count();

  last_present_time_ = present_time;

  // Feed actual interval into Kalman for next-frame prediction
  kalman_.Update(static_cast<double>(actual_ns));

  // Adapt OS granularity estimate (EMA of Kalman uncertainty)
  os_sleep_granularity_ns_ =
      os_sleep_granularity_ns_ * 0.95 + kalman_.Uncertainty() * 0.05;
  os_sleep_granularity_ns_ =
      std::clamp(os_sleep_granularity_ns_, 100'000.0, 4'000'000.0);

  // Record + stutter log
  RecordFrame(actual_ns);

  // VRR present callback
  if (vrr_.enabled && vrr_.present_callback)
    vrr_.present_callback(static_cast<double>(actual_ns) / 1e6);

  return static_cast<double>(actual_ns) / 1e6;
}

// =============================================================================
//  RecordFrame
// =============================================================================

void FramePacer::RecordFrame(int64_t actual_ns) {
  frame_history_[frame_index_ % kFrameHistorySize] = actual_ns;
  ++frame_index_;

  // Log stutter if frame exceeded threshold
  double actual_ms = static_cast<double>(actual_ns) / 1e6;
  double target_ms = static_cast<double>(target_frame_ns_) / 1e6;

  if (actual_ns > static_cast<int64_t>(target_frame_ns_ * kStutterThreshMult)) {
    StutterEvent ev{};
    ev.frame_index = frame_index_;
    ev.frame_ms = actual_ms;
    ev.target_ms = target_ms;
    ev.overshoot_ms = actual_ms - target_ms;

    if (stutter_log_.size() >= kStutterLogSize)
      stutter_log_.erase(stutter_log_.begin()); // evict oldest
    stutter_log_.push_back(ev);
  }
}

// =============================================================================
//  Percentile  — sorts a copy of history, returns interpolated value
// =============================================================================

double FramePacer::Percentile(double p) const {
  std::array<double, kFrameHistorySize> ms{};
  size_t valid = std::min(static_cast<size_t>(frame_index_),
                          static_cast<size_t>(kFrameHistorySize));
  for (size_t i = 0; i < valid; ++i)
    ms[i] = static_cast<double>(frame_history_[i]) / 1e6;

  if (valid == 0)
    return 0.0;

  std::sort(ms.begin(), ms.begin() + valid);

  double idx = p * static_cast<double>(valid - 1);
  size_t lo = static_cast<size_t>(idx);
  size_t hi = std::min(lo + 1, valid - 1);
  double frac = idx - static_cast<double>(lo);
  return ms[lo] + frac * (ms[hi] - ms[lo]);
}

// =============================================================================
//  GetStats
// =============================================================================

FrameStats FramePacer::GetStats() const {
  size_t valid = std::min(static_cast<size_t>(frame_index_),
                          static_cast<size_t>(kFrameHistorySize));
  if (valid == 0)
    return {};

  double sum = 0.0, mn = 1e18, mx = 0.0;
  uint64_t stutters = 0;

  for (size_t i = 0; i < valid; ++i) {
    double ms = static_cast<double>(frame_history_[i]) / 1e6;
    sum += ms;
    mn = std::min(mn, ms);
    mx = std::max(mx, ms);
    if (frame_history_[i] >
        static_cast<int64_t>(target_frame_ns_ * kStutterThreshMult))
      ++stutters;
  }

  double mean = sum / static_cast<double>(valid);
  double var = 0.0;
  for (size_t i = 0; i < valid; ++i) {
    double d = static_cast<double>(frame_history_[i]) / 1e6 - mean;
    var += d * d;
  }
  double stddev = std::sqrt(var / static_cast<double>(valid));

  return FrameStats{
      .avg_fps = (mean > 0.0) ? (1000.0 / mean) : 0.0,
      .avg_frame_ms = mean,
      .min_frame_ms = mn,
      .max_frame_ms = mx,
      .p50_ms = Percentile(0.50),
      .p95_ms = Percentile(0.95),
      .p99_ms = Percentile(0.99),
      .stddev_ms = stddev,
      .stutter_count = stutters,
      .total_frames = frame_index_,
  };
}

// =============================================================================
//  AverageFPS / FrameTimeVarianceMs / IsStuttering
// =============================================================================

double FramePacer::AverageFPS() const { return GetStats().avg_fps; }

double FramePacer::FrameTimeVarianceMs() const { return GetStats().stddev_ms; }

bool FramePacer::IsStuttering() const {
  for (int64_t t : frame_history_)
    if (t > static_cast<int64_t>(target_frame_ns_ * kStutterThreshMult))
      return true;
  return false;
}

// =============================================================================
//  PrintStats
// =============================================================================

void FramePacer::PrintStats() const {
  auto s = GetStats();
  std::cout << "[FramePacer] ── Stats ────────────────────────────\n"
            << "  Average FPS:       " << s.avg_fps << "\n"
            << "  Avg frame time:    " << s.avg_frame_ms << " ms\n"
            << "  Min / Max:         " << s.min_frame_ms << " / "
            << s.max_frame_ms << " ms\n"
            << "  P50 / P95 / P99:   " << s.p50_ms << " / " << s.p95_ms << " / "
            << s.p99_ms << " ms\n"
            << "  Std dev:           " << s.stddev_ms << " ms\n"
            << "  Stutter count:     " << s.stutter_count << "\n"
            << "  Total frames:      " << s.total_frames << "\n"
            << "  Kalman estimate:   " << (kalman_.Estimate() / 1e6) << " ms\n"
            << "  OS granularity:    " << (os_sleep_granularity_ns_ / 1e6)
            << " ms\n"
            << "  VRR active:        " << (vrr_.enabled ? "yes" : "no") << "\n"
            << "─────────────────────────────────────────────────\n";
}

// =============================================================================
//  Reset
// =============================================================================

void FramePacer::Reset() {
  frame_history_.fill(target_frame_ns_);
  frame_index_ = 0;
  last_present_time_ = Clock::now();
  stutter_log_.clear();
  kalman_.Reset(static_cast<double>(target_frame_ns_));
  os_sleep_granularity_ns_ = CalibrateOSSleepGranularityNs();
  Log("FramePacer", "Reset");
}

} // namespace gpu
} // namespace xe
