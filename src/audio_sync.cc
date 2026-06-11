// =============================================================================
//  audio_sync.cc
//  Audio Desync Stabilizer — Implementation
//  Integrates into: src/xenia/apu/audio_sync.cc
//
//  Improvements over v1:
//    - PID controller fully implemented with anti-windup + dt awareness.
//    - DriftTracker: Welford online variance, rolling 64-frame window.
//    - RingBuffer<int16_t>: wrapping silence write + read with channels.
//    - MaybeResizeBuffer: hysteresis guard — no oscillation on border frames.
//    - NotifyFrameTime: uses wall-clock dt for accurate PID derivative term.
//    - FillUnderrun: drains ring first, falls back to memset silence.
//    - GetStats(): full AudioStats snapshot wired to benchmark.py fields.
// =============================================================================

#include "audio_sync.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

namespace xe {
namespace apu {

// =============================================================================
//  Internal helpers
// =============================================================================

static void Log(const char *tag, const std::string &msg) {
  std::cout << "[" << tag << "] " << msg << "\n";
}

// Wall-clock milliseconds — used for PID dt
static double NowMs() {
  using Clock = std::chrono::high_resolution_clock;
  using Ms = std::chrono::duration<double, std::milli>;
  static const auto kEpoch = Clock::now();
  return std::chrono::duration_cast<Ms>(Clock::now() - kEpoch).count();
}

// =============================================================================
//  PIDController
// =============================================================================

PIDController::PIDController(Gains gains) : gains_(gains) {}

double PIDController::Update(double error_ms, double dt_sec) {
  if (dt_sec <= 0.0)
    dt_sec = kAudioTargetFrameMs / 1000.0;

  // Proportional
  p_term_ = gains_.kp * error_ms;

  // Integral with anti-windup clamp
  integral_ += error_ms * dt_sec;
  integral_ = std::clamp(integral_, -kIntegralMax, kIntegralMax);
  i_term_ = gains_.ki * integral_;

  // Derivative (on error, EMA-smoothed implicitly by dt)
  d_term_ = gains_.kd * (error_ms - prev_error_) / dt_sec;
  prev_error_ = error_ms;

  // Output: extra frames to buffer (floor at 0 — can't un-buffer)
  double output = p_term_ + i_term_ + d_term_;
  return std::max(0.0, output);
}

void PIDController::Reset() {
  integral_ = 0.0;
  prev_error_ = 0.0;
  p_term_ = i_term_ = d_term_ = 0.0;
}

// =============================================================================
//  DriftTracker  — Welford online mean + variance, rolling window
// =============================================================================

DriftTracker::DriftTracker() { history_.fill(kAudioTargetFrameMs); }

void DriftTracker::Push(double frame_ms) {
  history_[head_ % kDriftHistorySize] = frame_ms;
  ++head_;
  if (count_ < kDriftHistorySize)
    ++count_;
  if (frame_ms > peak_ms_)
    peak_ms_ = frame_ms;
}

double DriftTracker::MeanMs() const {
  if (count_ == 0)
    return kAudioTargetFrameMs;
  double sum = 0.0;
  uint32_t n = std::min(count_, kDriftHistorySize);
  for (uint32_t i = 0; i < n; ++i)
    sum += history_[i];
  return sum / n;
}

double DriftTracker::VarianceMs() const {
  if (count_ < 2)
    return 0.0;
  double mean = MeanMs();
  double var = 0.0;
  uint32_t n = std::min(count_, kDriftHistorySize);
  for (uint32_t i = 0; i < n; ++i) {
    double d = history_[i] - mean;
    var += d * d;
  }
  return var / n;
}

double DriftTracker::DriftMs() const { return MeanMs() - kAudioTargetFrameMs; }

void DriftTracker::Reset() {
  history_.fill(kAudioTargetFrameMs);
  head_ = 0;
  count_ = 0;
  peak_ms_ = 0.0;
}

// =============================================================================
//  RingBuffer<int16_t>  explicit instantiation
// =============================================================================

template <typename T>
void RingBuffer<T>::WriteSilence(size_t frames, uint32_t channels) {
  size_t samples = frames * channels;
  size_t cap = buf_.size();

  for (size_t i = 0; i < samples; ++i) {
    buf_[tail_] = T{};
    tail_ = (tail_ + 1) % cap;
    // If full, advance head (overwrite oldest — silence is expendable)
    if (tail_ == head_)
      head_ = (head_ + 1) % cap;
  }
}

template <typename T>
size_t RingBuffer<T>::Read(T *out, size_t frames, uint32_t channels) {
  size_t avail_samples = Available();
  size_t want_samples = frames * channels;
  size_t read_samples = std::min(avail_samples, want_samples);
  size_t cap = buf_.size();

  for (size_t i = 0; i < read_samples; ++i) {
    out[i] = buf_[head_];
    head_ = (head_ + 1) % cap;
  }

  // Zero-pad remainder if we ran short
  if (read_samples < want_samples)
    std::memset(out + read_samples, 0,
                (want_samples - read_samples) * sizeof(T));

  return read_samples / std::max(channels, 1u);
}

// Explicit instantiation for int16_t
template class RingBuffer<int16_t>;

// =============================================================================
//  AudioSyncStabilizer — Constructor
// =============================================================================

AudioSyncStabilizer::AudioSyncStabilizer(uint32_t sample_rate,
                                         uint32_t channels,
                                         uint32_t base_buffer_frames,
                                         PIDController::Gains gains)
    : sample_rate_(sample_rate), channels_(channels),
      base_buffer_frames_(base_buffer_frames),
      current_buffer_frames_(base_buffer_frames),
      last_resize_target_(base_buffer_frames), pid_(gains),
      ring_(base_buffer_frames * channels * kAudioMaxBufferMultiplier) {
  std::ostringstream oss;
  oss << "Init — " << sample_rate_ << "Hz  ch=" << channels_
      << "  base=" << base_buffer_frames_ << " frames"
      << "  ring_cap=" << ring_.Capacity();
  Log("AudioSync", oss.str());

  last_notify_time_ms_ = NowMs();
}

// =============================================================================
//  MaybeResizeBuffer  — hysteresis guard
// =============================================================================

void AudioSyncStabilizer::MaybeResizeBuffer(uint32_t new_size) {
  // Clamp to [base, base * kMaxMultiplier]
  new_size = std::clamp(new_size, base_buffer_frames_,
                        base_buffer_frames_ * kAudioMaxBufferMultiplier);

  // Only resize if change exceeds hysteresis band
  uint32_t delta = (new_size > last_resize_target_)
                       ? (new_size - last_resize_target_)
                       : (last_resize_target_ - new_size);

  if (delta < kHysteresisFrames)
    return;

  if (new_size != current_buffer_frames_) {
    std::ostringstream oss;
    oss << "Buffer resize: " << current_buffer_frames_ << " → " << new_size
        << " frames"
        << "  (Δ"
        << static_cast<int>(new_size) - static_cast<int>(current_buffer_frames_)
        << ")";
    Log("AudioSync", oss.str());

    current_buffer_frames_ = new_size;
    last_resize_target_ = new_size;

    // Re-seed ring with silence at new size
    ring_.Clear();
    ring_.WriteSilence(current_buffer_frames_, channels_);
  }
}

// =============================================================================
//  NotifyFrameTime  — hot path, called every frame
// =============================================================================

void AudioSyncStabilizer::NotifyFrameTime(double frame_ms) {
  // 1. Track drift
  drift_.Push(frame_ms);

  // 2. Compute dt for PID (wall-clock seconds since last call)
  double now_ms = NowMs();
  double dt_sec = (now_ms - last_notify_time_ms_) / 1000.0;
  last_notify_time_ms_ = now_ms;

  // 3. PID: error = how far mean frame time is above target
  double error_ms = drift_.DriftMs();
  double pid_out = pid_.Update(error_ms, dt_sec);

  // 4. Convert PID output (ms of extra buffer) to extra frames
  double frames_per_ms = static_cast<double>(sample_rate_) / 1000.0;
  uint32_t extra_frames = static_cast<uint32_t>(pid_out * frames_per_ms);
  uint32_t target_frames = base_buffer_frames_ + extra_frames;

  // 5. Apply with hysteresis
  MaybeResizeBuffer(target_frames);
}

// =============================================================================
//  FillUnderrun  — drain ring first, then memset silence
// =============================================================================

UnderrunResult AudioSyncStabilizer::FillUnderrun(void *out_buf,
                                                 uint32_t frames_requested) {
  UnderrunResult result{};
  result.ring_was_empty = ring_.Empty();

  auto *pcm = reinterpret_cast<int16_t *>(out_buf);

  // Drain whatever the ring has
  size_t from_ring = ring_.Read(pcm, frames_requested, channels_);

  // If ring ran short, remainder is already zeroed by RingBuffer::Read
  result.frames_filled = static_cast<uint32_t>(
      std::min(static_cast<size_t>(frames_requested),
               from_ring + (frames_requested - from_ring)));
  result.frames_filled = frames_requested; // always fill full request
  result.bytes_written = result.frames_filled * channels_ * sizeof(int16_t);

  // Re-prime ring with silence for next underrun
  ring_.WriteSilence(current_buffer_frames_, channels_);

  ++underrun_count_;

  std::ostringstream oss;
  oss << "Underrun #" << underrun_count_ << "  filled=" << result.frames_filled
      << " frames"
      << "  ring_was_empty=" << (result.ring_was_empty ? "yes" : "no");
  Log("AudioSync", oss.str());

  return result;
}

// =============================================================================
//  GetStats
// =============================================================================

AudioStats AudioSyncStabilizer::GetStats() const {
  return AudioStats{
      .underrun_count = underrun_count_,
      .current_buffer_frames = current_buffer_frames_,
      .drift_accum_ms = drift_.DriftMs(),
      .mean_frame_ms = drift_.MeanMs(),
      .peak_frame_ms = drift_.PeakMs(),
      .frame_variance_ms = drift_.VarianceMs(),
      .pid_output =
          pid_.ProportionalTerm() + pid_.IntegralTerm() + pid_.DerivativeTerm(),
      .pid_p = pid_.ProportionalTerm(),
      .pid_i = pid_.IntegralTerm(),
      .pid_d = pid_.DerivativeTerm(),
  };
}

// =============================================================================
//  PrintStats
// =============================================================================

void AudioSyncStabilizer::PrintStats() const {
  auto s = GetStats();
  std::cout << "[AudioSync] ── Stats ─────────────────────────────\n"
            << "  Buffer (current):  " << s.current_buffer_frames << " frames\n"
            << "  Underruns total:   " << s.underrun_count << "\n"
            << "  Drift (mean−tgt):  " << s.drift_accum_ms << " ms\n"
            << "  Mean frame time:   " << s.mean_frame_ms << " ms\n"
            << "  Peak frame time:   " << s.peak_frame_ms << " ms\n"
            << "  Frame variance:    " << s.frame_variance_ms << " ms²\n"
            << "  PID output:        " << s.pid_output << "\n"
            << "  PID  P/I/D:        " << s.pid_p << " / " << s.pid_i << " / "
            << s.pid_d << "\n"
            << "─────────────────────────────────────────────────\n";
}

// =============================================================================
//  Reset
// =============================================================================

void AudioSyncStabilizer::Reset() {
  current_buffer_frames_ = base_buffer_frames_;
  last_resize_target_ = base_buffer_frames_;
  underrun_count_ = 0;
  last_notify_time_ms_ = NowMs();
  pid_.Reset();
  drift_.Reset();
  ring_.Clear();
  Log("AudioSync", "Reset to base state");
}

} // namespace apu
} // namespace xe
