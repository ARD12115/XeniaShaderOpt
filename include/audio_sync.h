#pragma once
// =============================================================================
//  audio_sync.h
//  Audio Desync Stabilizer — Header
//  Integrates into: src/xenia/apu/
//
//  What this solves:
//    - Shader compile spikes (50–300ms) drain or overflow the audio ring
//      buffer → crackling, pops, and cumulative drift.
//
//  Improvements over v1:
//    - PID controller replaces bare EMA for buffer resize decisions.
//      EMA still used internally for smoothing the derivative term.
//    - Hysteresis thresholds prevent rapid oscillation (grow/shrink chatter).
//    - RingBuffer<T> helper encapsulates the circular audio staging area.
//    - DriftTracker accumulates rolling frame-time statistics independently
//      so the PID controller can query error, derivative, integral cleanly.
//    - FillUnderrun() returns a typed result (frames filled + silence bytes).
//    - Full stats struct for benchmark.py integration.
// =============================================================================

#ifndef XENIA_APU_AUDIO_SYNC_H_
#define XENIA_APU_AUDIO_SYNC_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace xe {
namespace apu {

// =============================================================================
//  Constants
// =============================================================================

inline constexpr uint32_t kAudioDefaultSampleRate = 48000;
inline constexpr uint32_t kAudioDefaultChannels = 2;
inline constexpr uint32_t kAudioBaseBufferFrames = 384;  // ~8ms @ 48kHz
inline constexpr uint32_t kAudioMaxBufferMultiplier = 4; // cap at 4× base
inline constexpr double kAudioTargetFrameMs = 16.667;
inline constexpr uint32_t kDriftHistorySize = 64; // rolling window

// =============================================================================
//  PIDController  — adaptive buffer sizing
//
//  Controls the number of extra audio frames buffered to absorb GPU stalls.
//  Setpoint = 0ms drift. Input = measured drift_ms. Output = extra frames.
// =============================================================================

class PIDController {
public:
  struct Gains {
    double kp = 2.0;  // proportional — reacts to current error
    double ki = 0.05; // integral     — corrects persistent offset
    double kd = 0.8;  // derivative   — damps overshoot
  };

  explicit PIDController(Gains gains = {});

  // Feed one measurement; returns control output (extra buffer frames, ≥0).
  double Update(double error_ms, double dt_sec);

  void Reset();
  void SetGains(Gains g) { gains_ = g; }
  Gains GetGains() const { return gains_; }

  double ProportionalTerm() const { return p_term_; }
  double IntegralTerm() const { return i_term_; }
  double DerivativeTerm() const { return d_term_; }

private:
  Gains gains_;
  double integral_ = 0.0;
  double prev_error_ = 0.0;
  double p_term_ = 0.0;
  double i_term_ = 0.0;
  double d_term_ = 0.0;

  // Anti-windup clamp on integral accumulator
  static constexpr double kIntegralMax = 200.0;
};

// =============================================================================
//  DriftTracker  — rolling frame-time statistics
// =============================================================================

class DriftTracker {
public:
  DriftTracker();

  // Record one frame duration (milliseconds).
  void Push(double frame_ms);

  double MeanMs() const;
  double VarianceMs() const;
  double PeakMs() const { return peak_ms_; }
  double DriftMs() const; // mean − target (signed)
  void Reset();

private:
  std::array<double, kDriftHistorySize> history_{};
  uint32_t head_ = 0;
  uint32_t count_ = 0;
  double peak_ms_ = 0.0;
};

// =============================================================================
//  RingBuffer<T>  — lock-free single-producer / single-consumer staging
//
//  Used internally to stage silence frames on underrun.
//  T = int16_t (PCM sample).
// =============================================================================

template <typename T> class RingBuffer {
public:
  explicit RingBuffer(size_t capacity) : buf_(capacity, T{}) {}

  void Resize(size_t new_cap) {
    buf_.assign(new_cap, T{});
    head_ = tail_ = 0;
  }
  size_t Capacity() const { return buf_.size(); }
  size_t Available() const {
    return (tail_ >= head_) ? (tail_ - head_) : (buf_.size() - head_ + tail_);
  }
  bool Empty() const { return head_ == tail_; }

  // Write silence (zeros) into ring — used for underrun fill
  void WriteSilence(size_t frames, uint32_t channels);

  // Read into caller buffer. Returns frames actually read.
  size_t Read(T *out, size_t frames, uint32_t channels);

  void Clear() { head_ = tail_ = 0; }

private:
  std::vector<T> buf_;
  size_t head_ = 0;
  size_t tail_ = 0;
};

// =============================================================================
//  UnderrunResult
// =============================================================================

struct UnderrunResult {
  uint32_t frames_filled; // actual frames written to caller buffer
  uint32_t bytes_written; // = frames_filled * channels * sizeof(int16_t)
  bool ring_was_empty;    // true if ring had nothing — pure silence fill
};

// =============================================================================
//  AudioStats  — snapshot for benchmark / logging
// =============================================================================

struct AudioStats {
  uint32_t underrun_count;
  uint32_t current_buffer_frames;
  double drift_accum_ms;
  double mean_frame_ms;
  double peak_frame_ms;
  double frame_variance_ms;
  double pid_output;
  double pid_p;
  double pid_i;
  double pid_d;
};

// =============================================================================
//  AudioSyncStabilizer  — main public API
// =============================================================================

class AudioSyncStabilizer {
public:
  AudioSyncStabilizer(uint32_t sample_rate = kAudioDefaultSampleRate,
                      uint32_t channels = kAudioDefaultChannels,
                      uint32_t base_buffer_frames = kAudioBaseBufferFrames,
                      PIDController::Gains gains = {});

  // ── Hot path ──────────────────────────────────────────────────────────────

  // Call once per rendered frame with actual frame duration (milliseconds).
  // Internally updates DriftTracker → PID → buffer resize with hysteresis.
  void NotifyFrameTime(double frame_ms);

  // Called by audio backend on ring-buffer underrun.
  // Fills `frames_requested` frames of silence into out_buf.
  UnderrunResult FillUnderrun(void *out_buf, uint32_t frames_requested);

  // ── Query ─────────────────────────────────────────────────────────────────

  uint32_t GetCurrentBufferFrames() const { return current_buffer_frames_; }
  uint32_t GetUnderrunCount() const { return underrun_count_; }
  AudioStats GetStats() const;

  // ── Control ───────────────────────────────────────────────────────────────

  void PrintStats() const;
  void Reset();
  void SetPIDGains(PIDController::Gains g) { pid_.SetGains(g); }

private:
  // Resize buffer only when change exceeds hysteresis band
  void MaybeResizeBuffer(uint32_t new_size);

  uint32_t sample_rate_;
  uint32_t channels_;
  uint32_t base_buffer_frames_;
  uint32_t current_buffer_frames_;
  uint32_t underrun_count_ = 0;

  // Hysteresis: only resize when delta exceeds this many frames
  static constexpr uint32_t kHysteresisFrames = 24; // ~0.5ms @ 48kHz
  uint32_t last_resize_target_ = 0;

  PIDController pid_;
  DriftTracker drift_;
  RingBuffer<int16_t> ring_;

  // Timestamp for PID dt computation
  double last_notify_time_ms_ = 0.0;
};

} // namespace apu
} // namespace xe

#endif // XENIA_APU_AUDIO_SYNC_H_
