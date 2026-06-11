#!/usr/bin/env python3
# =============================================================================
#  benchmark.py
#  Xenia Canary — Shader Compilation Optimization Benchmark Tool
#  Target: Midnight Club: LA Complete Edition (545407F8)
#
#  Usage:
#    python benchmark.py baseline.log
#    python benchmark.py baseline.log optimized.log   # compare mode
#
#  What it measures:
#    - Shader compile events: count, total time, P50/P95/P99 durations
#    - Frame times: stutter rate, P99, avg FPS
#    - Audio drift: mean, peak, underrun count
#    - Stutter ↔ shader compile correlation (overlap detection)
#    - Time-series clustering of recurring stutter patterns
#    - ASCII heatmap of shader compile hotspots over session timeline
#
#  Output:
#    - Console report (colour-coded pass/fail against PRD gates)
#    - JSON export  (<logname>_report.json)
#    - Regression exit code: 0 = all gates pass, 1 = any gate fails
#
#  PRD gates:
#    - Stutter rate   < 3 events / 10 min
#    - Audio drift    < 10 ms
#    - P99 frame time < 33 ms
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
#  ANSI colour helpers
# =============================================================================

_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def green(t: str)  -> str: return _c("32;1", t)
def red(t: str)    -> str: return _c("31;1", t)
def yellow(t: str) -> str: return _c("33;1", t)
def cyan(t: str)   -> str: return _c("36;1", t)
def bold(t: str)   -> str: return _c("1", t)

# =============================================================================
#  PRD gate thresholds
# =============================================================================

GATE_STUTTER_PER_10MIN  = 3.0    # events / 10 min
GATE_AUDIO_DRIFT_MS     = 10.0   # ms
GATE_P99_FRAME_MS       = 33.0   # ms

# =============================================================================
#  Regex patterns
#  Match lines emitted by ShaderCache, FramePacer, AudioSync Log() helpers.
# =============================================================================

# [ShaderCache] Loaded 412 shaders from cache/545407F8/vulkan_nvidia.xshc
RE_CACHE_LOAD   = re.compile(
    r"\[ShaderCache\].*Loaded (\d+) shaders")

# [ShaderPrecompiler] Done — 128/512 shaders precompiled
RE_PRECOMPILE   = re.compile(
    r"\[ShaderPrecompiler\].*Done.*?(\d+)/(\d+)")

# Shader compile event: timestamp + duration
# Expected format: [t=1234.56ms] [ShaderCompile] hash=0xDEADBEEF  duration=47.3ms
RE_SHADER_COMPILE = re.compile(
    r"\[t=([0-9.]+)ms\].*\[ShaderCompile\].*duration=([0-9.]+)ms")

# Frame time log line
# [t=1234.56ms] [FramePacer] frame=18.23ms
RE_FRAME_TIME   = re.compile(
    r"\[t=([0-9.]+)ms\].*\[FramePacer\].*frame=([0-9.]+)ms")

# Audio drift log line
# [t=1234.56ms] [AudioSync] drift=3.21ms underruns=0
RE_AUDIO_DRIFT  = re.compile(
    r"\[t=([0-9.]+)ms\].*\[AudioSync\].*drift=([0-9.]+)ms.*underruns=(\d+)")

# Audio underrun standalone
RE_UNDERRUN     = re.compile(
    r"\[AudioSync\].*Underrun #(\d+)")

# Buffer resize
RE_BUF_RESIZE   = re.compile(
    r"\[AudioSync\].*Buffer resize:.*-> (\d+) frames")

# =============================================================================
#  Data containers
# =============================================================================

@dataclass
class ShaderEvent:
    time_ms:     float
    duration_ms: float

@dataclass
class FrameEvent:
    time_ms:  float
    frame_ms: float

@dataclass
class AudioEvent:
    time_ms:   float
    drift_ms:  float
    underruns: int

@dataclass
class ParsedLog:
    path:           str
    shader_events:  List[ShaderEvent]  = field(default_factory=list)
    frame_events:   List[FrameEvent]   = field(default_factory=list)
    audio_events:   List[AudioEvent]   = field(default_factory=list)
    underrun_total: int                = 0
    cache_loaded:   int                = 0
    precompiled:    Tuple[int, int]    = (0, 0)
    duration_ms:    float              = 0.0

# =============================================================================
#  Statistics helpers
# =============================================================================

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s   = sorted(data)
    idx = p / 100.0 * (len(s) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])

def mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def stddev(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

# =============================================================================
#  Log parser
# =============================================================================

def parse_log(path: str) -> ParsedLog:
    log = ParsedLog(path=path)
    p   = Path(path)

    if not p.exists():
        print(red(f"  x File not found: {path}"))
        return log

    with p.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        if m := RE_SHADER_COMPILE.search(line):
            log.shader_events.append(
                ShaderEvent(float(m.group(1)), float(m.group(2))))

        if m := RE_FRAME_TIME.search(line):
            log.frame_events.append(
                FrameEvent(float(m.group(1)), float(m.group(2))))

        if m := RE_AUDIO_DRIFT.search(line):
            log.audio_events.append(
                AudioEvent(float(m.group(1)), float(m.group(2)), int(m.group(3))))

        if m := RE_UNDERRUN.search(line):
            log.underrun_total = int(m.group(1))

        if m := RE_CACHE_LOAD.search(line):
            log.cache_loaded = int(m.group(1))

        if m := RE_PRECOMPILE.search(line):
            log.precompiled = (int(m.group(1)), int(m.group(2)))

    # Session duration from first to last timestamped event
    all_times = (
        [e.time_ms for e in log.shader_events] +
        [e.time_ms for e in log.frame_events]  +
        [e.time_ms for e in log.audio_events]
    )
    if all_times:
        log.duration_ms = max(all_times) - min(all_times)

    return log

# =============================================================================
#  Stutter detection
#  A frame is a stutter if it exceeds 2x the median frame time.
# =============================================================================

def detect_stutters(log: ParsedLog) -> List[FrameEvent]:
    if not log.frame_events:
        return []
    times     = [e.frame_ms for e in log.frame_events]
    threshold = percentile(times, 50) * 2.0
    return [e for e in log.frame_events if e.frame_ms >= threshold]

# =============================================================================
#  Stutter <-> shader compile correlation
#  A stutter is "correlated" if a shader compile started within
#  +/-window_ms of the stutter frame timestamp.
# =============================================================================

def correlate_stutter_compile(
        stutters:      List[FrameEvent],
        shader_events: List[ShaderEvent],
        window_ms:     float = 50.0) -> List[Tuple[FrameEvent, ShaderEvent]]:

    correlated = []
    for stutter in stutters:
        for ev in shader_events:
            if abs(ev.time_ms - stutter.time_ms) <= window_ms:
                correlated.append((stutter, ev))
                break
    return correlated

# =============================================================================
#  Time-series clustering  (equal-width binning)
#  Partitions session into N time buckets, counts stutters per bucket.
# =============================================================================

def cluster_stutters(
        stutters:    List[FrameEvent],
        duration_ms: float,
        n_buckets:   int = 20) -> List[Tuple[float, int]]:

    if duration_ms <= 0 or not stutters:
        return []

    width  = duration_ms / n_buckets
    counts = [0] * n_buckets

    for ev in stutters:
        bucket = min(int(ev.time_ms / width), n_buckets - 1)
        counts[bucket] += 1

    return [(i * width, counts[i]) for i in range(n_buckets)]

# =============================================================================
#  ASCII heatmap of shader compile density over session timeline
# =============================================================================

_HEAT_CHARS = " .:;+=xX$&#"

def ascii_heatmap(log: ParsedLog, width: int = 60) -> str:
    if not log.shader_events or log.duration_ms <= 0:
        return "  (no shader events)"

    t_min     = min(e.time_ms for e in log.shader_events)
    bucket_ms = log.duration_ms / width
    counts    = [0] * width

    for ev in log.shader_events:
        rel = ev.time_ms - t_min
        idx = min(int(rel / bucket_ms), width - 1)
        counts[idx] += 1

    peak = max(counts) if max(counts) > 0 else 1
    bar  = ""
    for c in counts:
        level = int((c / peak) * (len(_HEAT_CHARS) - 1))
        bar  += _HEAT_CHARS[level]

    end_s = f"{log.duration_ms / 1000:.1f}s"
    pad   = width - 4 - len(end_s)
    return f"  |{bar}|\n  0s{' ' * pad}{end_s}"

# =============================================================================
#  Report builder
# =============================================================================

@dataclass
class Report:
    label:               str
    duration_s:          float
    # Shaders
    shader_count:        int
    shader_total_ms:     float
    shader_p50_ms:       float
    shader_p95_ms:       float
    shader_p99_ms:       float
    cache_loaded:        int
    precompiled:         Tuple[int, int]
    # Frames
    frame_count:         int
    avg_fps:             float
    avg_frame_ms:        float
    p50_frame_ms:        float
    p95_frame_ms:        float
    p99_frame_ms:        float
    stutter_count:       int
    stutter_per_10min:   float
    correlated_stutters: int
    # Audio
    audio_drift_mean_ms: float
    audio_drift_peak_ms: float
    audio_drift_stddev:  float
    underrun_count:      int
    # Gates
    gate_stutter:        bool
    gate_drift:          bool
    gate_p99:            bool

def build_report(log: ParsedLog, label: str) -> Report:
    durations   = [e.duration_ms for e in log.shader_events]
    frame_times = [e.frame_ms    for e in log.frame_events]
    drift_vals  = [e.drift_ms    for e in log.audio_events]

    stutters    = detect_stutters(log)
    correlated  = correlate_stutter_compile(stutters, log.shader_events)

    duration_s   = log.duration_ms / 1000.0
    stutter_rate = (len(stutters) / (duration_s / 60.0)) * 10.0 \
                   if duration_s > 0 else 0.0

    p99_frame  = percentile(frame_times, 99)
    drift_peak = max(drift_vals) if drift_vals else 0.0
    drift_mean = mean(drift_vals)
    avg_fps    = 1000.0 / mean(frame_times) if frame_times else 0.0

    return Report(
        label               = label,
        duration_s          = duration_s,
        shader_count        = len(log.shader_events),
        shader_total_ms     = sum(durations),
        shader_p50_ms       = percentile(durations, 50),
        shader_p95_ms       = percentile(durations, 95),
        shader_p99_ms       = percentile(durations, 99),
        cache_loaded        = log.cache_loaded,
        precompiled         = log.precompiled,
        frame_count         = len(log.frame_events),
        avg_fps             = avg_fps,
        avg_frame_ms        = mean(frame_times),
        p50_frame_ms        = percentile(frame_times, 50),
        p95_frame_ms        = percentile(frame_times, 95),
        p99_frame_ms        = p99_frame,
        stutter_count       = len(stutters),
        stutter_per_10min   = stutter_rate,
        correlated_stutters = len(correlated),
        audio_drift_mean_ms = drift_mean,
        audio_drift_peak_ms = drift_peak,
        audio_drift_stddev  = stddev(drift_vals),
        underrun_count      = log.underrun_total,
        gate_stutter        = stutter_rate < GATE_STUTTER_PER_10MIN,
        gate_drift          = drift_mean   < GATE_AUDIO_DRIFT_MS,
        gate_p99            = p99_frame    < GATE_P99_FRAME_MS,
    )

# =============================================================================
#  Console printer
# =============================================================================

def _gate(passed: bool, value: str, threshold: str) -> str:
    icon  = green("PASS") if passed else red("FAIL")
    color = green if passed else red
    return f"  [{icon}]  {color(value):<36}  gate: {threshold}"

def print_report(r: Report, log: ParsedLog) -> None:
    print()
    sep = "=" * 60
    print(bold(f"{sep}"))
    print(bold(f"  {r.label}  ({r.duration_s:.1f}s session)"))
    print(bold(f"{sep}"))

    # Shader
    print(cyan("\n  Shader Compile"))
    print(f"    Events         {r.shader_count}")
    print(f"    Total time     {r.shader_total_ms:.1f} ms")
    print(f"    P50 / P95 / P99  "
          f"{r.shader_p50_ms:.1f} / {r.shader_p95_ms:.1f} / {r.shader_p99_ms:.1f} ms")
    print(f"    Cache loaded   {r.cache_loaded} shaders")
    print(f"    Precompiled    {r.precompiled[0]} / {r.precompiled[1]}")

    # Heatmap
    print(cyan("\n  Compile Hotspot Heatmap  (denser = more compiles)"))
    print(ascii_heatmap(log))

    # Frame
    print(cyan("\n  Frame Pacing"))
    print(f"    Frames         {r.frame_count}")
    print(f"    Avg FPS        {r.avg_fps:.2f}")
    print(f"    Avg / P50      {r.avg_frame_ms:.2f} / {r.p50_frame_ms:.2f} ms")
    print(f"    P95 / P99      {r.p95_frame_ms:.2f} / {r.p99_frame_ms:.2f} ms")
    print(f"    Stutters       {r.stutter_count}  "
          f"({r.correlated_stutters} correlated with shader compile)")

    # Audio
    print(cyan("\n  Audio Sync"))
    print(f"    Drift mean     {r.audio_drift_mean_ms:.2f} ms")
    print(f"    Drift peak     {r.audio_drift_peak_ms:.2f} ms")
    print(f"    Drift stddev   {r.audio_drift_stddev:.2f} ms")
    print(f"    Underruns      {r.underrun_count}")

    # PRD gates
    print(cyan("\n  PRD Gates"))
    print(_gate(r.gate_stutter,
                f"Stutter {r.stutter_per_10min:.2f} / 10min",
                f"< {GATE_STUTTER_PER_10MIN} / 10min"))
    print(_gate(r.gate_drift,
                f"Audio drift {r.audio_drift_mean_ms:.2f} ms",
                f"< {GATE_AUDIO_DRIFT_MS} ms"))
    print(_gate(r.gate_p99,
                f"P99 frame {r.p99_frame_ms:.2f} ms",
                f"< {GATE_P99_FRAME_MS} ms"))

    all_pass = r.gate_stutter and r.gate_drift and r.gate_p99
    label    = green("ALL GATES PASS") if all_pass else red("GATES FAILING")
    print(f"\n  {bold(label)}\n")

# =============================================================================
#  Comparison printer  (baseline vs optimized)
# =============================================================================

def _delta(before: float, after: float, lower_is_better: bool = True) -> str:
    if before == 0:
        return "    --"
    pct      = (after - before) / before * 100.0
    improved = (pct < 0) if lower_is_better else (pct > 0)
    sign     = "v" if pct < 0 else "^"
    s        = f"{sign}{abs(pct):.1f}%"
    return green(s) if improved else red(s)

def print_comparison(base: Report, opt: Report) -> None:
    print()
    print(bold("=" * 60))
    print(bold("  COMPARISON: Baseline vs Optimized"))
    print(bold("=" * 60))

    rows = [
        ("Shader compile P99 (ms)",  base.shader_p99_ms,      opt.shader_p99_ms,      True),
        ("Frame avg (ms)",            base.avg_frame_ms,       opt.avg_frame_ms,       True),
        ("Frame P95 (ms)",            base.p95_frame_ms,       opt.p95_frame_ms,       True),
        ("Frame P99 (ms)",            base.p99_frame_ms,       opt.p99_frame_ms,       True),
        ("Avg FPS",                   base.avg_fps,            opt.avg_fps,            False),
        ("Stutter / 10min",           base.stutter_per_10min,  opt.stutter_per_10min,  True),
        ("Audio drift mean (ms)",     base.audio_drift_mean_ms,opt.audio_drift_mean_ms,True),
        ("Audio drift peak (ms)",     base.audio_drift_peak_ms,opt.audio_drift_peak_ms,True),
        ("Underruns",                 float(base.underrun_count),
                                      float(opt.underrun_count), True),
        ("Cache loaded",              float(base.cache_loaded), float(opt.cache_loaded), False),
    ]

    col = 28
    print(f"\n  {'Metric':<{col}}  {'Baseline':>10}  {'Optimized':>10}  {'Delta':>8}")
    print("  " + "-" * (col + 36))
    for name, b, o, lib in rows:
        print(f"  {name:<{col}}  {b:>10.2f}  {o:>10.2f}  {_delta(b, o, lib):>8}")
    print()

# =============================================================================
#  JSON export
# =============================================================================

def export_json(report: Report, log: ParsedLog, out_path: str) -> None:
    stutters = detect_stutters(log)
    clusters = cluster_stutters(stutters, log.duration_ms)

    # Convert precompiled tuple to list for JSON serialisation
    rep_dict = asdict(report)
    rep_dict["precompiled"] = list(report.precompiled)

    payload = {
        "report":   rep_dict,
        "clusters": [
            {"bucket_start_ms": b, "stutter_count": c}
            for b, c in clusters
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"  JSON -> {out_path}")

# =============================================================================
#  Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Xenia Canary Shader Optimization Benchmark Tool")
    parser.add_argument(
        "logs", nargs="+", metavar="LOG",
        help="One log file (analysis) or two (baseline + optimized compare)")
    parser.add_argument(
        "--no-colour", action="store_true",
        help="Disable ANSI colour output")
    parser.add_argument(
        "--heatmap-width", type=int, default=60, metavar="N",
        help="Width of ASCII heatmap in characters (default: 60)")
    args = parser.parse_args()

    global _USE_COLOUR
    if args.no_colour:
        _USE_COLOUR = False

    if len(args.logs) == 1:
        # Single log mode
        log    = parse_log(args.logs[0])
        report = build_report(log, label=Path(args.logs[0]).stem)
        print_report(report, log)

        json_path = str(Path(args.logs[0]).with_suffix("")) + "_report.json"
        export_json(report, log, json_path)

        return 0 if (report.gate_stutter and report.gate_drift and report.gate_p99) else 1

    elif len(args.logs) == 2:
        # Compare mode
        base_log = parse_log(args.logs[0])
        opt_log  = parse_log(args.logs[1])
        base_rep = build_report(base_log, label="Baseline")
        opt_rep  = build_report(opt_log,  label="Optimized")

        print_report(base_rep, base_log)
        print_report(opt_rep,  opt_log)
        print_comparison(base_rep, opt_rep)

        for rep, log in [(base_rep, base_log), (opt_rep, opt_log)]:
            json_path = str(Path(log.path).with_suffix("")) + "_report.json"
            export_json(rep, log, json_path)

        return 0 if (opt_rep.gate_stutter and opt_rep.gate_drift and opt_rep.gate_p99) else 1

    else:
        print(red("Error: provide 1 or 2 log files."))
        return 2


if __name__ == "__main__":
    sys.exit(main())
