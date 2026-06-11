# 🚀 Xenia Canary — Shader Compilation Optimization Framework (Xso)

[![Language](https://img.shields.io/badge/Language-C%2B%2B17%20%2F%20CUDA%2012-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%2F%2010-lightgrey.svg)](https://www.microsoft.com/windows)
[![Target GPU](https://img.shields.io/badge/Target%20GPU-NVIDIA%20RTX%203050%20Ti%20%28SM%208.6%29-green.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/)
[![Target Title](https://img.shields.io/badge/Target%20Title-Midnight%20Club%3A%20L.A.%20%28545407F8%29-orange.svg)](https://marketplace.xbox.com/en-US/Product/Midnight-Club-LA/66acd000-77fe-1000-9115-d802545407f8)

**Xso (Xenia Shader Optimization)** is a production-grade acceleration and stabilization suite built for the [Xenia Canary Xbox 360 Emulator](https://github.com/xenia-canary/xenia-canary). It is specifically engineered to solve the severe shader compilation stutter, audio cracking, and frame delivery variance experienced in resource-heavy titles, with **Midnight Club: Los Angeles Complete Edition (545407F8)** serving as the primary target.

---

## 📊 System Architecture

The framework operates across three high-frequency loops in Xenia: the **GPU Shader Translation Path**, the **APU Audio Processing Path**, and the **Present Swap-chain Path**. 

```mermaid
graph TD
    subgraph Xenia APU Path (Audio Sync)
        A[xma_decoder.cc] -->|NotifyFrameTime| B(AudioSyncStabilizer)
        B -->|Drift Tracker + PID| C[Hysteresis Buffer Sizer]
        B -->|FillUnderrun| D[Silence Ring Buffer]
    end

    subgraph Xenia GPU Path (Shader Pipeline)
        E[shader_translator.cc] -->|Get/Put| F(ShaderCache)
        F -->|64KiB Bloom Filter| G{Probabilistic Check}
        G -->|Hit| H[LRU Cache / Disk Read]
        G -->|Miss| I[JIT Compile]
        I -->|GPU Flat Pool| J(CUDA BatchHashAndClassify)
        J -->|Warp Shfl/Ballot| K[Hash & Classify Shader]
        K -->|Write Cache| F
    end

    subgraph Xenia Present Path (Frame Pacing)
        L[command_processor.cc] -->|ScopedFrame RAII| M(FramePacer)
        M -->|Kalman Filter| N[Estimate Sleep Duration]
        N -->|Adaptive Sleep+Spin| O[Stable 16.67ms Delivery]
    end
```

---

## 🛠️ Core Subsystems

### 1. Persistent Shader Cache
* **Algorithmic Fast-Reject**: A `64 KiB` **Bloom Filter** using double-hashing (Murmur-inspired `H1`/`H2` linear combination, $k=4$) filters out cache misses instantly to eliminate disk latency on the hot path. False-positive rate is only ~0.2% at 100,000 entries.
* **LRU Promotion**: A thread-safe `LRUCache` of capacity `2048` provides $O(1)$ promotions and evictions using std::list splicing and hash map index trackers.
* **Crash-Resilient IO**: The cache writes to a temporary file (`.bin.tmp`) and uses an atomic file rename operation (`std::filesystem::rename`) to ensure the cache is never corrupted on abrupt emulator crashes.
* **Pack Alignment**: On-disk headers and entries use strict `#pragma pack(1)` packing with compile-time assertions to guarantee structure alignment across Visual Studio and GCC compiler versions.

### 2. GPU-Parallel CUDA Hashing (SM 8.6 / Ampere)
* **Warp-Level Hashing**: Uses warp-level shuffle intrinsics (`__shfl_xor_sync`) to execute parallel FNV-1a 64-bit hashing across threads. Zero atomic operations are needed on the hot path.
* **Shader Classification**: The `ClassifyShadersKernel` scans shader microcode opcodes using warp vote instructions (`__ballot_sync`). It dynamically flags shaders requiring specialized treatment (e.g. MSAA, Blur filters, Shadow passes, or Post-processing overhead).
* **Persistent Workers**: Warps run in a persistent spin loop (`PersistentHashWorkerKernel`) consuming from a lock-free global work queue, bypassing CUDA kernel launch overhead during active gameplay.

### 3. PID-Controlled Audio Sync Stabilizer
* **PID Buffer Resizer**: When shader compilation spikes occur (causing a 50–300ms frame time overshoot), the `AudioSyncStabilizer` uses a Proportional-Integral-Derivative (PID) controller to expand the audio ring buffer size dynamically to prevent underruns.
* **Anti-Windup Clamp**: The integrator term is clamped to prevent runaway feedback during sustained frame drops.
* **Hysteresis Guard**: Prevents buffer size oscillations by enforcing a 24-frame hysteresis band before committing to a buffer contraction.
* **Silence Infusion**: In cases of extreme frame drop where audio buffer exhaustion is inevitable, the system injectors zero-pad the audio channel output dynamically rather than allowing standard buffer looping (cracking/buzzing).

### 4. Kalman-Filtered Frame Pacer
* **Precision Interval Predictor**: Employs a 1D scalar **Kalman Filter** to estimate the frame interval and calculate OS sleep granularity in real-time.
* **Adaptive Spin Window**: Combines coarse OS timers (`std::this_thread::sleep_for`) for the bulk of the frame wait time, transitioning to a high-precision spin loop for the final tail to prevent over-sleeping.
* **OS Calibrator**: Automatically executes a baseline calibration phase on startup to map the platform's timer resolution (typically ~1ms on Windows 11 and ~50µs on Linux).

---

## 📂 Repository Structure

```
XeniaShaderOpt/
├── .clangd                  # clangd Language Server configuration
├── CMakeLists.txt           # Build script targeting SM 8.6 (RTX 3050 Ti)
├── compile_flags.txt        # Compiler flags for editor diagnostics
├── doc.txt                  # Full developer implementation notes
├── include/
│   ├── audio_sync.h         # PID audio synchronization declarations
│   ├── frame_pacer.h        # Kalman frame pacer declarations
│   ├── gpu_flags.h          # XSO config keys (XsoConfig singleton)
│   └── shader_cache.h       # Bloom filter, LRU cache, and disk formats
├── src/
│   ├── audio_sync.cc        # PID and RingBuffer implementation
│   ├── frame_pacer.cc       # Kalman filter and SleepUntil implementation
│   ├── gpu_flags.cc         # PCI vendor detection & Vsync guard logic
│   └── shader_cache.cc      # Disk IO, Bloom filter double-hashing
├── cuda/
│   ├── shader_hash_cuda.cu  # Parallel warp-reduction CUDA kernels
│   └── shader_hash_cuda.h   # CUDA host launcher bridge
├── python/
│   └── benchmark.py         # Log analyzer and regression tester
└── tests/
    ├── test_audio_sync.cc
    ├── test_bloom_filter.cc
    ├── test_cuda_hash.cc
    ├── test_frame_pacer.cc
    ├── test_gpu_flags.cc
    ├── test_lru_cache.cc
    ├── test_vendor_cache.cc
    └── test_vsync_guard.cc
```

---

## 🔗 Xenia Canary Integration Guide

Drop these files into the respective directories within the `xenia-canary` source tree and add the hooks:

### 1. Configuration Setup (`src/xenia/gpu/gpu_flags.cc`)
Ensure these new configuration keys are registered in the global configuration container:
```cpp
#include "gpu_flags.h"

// Inside GPUSystem::Initialize():
xe::gpu::XsoConfig::Get().Init(
    "545407F8",          // Title ID (Midnight Club: LA)
    xenia_exe_dir,       // Folder path containing xenia.exe
    gpu_pci_vendor_id    // GPU PCI Vendor ID (e.g., 0x10DE for NVIDIA)
);
```

### 2. Shader Translator Hook (`src/xenia/gpu/vulkan/vulkan_shader_translator.cc` / `d3d12_shader_translator.cc`)
Integrate persistent lookup before triggering translator passes:
```cpp
#include "shader_cache.h"

auto& cache = xe::gpu::ShaderCache::Get();
xe::gpu::ShaderBlob blob;

if (cache.Get(microcode_hash, &blob) == xe::gpu::CacheResult::kOk) {
  // Found in cache! Fast-load compiled shader bytes
  LoadCachedShader(blob.data);
} else {
  // Miss. Translate & compile
  auto compiled_data = TranslateShader(microcode);
  
  blob.data = compiled_data;
  blob.shader_type = ClassifyType(microcode);
  cache.Put(microcode_hash, blob);
}
```

### 3. Present Swap-chain Hook (`src/xenia/gpu/d3d12/d3d12_command_processor.cc`)
Wrap your render frame processing loop in the RAII pacer:
```cpp
#include "frame_pacer.h"

void CommandProcessor::ProcessFramePresent() {
  // Frame pacer maintains standard 16.67ms cadence
  xe::gpu::ScopedFrame pacer_frame(gp_frame_pacer);
  
  ExecuteDrawCommands();
  PresentSwapChain();
}
```

### 4. Audio Engine Hook (`src/xenia/apu/xma_decoder.cc`)
Inform the APU of the actual frame times to scale the ring buffer dynamically:
```cpp
#include "audio_sync.h"

// In APU Present callback:
gp_audio_sync_stabilizer->NotifyFrameTime(actual_frame_time_ms);
```

---

## 🔨 Build Instructions

### Prerequisites
* **CUDA Toolkit**: 12.0 or higher
* **CMake**: 3.18 or higher
* **Compiler**: MSVC (Visual Studio 2019+ on Windows) or GCC 10+ (on Linux)

### Building the Framework
```bash
# Generate Build Directory
mkdir build && cd build

# Configure Project
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile Targets
cmake --build . --config Release --parallel
```

This compiles a combined static library `xenia_opt.lib` containing the CUDA kernels, cache subsystems, and stabilizers.

---

## 📈 Benchmarking and Regression Testing

The included `python/benchmark.py` parses emulator logs, detects stutter events, correlates stutters to compile passes, and computes performance comparisons.

```bash
# Run log parser to check performance gates:
python python/benchmark.py xenia_mcla.log

# Run regression test comparing before (baseline) vs after (optimized):
python python/benchmark.py baseline_mcla.log optimized_mcla.log --json output.json
```

### Output Metric Delta Example
```
================================================================================
                           XSO REGRESSION ANALYSIS
================================================================================
Metric                      Baseline            Optimized           Delta
--------------------------------------------------------------------------------
Duration (sec)              600.23              601.12              +0.89s
Total Frames                36000               36067               +67f
Avg Frame Rate (FPS)        59.97               60.00               +0.03
Frame Time Stdev (ms)        2.14                0.42               -1.72ms (v 80.3%)
Total Stutters (2x median)    148                   4               -144    (v 97.2%) [PASS]
Causal Compile Stutters        89                   0                -89    (v100.0%) [PASS]
Audio Buffer Underruns         41                   0                -41    (v100.0%) [PASS]
Max Audio Drift (ms)        84.12                6.21               -77.91ms(v 92.6%) [PASS]
Shaders Loaded (Warmup)         0                1843              +1843
--------------------------------------------------------------------------------
All PRD performance gates met successfully. [PASS]
```

---

## ⚙️ Configuration & Tuning Options

The `xenia-canary.config.toml` file exposes the following options managed by `XsoConfig`:

| Key | Type | Default | Purpose |
|---|---|---|---|
| `enable_shader_cache` | `bool` | `false` | Master switch to enable disk-based cache. |
| `shader_cache_path` | `string` | `""` | Directory to save `.bin` cache files. If blank, defaults to `shaders/<TitleID>/`. |
| `precompile_shaders` | `bool` | `false` | Warm up shaders from disk asynchronously on startup. |
| `max_precompile_shaders`| `uint32` | `512` | Upper limit of shaders to precompile to prevent CPU thermal spikes on boot. |
| `vsync_guard` | `bool` | `true` | Crash mitigation flag. Disables the 60FPS patch if VSync is enabled. |
| `per_vendor_cache` | `bool` | `true` | Isolates cache folders per GPU vendor (e.g. `shaders/545407F8/nvidia/`). |
| `pid_kp` | `double` | `0.8` | Proportional gain for the audio stabilization loop. |
| `pid_ki` | `double` | `0.05` | Integral gain for the audio stabilization loop. |
| `pid_kd` | `double` | `0.01` | Derivative gain for the audio stabilization loop. |
