#pragma once
// =============================================================================
// gpu_flags.h
// XSO Config Key Bridge
//
// Drop into Xenia source at: src/xenia/gpu/gpu_flags.h
// =============================================================================

#ifndef XENIA_GPU_FLAGS_H_
#define XENIA_GPU_FLAGS_H_

#include <cstdint>
#include <string>
#include <string_view> // C++17 — required

namespace xe {
namespace gpu {

// ---------------------------------------------------------------------------
// GPU vendor — per-vendor cache directory split (PRD §6)
// ---------------------------------------------------------------------------
enum class GpuVendor : uint8_t {
  kUnknown = 0,
  kNvidia = 1, // 0x10DE — RTX 3050 Ti (primary target)
  kAmd = 2,    // 0x1002 — RX 5000+ (supported)
  kIntel = 3,  // 0x8086 — fallback
};

std::string_view GpuVendorName(GpuVendor v) noexcept;
GpuVendor DetectVendor(uint32_t pci_vendor_id) noexcept;

// ---------------------------------------------------------------------------
// XsoConfig — singleton holding all PRD §4.3 config keys
//
// xenia-canary.config.toml keys owned here:
//   enable_shader_cache                bool   default false
//   shader_cache_path                  string default "shaders/<TitleID>/"
//   precompile_shaders                 bool   default false
//   max_precompile_shaders             uint32 default 512
//   shader_cache_invalidation_required bool   default false
// ---------------------------------------------------------------------------
struct XsoConfig {
  // PRD §4.3 — base config keys
  bool enable_shader_cache = false;
  std::string shader_cache_path = "";
  bool precompile_shaders = false;
  uint32_t max_precompile_shaders = 512;
  bool shader_cache_invalidation_required = false;

  // PRD §6 — risk mitigations
  bool vsync_guard = true;      // HIGH severity
  bool per_vendor_cache = true; // MEDIUM severity

  // Debug
  bool verbose_cache = false;

  // Audio PID gains — fed to AudioSyncStabilizer::SetPIDGains()
  double pid_kp = 0.8;
  double pid_ki = 0.05;
  double pid_kd = 0.01;

  // Resolved at Init() time
  GpuVendor detected_vendor = GpuVendor::kUnknown;
  std::string resolved_cache_path = "";
  std::string build_hash = "";

  // --- Singleton ---
  static XsoConfig &Get() noexcept;

  // Call once at startup before any shader translation.
  //   title_id   e.g. "545407F8"
  //   xenia_dir  absolute path to xenia.exe directory
  //   vendor_id  PCI vendor ID from GPU (0x10DE, 0x1002, 0x8086)
  void Init(std::string_view title_id, std::string_view xenia_dir,
            uint32_t vendor_id);

  // Returns true when cached build hash differs from current XSO_BUILD_HASH.
  // Triggers automatic cache rebuild. (PRD §4.1, §6)
  bool NeedsInvalidation(std::string_view cached_hash) const noexcept;

  // Returns false and logs error when vsync=true conflicts with 60FPS patch.
  // Caller should abort or disable the patch. (PRD §6 HIGH severity)
  bool CheckVsyncGuard(bool vsync_enabled) const noexcept;

  // Builds the resolved cache path for a title + vendor combination.
  //   per_vendor=true  → <xenia_dir>/shaders/<TitleID>/<vendor>/
  //   per_vendor=false → <xenia_dir>/shaders/<TitleID>/
  static std::string BuildCachePath(std::string_view xenia_dir,
                                    std::string_view title_id, GpuVendor vendor,
                                    bool per_vendor) noexcept;

private:
  XsoConfig() = default;
};

} // namespace gpu
} // namespace xe

#endif // XENIA_GPU_FLAGS_H_
