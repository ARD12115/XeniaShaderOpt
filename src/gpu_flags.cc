// =============================================================================
// gpu_flags.cc
// XSO Config Key Bridge — Implementation
//
// Drop into Xenia source at: src/xenia/gpu/gpu_flags.cc
//
// Call at emulator startup (e.g. GPUSystem::Initialize()):
//   xe::gpu::XsoConfig::Get().Init("545407F8", xenia_dir, pci_vendor_id);
// =============================================================================

#include "gpu_flags.h" // resolves via include/ — set include path in IDE

#include <cassert>
#include <cstdarg> // va_list, va_start, va_end
#include <cstdio>  // vsnprintf, fprintf, stderr
#include <filesystem>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

namespace xe {
namespace gpu {

// ---------------------------------------------------------------------------
// Internal logger
// Swap XSO_INFO/WARN/ERROR for Xenia's XELOGI/XELOGW/XELOGE when integrating.
// ---------------------------------------------------------------------------
namespace {

void XsoLog(const char *level, const char *fmt, ...) noexcept {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  std::fprintf(stderr, "[XSO][%s] %s\n", level, buf);
}

} // namespace

#define XSO_INFO(fmt, ...) XsoLog("INFO ", fmt, ##__VA_ARGS__)
#define XSO_WARN(fmt, ...) XsoLog("WARN ", fmt, ##__VA_ARGS__)
#define XSO_ERROR(fmt, ...) XsoLog("ERROR", fmt, ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// GpuVendor helpers
// ---------------------------------------------------------------------------

std::string_view GpuVendorName(GpuVendor v) noexcept {
  switch (v) {
  case GpuVendor::kNvidia:
    return "nvidia";
  case GpuVendor::kAmd:
    return "amd";
  case GpuVendor::kIntel:
    return "intel";
  case GpuVendor::kUnknown:
    return "unknown";
  }
  return "unknown";
}

GpuVendor DetectVendor(uint32_t pci_vendor_id) noexcept {
  switch (pci_vendor_id) {
  case 0x10DE:
    return GpuVendor::kNvidia;
  case 0x1002:
    return GpuVendor::kAmd;
  case 0x8086:
    return GpuVendor::kIntel;
  default:
    return GpuVendor::kUnknown;
  }
}

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

XsoConfig &XsoConfig::Get() noexcept {
  static XsoConfig instance;
  return instance;
}

// ---------------------------------------------------------------------------
// BuildCachePath
// PRD §4.1: "shaders/<TitleID>/"
// PRD §6:   separate dirs per GPU vendor
// ---------------------------------------------------------------------------

std::string XsoConfig::BuildCachePath(std::string_view xenia_dir,
                                      std::string_view title_id,
                                      GpuVendor vendor,
                                      bool per_vendor) noexcept {
  fs::path base =
      fs::path(std::string(xenia_dir)) / "shaders" / std::string(title_id);
  if (per_vendor) {
    base /= std::string(GpuVendorName(vendor));
  }
  return (base / "").string(); // trailing separator
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

void XsoConfig::Init(std::string_view title_id, std::string_view xenia_dir,
                     uint32_t vendor_id) {
  // Build hash injected by CMake -DXSO_BUILD_HASH="..."
#if defined(XSO_BUILD_HASH)
  build_hash = XSO_BUILD_HASH;
#else
  build_hash = "dev-build";
#endif

  // CMake option overrides
#if defined(XSO_PRECOMPILE_SHADERS)
  precompile_shaders = true;
#endif
#if defined(XSO_VERBOSE_CACHE)
  verbose_cache = true;
#endif
#if defined(XSO_VSYNC_GUARD)
  vsync_guard = true;
#endif
#if defined(XSO_PER_VENDOR_CACHE)
  per_vendor_cache = true;
#endif

  // Vendor detection
  detected_vendor = DetectVendor(vendor_id);
  XSO_INFO("GPU vendor: 0x%04X → %s", vendor_id,
           std::string(GpuVendorName(detected_vendor)).c_str());

  // Resolve cache path
  if (shader_cache_path.empty()) {
    resolved_cache_path =
        BuildCachePath(xenia_dir, title_id, detected_vendor, per_vendor_cache);
  } else {
    fs::path p = fs::path(shader_cache_path);
    if (per_vendor_cache) {
      p /= std::string(GpuVendorName(detected_vendor));
    }
    resolved_cache_path = (p / "").string();
  }

  // Create cache directory
  std::error_code ec;
  fs::create_directories(resolved_cache_path, ec);
  if (ec) {
    XSO_WARN("Cannot create cache dir '%s': %s", resolved_cache_path.c_str(),
             ec.message().c_str());
  } else {
    XSO_INFO("Cache path: %s", resolved_cache_path.c_str());
  }

  if (shader_cache_invalidation_required) {
    XSO_WARN(
        "shader_cache_invalidation_required=true — full rebuild on next load");
  }

  XSO_INFO("Init done. hash=%s precompile=%s vendor_split=%s vsync_guard=%s",
           build_hash.c_str(), precompile_shaders ? "on" : "off",
           per_vendor_cache ? "on" : "off", vsync_guard ? "on" : "off");
}

// ---------------------------------------------------------------------------
// NeedsInvalidation — PRD §4.1 cache version check
// ---------------------------------------------------------------------------

bool XsoConfig::NeedsInvalidation(std::string_view cached_hash) const noexcept {
  if (shader_cache_invalidation_required) {
    XSO_WARN("Manual invalidation flag set — forcing rebuild");
    return true;
  }
  if (cached_hash != build_hash) {
    XSO_WARN("Hash mismatch: cached='%s' current='%s' — rebuilding",
             std::string(cached_hash).c_str(), build_hash.c_str());
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// CheckVsyncGuard — PRD §6 HIGH severity
// ---------------------------------------------------------------------------

bool XsoConfig::CheckVsyncGuard(bool vsync_enabled) const noexcept {
  if (!vsync_guard)
    return true;
  if (!vsync_enabled)
    return true;

  XSO_ERROR("vsync guard: vsync=true conflicts with 60FPS patch.");
  XSO_ERROR("Fix: set vsync=false in xenia-canary.config.toml");
  return false;
}

} // namespace gpu
} // namespace xe
