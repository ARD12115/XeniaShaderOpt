// =============================================================================
// shader_cache.cc
// Persistent Shader Cache — Implementation
// Integrates into: src/xenia/gpu/shader_cache.cc
//
// v3 changes over v2:
//   + Singleton ShaderCache::Get() — wired to XsoConfig
//   + CacheResult returned from Init/Get/Put/Save — no silent failures
//   + XsoConfig::resolved_cache_path drives cache directory
//   + XsoConfig::build_hash drives on-disk version check
//   + XsoConfig::max_precompile_shaders enforced in ShaderPrecompiler::Start()
//   + CacheResultName() for logging
//   + Removed unused includes (<iomanip>, <sstream>, <iostream>)
//   + Log helper uses stderr (consistent with gpu_flags.cc)
//   + Save() creates parent dirs before writing
//   + Init() handles XsoConfig::enable_shader_cache=false gracefully
// =============================================================================

#include "shader_cache.h"
#include "gpu_flags.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

namespace fs = std::filesystem;

namespace xe {
namespace gpu {

// ---------------------------------------------------------------------------
// Internal logger — swap for Xenia XELOGI/XELOGW when integrating natively
// ---------------------------------------------------------------------------
namespace {

void XsoLog(const char *level, const char *fmt, ...) noexcept {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  std::fprintf(stderr, "[XSO][%s][Cache] %s\n", level, buf);
}

} // namespace

#define SC_INFO(fmt, ...) XsoLog("INFO ", fmt, ##__VA_ARGS__)
#define SC_WARN(fmt, ...) XsoLog("WARN ", fmt, ##__VA_ARGS__)
#define SC_ERROR(fmt, ...) XsoLog("ERROR", fmt, ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// CacheResultName
// ---------------------------------------------------------------------------

std::string_view CacheResultName(CacheResult r) noexcept {
  switch (r) {
  case CacheResult::kOk:
    return "Ok";
  case CacheResult::kNotFound:
    return "NotFound";
  case CacheResult::kCorrupt:
    return "Corrupt";
  case CacheResult::kVersionMismatch:
    return "VersionMismatch";
  case CacheResult::kIOError:
    return "IOError";
  case CacheResult::kDisabled:
    return "Disabled";
  }
  return "Unknown";
}

// ---------------------------------------------------------------------------
// CRC32 — constexpr table (IEEE 0xEDB88320)
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t BuildCRC32Entry(uint32_t i) noexcept {
  uint32_t c = i;
  for (int j = 0; j < 8; ++j)
    c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
  return c;
}

constexpr auto kCRC32Table = []() {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i)
    t[i] = BuildCRC32Entry(i);
  return t;
}();

} // namespace

uint32_t CRC32(const void *data, size_t len) noexcept {
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i)
    crc = kCRC32Table[(crc ^ bytes[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// FNV-1a 64-bit shader hash
// ---------------------------------------------------------------------------

uint64_t ShaderHash(const void *data, size_t len) noexcept {
  constexpr uint64_t kOffset = 0xcbf29ce484222325ULL;
  constexpr uint64_t kPrime = 0x00000100000001b3ULL;
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint64_t h = kOffset;
  for (size_t i = 0; i < len; ++i) {
    h ^= static_cast<uint64_t>(bytes[i]);
    h *= kPrime;
  }
  return h;
}

// ---------------------------------------------------------------------------
// BloomFilter
// ---------------------------------------------------------------------------

uint64_t BloomFilter::H1(uint64_t x) noexcept {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

uint64_t BloomFilter::H2(uint64_t x) noexcept {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

void BloomFilter::Insert(uint64_t hash) noexcept {
  const uint64_t h1 = H1(hash);
  const uint64_t h2 = H2(hash);
  for (uint64_t i = 0; i < 4; ++i)
    bits_.set((h1 + i * h2) % kBits);
}

bool BloomFilter::MightContain(uint64_t hash) const noexcept {
  const uint64_t h1 = H1(hash);
  const uint64_t h2 = H2(hash);
  for (uint64_t i = 0; i < 4; ++i)
    if (!bits_.test((h1 + i * h2) % kBits))
      return false;
  return true;
}

void BloomFilter::Clear() noexcept { bits_.reset(); }

double BloomFilter::FalsePositiveRate(size_t n) const noexcept {
  const double exp_arg =
      -4.0 * static_cast<double>(n) / static_cast<double>(kBits);
  return std::pow(1.0 - std::exp(exp_arg), 4.0);
}

// ---------------------------------------------------------------------------
// ShaderCache — singleton
// ---------------------------------------------------------------------------

ShaderCache &ShaderCache::Get() noexcept {
  static ShaderCache instance;
  return instance;
}

// ---------------------------------------------------------------------------
// ValidateHeader
// ---------------------------------------------------------------------------

CacheResult
ShaderCache::ValidateHeader(const ShaderCacheHeader &h) const noexcept {
  if (h.magic != 0x58534F43u) {
    SC_WARN("Bad magic: 0x%08X", h.magic);
    return CacheResult::kCorrupt;
  }
  if (h.version != 3) {
    SC_WARN("Version mismatch: file=%u expected=3", h.version);
    return CacheResult::kVersionMismatch;
  }
  // Build hash check — PRD §4.1 / §6
  const auto &cfg = XsoConfig::Get();
  if (std::string_view(h.build_hash) != cfg.build_hash) {
    SC_WARN("Build hash mismatch: cached='%s' current='%s'", h.build_hash,
            cfg.build_hash.c_str());
    return CacheResult::kVersionMismatch;
  }
  return CacheResult::kOk;
}

// ---------------------------------------------------------------------------
// Init — call once after XsoConfig::Get().Init()
// ---------------------------------------------------------------------------

CacheResult ShaderCache::Init() {
  const auto &cfg = XsoConfig::Get();

  // PRD §4.3: enable_shader_cache=false → skip entirely
  if (!cfg.enable_shader_cache) {
    SC_INFO("enable_shader_cache=false — cache disabled");
    return CacheResult::kDisabled;
  }

  if (cfg.resolved_cache_path.empty()) {
    SC_ERROR("resolved_cache_path is empty — call XsoConfig::Init() first");
    return CacheResult::kIOError;
  }

  cache_file_ = fs::path(cfg.resolved_cache_path) / "shaders.xshc";
  SC_INFO("Cache file: %s", cache_file_.string().c_str());

  // Force rebuild if flag set
  if (cfg.shader_cache_invalidation_required) {
    SC_WARN("Invalidation flag set — skipping load");
    return CacheResult::kOk;
  }

  return LoadFromDisk();
}

// ---------------------------------------------------------------------------
// LoadFromDisk
// ---------------------------------------------------------------------------

CacheResult ShaderCache::LoadFromDisk() {
  if (!fs::exists(cache_file_)) {
    SC_INFO("No cache file — cold start");
    return CacheResult::kOk;
  }

  std::ifstream f(cache_file_, std::ios::binary);
  if (!f) {
    SC_ERROR("Cannot open cache file for reading");
    return CacheResult::kIOError;
  }

  ShaderCacheHeader hdr{};
  f.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
  if (!f)
    return CacheResult::kIOError;

  CacheResult hdr_result = ValidateHeader(hdr);
  if (hdr_result != CacheResult::kOk) {
    Invalidate();
    return hdr_result;
  }

  size_t loaded = 0;
  size_t corrupt = 0;

  for (uint32_t i = 0; i < hdr.entry_count; ++i) {
    ShaderCacheEntry entry{};
    f.read(reinterpret_cast<char *>(&entry), sizeof(entry));
    if (!f)
      break;

    std::vector<uint8_t> data(entry.blob_size);
    f.read(reinterpret_cast<char *>(data.data()), entry.blob_size);
    if (!f)
      break;

    // Corruption guard — PRD §6
    if (CRC32(data.data(), data.size()) != entry.crc32) {
      SC_WARN("CRC mismatch on hash=0x%016llX — skipping",
              static_cast<unsigned long long>(entry.shader_hash));
      ++corrupt;
      continue;
    }

    ShaderBlob blob;
    blob.data = std::move(data);
    blob.shader_type = entry.shader_type;
    blob.flags = entry.flags;
    blob.backend = entry.backend;

    lru_.Put(entry.shader_hash, std::move(blob));
    bloom_.Insert(entry.shader_hash);
    ++loaded;
  }

  SC_INFO("Loaded %zu shaders (%zu corrupt). Bloom FPR=%.4f", loaded, corrupt,
          bloom_.FalsePositiveRate(loaded));
  return CacheResult::kOk;
}

// ---------------------------------------------------------------------------
// Get — Bloom fast path → LRU
// ---------------------------------------------------------------------------

CacheResult ShaderCache::Get(uint64_t hash, ShaderBlob *out) noexcept {
  if (!bloom_.MightContain(hash))
    return CacheResult::kNotFound;

  ShaderBlob *found = lru_.Get(hash);
  if (!found)
    return CacheResult::kNotFound;

  if (out)
    *out = *found;
  return CacheResult::kOk;
}

// ---------------------------------------------------------------------------
// Put
// ---------------------------------------------------------------------------

CacheResult ShaderCache::Put(uint64_t hash, ShaderBlob blob) {
  const auto &cfg = XsoConfig::Get();
  if (!cfg.enable_shader_cache)
    return CacheResult::kDisabled;

  lru_.Put(hash, std::move(blob));
  bloom_.Insert(hash);
  return CacheResult::kOk;
}

// ---------------------------------------------------------------------------
// Save — atomic write (tmp → rename)  PRD §6: no corrupt half-writes
// ---------------------------------------------------------------------------

CacheResult ShaderCache::Save() const {
  const auto &cfg = XsoConfig::Get();
  if (!cfg.enable_shader_cache)
    return CacheResult::kDisabled;

  fs::path tmp = cache_file_;
  tmp += ".tmp";

  // Ensure directory exists
  std::error_code ec;
  fs::create_directories(cache_file_.parent_path(), ec);
  if (ec) {
    SC_ERROR("Cannot create cache dir: %s", ec.message().c_str());
    return CacheResult::kIOError;
  }

  std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
  if (!f) {
    SC_ERROR("Cannot open tmp file for writing: %s", tmp.string().c_str());
    return CacheResult::kIOError;
  }

  // Build header
  ShaderCacheHeader hdr{};
  hdr.entry_count = static_cast<uint32_t>(lru_.Size());
  {
    const size_t copy_len = std::min(cfg.build_hash.size(), sizeof(hdr.build_hash) - 1);
    std::memcpy(hdr.build_hash, cfg.build_hash.data(), copy_len);
    hdr.build_hash[copy_len] = '\0';
  }

  f.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

  lru_.ForEach([&](const uint64_t &hash, const ShaderBlob &blob) {
    ShaderCacheEntry entry{};
    entry.shader_hash = hash;
    entry.blob_size = static_cast<uint32_t>(blob.data.size());
    entry.crc32 = CRC32(blob.data.data(), blob.data.size());
    entry.shader_type = blob.shader_type;
    entry.flags = blob.flags;
    entry.backend = blob.backend;

    f.write(reinterpret_cast<const char *>(&entry), sizeof(entry));
    f.write(reinterpret_cast<const char *>(blob.data.data()), blob.data.size());
  });

  f.close();

  // Atomic rename — PRD §6 risk mitigation
  fs::rename(tmp, cache_file_, ec);
  if (ec) {
    SC_ERROR("Atomic rename failed: %s", ec.message().c_str());
    return CacheResult::kIOError;
  }

  SC_INFO("Saved %zu shaders to %s", lru_.Size(), cache_file_.string().c_str());
  return CacheResult::kOk;
}

// ---------------------------------------------------------------------------
// Invalidate
// ---------------------------------------------------------------------------

void ShaderCache::Invalidate() {
  lru_.Clear();
  bloom_.Clear();

  if (!cache_file_.empty()) {
    std::error_code ec;
    fs::remove(cache_file_, ec);
    if (!ec)
      SC_INFO("Cache file removed: %s", cache_file_.string().c_str());
  }

  SC_INFO("Cache invalidated");
}

// ---------------------------------------------------------------------------
// EntryCount / PrintStats
// ---------------------------------------------------------------------------

size_t ShaderCache::EntryCount() const noexcept { return lru_.Size(); }

void ShaderCache::PrintStats() const {
  SC_INFO("Entries=%zu  BloomFPR=%.4f  File=%s", lru_.Size(),
          bloom_.FalsePositiveRate(lru_.Size()), cache_file_.string().c_str());
}

// ---------------------------------------------------------------------------
// ShaderPrecompiler
// ---------------------------------------------------------------------------

void ShaderPrecompiler::Start(std::vector<uint64_t> hashes,
                              CompileFn compile_fn) {
  const auto &cfg = XsoConfig::Get();

  // Enforce PRD §4.2 cap
  if (cfg.max_precompile_shaders > 0 &&
      hashes.size() > cfg.max_precompile_shaders) {
    hashes.resize(cfg.max_precompile_shaders);
    SC_INFO("Precompile cap applied: %u shaders", cfg.max_precompile_shaders);
  }

  cancel_flag_.store(false, std::memory_order_relaxed);

  worker_ = std::async(std::launch::async, [hashes = std::move(hashes),
                                            compile_fn = std::move(compile_fn),
                                            this]() mutable {
    auto &cache = ShaderCache::Get();
    size_t compiled = 0;

    for (uint64_t hash : hashes) {
      if (cancel_flag_.load(std::memory_order_relaxed))
        break;

      // Skip already cached
      ShaderBlob existing;
      if (cache.Get(hash, &existing) == CacheResult::kOk)
        continue;

      ShaderBlob blob = compile_fn(hash);
      if (!blob.data.empty()) {
        auto put_result = cache.Put(hash, std::move(blob));
        if (put_result != CacheResult::kOk) {
          SC_WARN("Put failed: %s",
                  std::string(CacheResultName(put_result)).c_str());
        }
        ++compiled;
      }
    }

    SC_INFO("Precompile done: %zu shaders compiled%s", compiled,
            cancel_flag_.load() ? " (cancelled)" : "");
  });
}

void ShaderPrecompiler::Cancel() noexcept {
  cancel_flag_.store(true, std::memory_order_relaxed);
}

bool ShaderPrecompiler::IsRunning() const noexcept {
  if (!worker_.valid())
    return false;
  return worker_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
}

void ShaderPrecompiler::Wait() {
  if (worker_.valid())
    worker_.wait();
}

} // namespace gpu
} // namespace xe
