#pragma once
// =============================================================================
// shader_cache.h
// Persistent Shader Cache — Header
// Integrates into: src/xenia/gpu/shader_cache.h
//
// v3 changes:
//   + XsoConfig integration — vendor path, build hash, invalidation
//   + max_precompile_shaders cap enforced in ShaderPrecompiler
//   + CacheResult error enum — no silent failures
//   + [[nodiscard]] on all load/save/get/put paths
//   + noexcept on all non-throwing paths
//   + BloomFilter::Clear() for cache rebuild without reallocation
//   + ShaderCacheHeader embeds build_hash for on-disk version check
// =============================================================================

#ifndef XENIA_GPU_SHADER_CACHE_H_
#define XENIA_GPU_SHADER_CACHE_H_

#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <list>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace xe {
namespace gpu {

// ---------------------------------------------------------------------------
// Shader type classification (detected by CUDA kernel)
// ---------------------------------------------------------------------------
enum class ShaderType : uint8_t {
  kUnknown = 0,
  kVertex = 1,
  kPixel = 2,
  kCompute = 3,
};

// Shader flags — set by ClassifyShadersKernel
enum ShaderFlags : uint8_t {
  kFlagNone = 0x00,
  kFlagMsaa = 0x01,
  kFlagBlur = 0x02,
  kFlagShadow = 0x04,
  kFlagPostProcess = 0x08,
};

// ---------------------------------------------------------------------------
// CacheResult — explicit error reporting on all cache operations
// ---------------------------------------------------------------------------
enum class CacheResult : uint8_t {
  kOk = 0,
  kNotFound = 1,
  kCorrupt = 2,         // checksum mismatch
  kVersionMismatch = 3, // build hash changed → rebuild
  kIOError = 4,         // disk read/write failure
  kDisabled = 5,        // enable_shader_cache=false
};

std::string_view CacheResultName(CacheResult r) noexcept;

// ---------------------------------------------------------------------------
// On-disk structures
// All packed — no padding surprises across compiler versions.
// ---------------------------------------------------------------------------

#pragma pack(push, 1)

// File header — written once at start of each .bin cache file.
struct ShaderCacheHeader {
  uint32_t magic = 0x58534F43; // 'XSOC'
  uint32_t version = 3;        // bump on breaking layout change
  uint32_t entry_count = 0;
  uint64_t title_id = 0;
  char build_hash[48] = {}; // XSO_BUILD_HASH null-terminated
  uint32_t reserved = 0;
};
static_assert(sizeof(ShaderCacheHeader) == 72,
              "ShaderCacheHeader layout changed");

// Per-shader entry stored after header.
struct ShaderCacheEntry {
  uint64_t shader_hash = 0;
  uint32_t blob_size = 0;
  uint32_t crc32 = 0; // checksum of blob bytes
  uint64_t last_used_frame = 0;
  ShaderType shader_type = ShaderType::kUnknown;
  uint8_t flags = kFlagNone;
  uint8_t backend = 0; // 0=Vulkan 1=D3D12
  uint8_t pad = 0;
};
static_assert(sizeof(ShaderCacheEntry) == 28,
              "ShaderCacheEntry layout changed");

#pragma pack(pop)

// In-memory blob carrying compiled shader bytes.
struct ShaderBlob {
  std::vector<uint8_t> data;
  ShaderType shader_type = ShaderType::kUnknown;
  uint8_t flags = kFlagNone;
  uint8_t backend = 0;
};

// ---------------------------------------------------------------------------
// BloomFilter — probabilistic pre-check before disk lookup
// k=4 hash functions, 64 KiB bitset → ~0.2% false positive at 100k entries
// ---------------------------------------------------------------------------
class BloomFilter {
public:
  static constexpr size_t kBits = 512 * 1024; // 64 KiB

  void Insert(uint64_t hash) noexcept;
  bool MightContain(uint64_t hash) const noexcept;
  void Clear() noexcept;
  double FalsePositiveRate(size_t n_items) const noexcept;

private:
  std::bitset<kBits> bits_;

  // Double-hashing: h_i(x) = h1(x) + i * h2(x)
  static uint64_t H1(uint64_t x) noexcept;
  static uint64_t H2(uint64_t x) noexcept;
};

// ---------------------------------------------------------------------------
// LRUCache — O(1) get/put/evict, fixed capacity
// ---------------------------------------------------------------------------
template <typename K, typename V> class LRUCache {
public:
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  // Returns nullptr if not found. Promotes to MRU on hit.
  V *Get(const K &key) noexcept;

  // Inserts or updates. Evicts LRU entry if at capacity.
  void Put(K key, V value);

  // Iterate in MRU → LRU order (used by Save()).
  void ForEach(std::function<void(const K &, const V &)> fn) const;

  size_t Size() const noexcept { return map_.size(); }
  size_t Capacity() const noexcept { return capacity_; }
  void Clear() noexcept;

private:
  using Pair = std::pair<K, V>;
  using ListIt = typename std::list<Pair>::iterator;

  size_t capacity_;
  std::list<Pair> list_;
  std::unordered_map<K, ListIt> map_;
};

// ---------------------------------------------------------------------------
// Hash / checksum utilities
// ---------------------------------------------------------------------------
uint64_t ShaderHash(const void *data, size_t size) noexcept; // FNV-1a 64-bit
uint32_t CRC32(const void *data, size_t size) noexcept;

// ---------------------------------------------------------------------------
// ShaderCache — main interface
//
// Usage (in vulkan_shader_translator.cc / d3d12_shader_translator.cc):
//
//   auto& cache = xe::gpu::ShaderCache::Get();
//   cache.Init();   // once, after XsoConfig::Get().Init()
//
//   // On shader lookup:
//   ShaderBlob blob;
//   if (cache.Get(microcode_hash, &blob) == CacheResult::kOk) { use(blob); }
//   else { compile(); cache.Put(microcode_hash, compiled_blob); }
// ---------------------------------------------------------------------------
class ShaderCache {
public:
  static ShaderCache &Get() noexcept;

  // Init — call once after XsoConfig::Get().Init().
  // Loads existing cache from disk; triggers rebuild on hash mismatch.
  [[nodiscard]] CacheResult Init();

  // Look up a compiled shader blob by microcode hash.
  [[nodiscard]] CacheResult Get(uint64_t hash, ShaderBlob *out) noexcept;

  // Insert a newly compiled shader blob.
  [[nodiscard]] CacheResult Put(uint64_t hash, ShaderBlob blob);

  // Persist all in-memory entries to disk atomically (tmp → rename).
  [[nodiscard]] CacheResult Save() const;

  // Wipe on-disk cache and clear in-memory state.
  void Invalidate();

  size_t EntryCount() const noexcept;
  void PrintStats() const;

private:
  ShaderCache() = default;

  [[nodiscard]] CacheResult LoadFromDisk();
  [[nodiscard]] CacheResult
  ValidateHeader(const ShaderCacheHeader &h) const noexcept;

  std::filesystem::path cache_file_;
  BloomFilter bloom_;
  LRUCache<uint64_t, ShaderBlob> lru_{2048};
  uint64_t frame_index_ = 0;
};

// ---------------------------------------------------------------------------
// ShaderPrecompiler — background compile thread
//
// Respects XsoConfig::max_precompile_shaders cap.
// Cancel() is safe to call from any thread.
// ---------------------------------------------------------------------------
class ShaderPrecompiler {
public:
  using CompileFn = std::function<ShaderBlob(uint64_t hash)>;

  // hashes: top-N most frequent shaders from Phase 1 log analysis
  // compile_fn: wraps Xenia's JIT compiler
  void Start(std::vector<uint64_t> hashes, CompileFn compile_fn);

  void Cancel() noexcept;
  bool IsRunning() const noexcept;

  // Block until done or cancelled.
  void Wait();

private:
  std::atomic<bool> cancel_flag_{false};
  std::future<void> worker_;
};

} // namespace gpu
} // namespace xe

// ---------------------------------------------------------------------------
// LRUCache template implementation (header-only — avoids linker issues)
// ---------------------------------------------------------------------------
namespace xe {
namespace gpu {

template <typename K, typename V>
V *LRUCache<K, V>::Get(const K &key) noexcept {
  auto it = map_.find(key);
  if (it == map_.end())
    return nullptr;
  // Promote to front (MRU)
  list_.splice(list_.begin(), list_, it->second);
  return &it->second->second;
}

template <typename K, typename V> void LRUCache<K, V>::Put(K key, V value) {
  auto it = map_.find(key);
  if (it != map_.end()) {
    it->second->second = std::move(value);
    list_.splice(list_.begin(), list_, it->second);
    return;
  }
  if (map_.size() >= capacity_) {
    // Evict LRU (back of list)
    map_.erase(list_.back().first);
    list_.pop_back();
  }
  list_.emplace_front(key, std::move(value));
  map_[key] = list_.begin();
}

template <typename K, typename V>
void LRUCache<K, V>::ForEach(
    std::function<void(const K &, const V &)> fn) const {
  for (const auto &[k, v] : list_)
    fn(k, v);
}

template <typename K, typename V> void LRUCache<K, V>::Clear() noexcept {
  list_.clear();
  map_.clear();
}

} // namespace gpu
} // namespace xe

#endif // XENIA_GPU_SHADER_CACHE_H_
