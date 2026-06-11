// =============================================================================
//  shader_hash_cuda.cu
//  GPU-Parallel Shader Hashing + Classification
//  Integrates into: cuda/shader_hash_cuda.cu
//
//  What this solves:
//    - Hashing thousands of Xbox 360 shader microcodes on CPU is serial and
//      slow during cache warm-up. This moves the work to the GPU using
//      warp-level primitives for maximum throughput.
//    - Classification detects MSAA, motion blur, and shadow impostor shaders
//      so the cache can tag them (flags byte) and the optimizer can handle
//      them differently (e.g. skip or simplify on mid-range GPUs).
//
//  Architecture targets: SM 8.6 (RTX 3050 Ti, Ampere)
//  Requires: CUDA Toolkit 12+, C++17
//
//  Kernel overview:
//    1. HashShadersKernel      — warp-level FNV-1a over microcode blobs
//    2. ClassifyShadersKernel  — pattern-match opcodes → flags byte
//    3. BatchHashAndClassify() — host-side launcher (pinned memory, streams)
// =============================================================================

#include "shader_hash_cuda.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

// =============================================================================
//  CUDA error check macro
// =============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[CUDA] %s:%d  %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(_e));                                         \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

// =============================================================================
//  Constants
// =============================================================================

// FNV-1a 64-bit constants
static constexpr uint64_t kFNVOffset = 0xcbf29ce484222325ULL;
static constexpr uint64_t kFNVPrime = 0x00000100000001b3ULL;

// Xbox 360 shader opcode patterns (simplified fingerprints)
// Real implementation would use full microcode spec tables.
static constexpr uint32_t kOpcodeALUMAA = 0x0034;     // MSAA resolve ALU
static constexpr uint32_t kOpcodeBlurSample = 0x0078; // Gaussian tap sample
static constexpr uint32_t kOpcodeShadowCmp = 0x00A1;  // Shadow map compare
static constexpr uint32_t kOpcodeImpostorUV = 0x00B3; // Impostor UV remap

// Flag bits (match ShaderCacheEntry::flags in shader_cache.h)
static constexpr uint8_t kFlagMSAA = 1u << 0;
static constexpr uint8_t kFlagBlur = 1u << 1;
static constexpr uint8_t kFlagShadowImpostor = 1u << 2;

// Threads per block — warp multiple, tuned for Ampere L1
static constexpr int kBlockSize = 128;

// Max microcode size we'll process per shader (in uint32_t words)
static constexpr int kMaxMicrocodeWords = 1024;

// =============================================================================
//  Device helpers
// =============================================================================

// Warp-level XOR reduction (used to combine partial hashes across lanes)
__device__ __forceinline__ uint64_t WarpReduceXOR(uint64_t val) {
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1)
    val ^= __shfl_xor_sync(0xFFFFFFFF, val, offset);
  return val;
}

// Single-word FNV-1a step
__device__ __forceinline__ uint64_t FNVStep(uint64_t hash, uint8_t byte) {
  return (hash ^ static_cast<uint64_t>(byte)) * kFNVPrime;
}

// =============================================================================
//  Kernel 1 — HashShadersKernel
//
//  Grid:  (num_shaders) blocks
//  Block: kBlockSize threads
//
//  Each block handles one shader. Threads stride over the microcode words,
//  compute partial FNV-1a hashes, then warp-reduce via XOR to produce one
//  64-bit hash per warp. Block reduces warp results into shared memory.
// =============================================================================

__global__ void HashShadersKernel(
    const uint8_t *__restrict__ microcode_pool, // flat array of all microcodes
    const uint32_t *__restrict__ offsets,       // byte offset of each shader
    const uint32_t *__restrict__ lengths,       // byte length of each shader
    uint64_t *__restrict__ out_hashes,          // one hash per shader
    uint32_t num_shaders) {
  const uint32_t shader_id = blockIdx.x;
  if (shader_id >= num_shaders)
    return;

  const uint8_t *mc = microcode_pool + offsets[shader_id];
  const uint32_t len = lengths[shader_id];

  // Each thread accumulates a partial FNV-1a over its strided bytes
  uint64_t partial = kFNVOffset;
  for (uint32_t i = threadIdx.x; i < len; i += blockDim.x)
    partial = FNVStep(partial, mc[i]);

  // Warp-level XOR reduction
  uint64_t warp_hash = WarpReduceXOR(partial);

  // Shared memory: one slot per warp
  __shared__ uint64_t smem[kBlockSize / 32];
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  if (lane_id == 0)
    smem[warp_id] = warp_hash;

  __syncthreads();

  // First warp reduces across all warp results
  if (warp_id == 0) {
    uint64_t block_hash =
        (threadIdx.x < (blockDim.x / 32)) ? smem[threadIdx.x] : kFNVOffset;
    block_hash = WarpReduceXOR(block_hash);

    if (lane_id == 0)
      out_hashes[shader_id] = block_hash;
  }
}

// =============================================================================
//  Kernel 2 — ClassifyShadersKernel
//
//  Grid:  (num_shaders) blocks
//  Block: kBlockSize threads
//
//  Each block scans one shader's microcode for known opcode patterns.
//  Threads vote via warp ballot; any match sets the corresponding flag bit.
// =============================================================================

__global__ void ClassifyShadersKernel(
    const uint32_t *__restrict__ microcode_pool_u32, // pool as 32-bit words
    const uint32_t *__restrict__ offsets_u32,        // word offsets
    const uint32_t *__restrict__ lengths_u32,        // word counts
    uint8_t *__restrict__ out_flags, // one flags byte per shader
    uint32_t num_shaders) {
  const uint32_t shader_id = blockIdx.x;
  if (shader_id >= num_shaders)
    return;

  const uint32_t *mc = microcode_pool_u32 + offsets_u32[shader_id];
  const uint32_t len =
      min(lengths_u32[shader_id], (uint32_t)kMaxMicrocodeWords);

  bool found_msaa = false;
  bool found_blur = false;
  bool found_shadow = false;

  // Each thread checks its strided words
  for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
    uint32_t word = mc[i];

    // Extract opcode from upper 16 bits (Xbox 360 microcode convention)
    uint32_t opcode = (word >> 16) & 0xFFFF;

    if (opcode == kOpcodeALUMAA)
      found_msaa = true;
    if (opcode == kOpcodeBlurSample)
      found_blur = true;
    if (opcode == kOpcodeShadowCmp || opcode == kOpcodeImpostorUV)
      found_shadow = true;
  }

  // Warp ballot: any lane found a pattern → flag is set for this shader
  uint32_t ballot_msaa = __ballot_sync(0xFFFFFFFF, found_msaa);
  uint32_t ballot_blur = __ballot_sync(0xFFFFFFFF, found_blur);
  uint32_t ballot_shadow = __ballot_sync(0xFFFFFFFF, found_shadow);

  // Shared flags accumulation across warps
  __shared__ uint8_t smem_flags;
  if (threadIdx.x == 0)
    smem_flags = 0;
  __syncthreads();

  const int lane_id = threadIdx.x % 32;
  if (lane_id == 0) {
    uint8_t warp_flags = 0;
    if (ballot_msaa)
      warp_flags |= kFlagMSAA;
    if (ballot_blur)
      warp_flags |= kFlagBlur;
    if (ballot_shadow)
      warp_flags |= kFlagShadowImpostor;
    atomicOr(reinterpret_cast<int *>(&smem_flags),
             static_cast<int>(warp_flags));
  }
  __syncthreads();

  if (threadIdx.x == 0)
    out_flags[shader_id] = smem_flags;
}

// =============================================================================
//  Persistent thread kernel — continuous compilation queue
//
//  Worker threads spin on a queue of shader indices. New work can be pushed
//  by the host without relaunching the kernel, minimising kernel launch
//  overhead during gameplay.
// =============================================================================

struct ShaderWorkQueue {
  uint32_t *indices; // indices into microcode pool
  int32_t head;      // atomic read cursor
  int32_t tail;      // atomic write cursor (host-updated)
  int32_t total;     // total items ever enqueued (termination guard)
  uint64_t *out_hashes;
  uint8_t *out_flags;
  const uint8_t *mc_pool;
  const uint32_t *offsets;
  const uint32_t *lengths;
  const uint32_t *offsets_u32;
  const uint32_t *lengths_u32;
};

__global__ void PersistentHashWorkerKernel(ShaderWorkQueue *queue,
                                           uint32_t num_shaders) {
  // Each warp claims one shader at a time from the queue
  const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = threadIdx.x % 32;
  const int total_warps = (gridDim.x * blockDim.x) / 32;

  while (true) {
    // Warp leader atomically claims next shader index
    uint32_t shader_id = UINT32_MAX;
    if (lane_id == 0) {
      int idx = atomicAdd(&queue->head, 1);
      shader_id = (idx < queue->total) ? queue->indices[idx] : UINT32_MAX;
    }
    shader_id = __shfl_sync(0xFFFFFFFF, shader_id, 0);
    if (shader_id == UINT32_MAX)
      break; // queue drained

    // Hash
    const uint8_t *mc = queue->mc_pool + queue->offsets[shader_id];
    const uint32_t len = queue->lengths[shader_id];

    uint64_t partial = kFNVOffset;
    for (uint32_t i = lane_id; i < len; i += 32)
      partial = FNVStep(partial, mc[i]);

    uint64_t hash = WarpReduceXOR(partial);
    if (lane_id == 0)
      queue->out_hashes[shader_id] = hash;

    // Classify
    const uint32_t *mc32 = reinterpret_cast<const uint32_t *>(
        queue->mc_pool + queue->offsets_u32[shader_id]);
    const uint32_t wlen =
        min(queue->lengths_u32[shader_id], (uint32_t)kMaxMicrocodeWords);

    bool fm = false, fb = false, fs = false;
    for (uint32_t i = lane_id; i < wlen; i += 32) {
      uint32_t op = (mc32[i] >> 16) & 0xFFFF;
      if (op == kOpcodeALUMAA)
        fm = true;
      if (op == kOpcodeBlurSample)
        fb = true;
      if (op == kOpcodeShadowCmp || op == kOpcodeImpostorUV)
        fs = true;
    }

    uint8_t flags = 0;
    if (__ballot_sync(0xFFFFFFFF, fm))
      flags |= kFlagMSAA;
    if (__ballot_sync(0xFFFFFFFF, fb))
      flags |= kFlagBlur;
    if (__ballot_sync(0xFFFFFFFF, fs))
      flags |= kFlagShadowImpostor;

    if (lane_id == 0)
      queue->out_flags[shader_id] = flags;
  }
}

// =============================================================================
//  Host-side launcher — BatchHashAndClassify
// =============================================================================

namespace xe {
namespace gpu {
namespace cuda {

CudaShaderProcessor::CudaShaderProcessor() {
  CUDA_CHECK(cudaStreamCreate(&stream_));

  // Query device for SM count and clock
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  sm_count_ = prop.multiProcessorCount;
  printf("[CudaShaderProcessor] Device: %s  SMs: %d  SM %.1f\n", prop.name,
         sm_count_, prop.major + prop.minor * 0.1f);
}

CudaShaderProcessor::~CudaShaderProcessor() { cudaStreamDestroy(stream_); }

// ---------------------------------------------------------------------------
//  BatchHashAndClassify
//
//  Input:  flat microcode pool + per-shader offset/length arrays (host)
//  Output: per-shader hash + flags arrays (host)
// ---------------------------------------------------------------------------

bool CudaShaderProcessor::BatchHashAndClassify(
    const std::vector<ShaderMicrocodeDesc> &shaders,
    std::vector<uint64_t> &out_hashes, std::vector<uint8_t> &out_flags) {
  const uint32_t N = static_cast<uint32_t>(shaders.size());
  if (N == 0)
    return true;

  out_hashes.assign(N, 0);
  out_flags.assign(N, 0);

  // ── Build flat pool on host ───────────────────────────────────────────────
  std::vector<uint32_t> h_offsets(N), h_lengths(N);
  std::vector<uint32_t> h_offsets_u32(N), h_lengths_u32(N);
  size_t pool_bytes = 0;

  for (uint32_t i = 0; i < N; ++i) {
    h_offsets[i] = static_cast<uint32_t>(pool_bytes);
    h_lengths[i] = static_cast<uint32_t>(shaders[i].microcode.size());
    pool_bytes += shaders[i].microcode.size();
  }

  // Word-aligned offsets for classification kernel
  for (uint32_t i = 0; i < N; ++i) {
    h_offsets_u32[i] = h_offsets[i] / 4;
    h_lengths_u32[i] = h_lengths[i] / 4;
  }

  // Pad pool to 4-byte alignment
  pool_bytes = (pool_bytes + 3) & ~size_t(3);

  std::vector<uint8_t> h_pool(pool_bytes, 0);
  for (uint32_t i = 0; i < N; ++i)
    std::memcpy(h_pool.data() + h_offsets[i], shaders[i].microcode.data(),
                shaders[i].microcode.size());

  // ── Device allocations ────────────────────────────────────────────────────
  uint8_t *d_pool = nullptr;
  uint32_t *d_offsets = nullptr;
  uint32_t *d_lengths = nullptr;
  uint32_t *d_offsets_u32 = nullptr;
  uint32_t *d_lengths_u32 = nullptr;
  uint64_t *d_hashes = nullptr;
  uint8_t *d_flags = nullptr;

  CUDA_CHECK(cudaMalloc(&d_pool, pool_bytes));
  CUDA_CHECK(cudaMalloc(&d_offsets, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_lengths, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_offsets_u32, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_lengths_u32, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_hashes, N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_flags, N * sizeof(uint8_t)));

  // ── Async H→D copies ──────────────────────────────────────────────────────
  CUDA_CHECK(cudaMemcpyAsync(d_pool, h_pool.data(), pool_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_offsets, h_offsets.data(), N * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_lengths, h_lengths.data(), N * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_offsets_u32, h_offsets_u32.data(),
                             N * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_lengths_u32, h_lengths_u32.data(),
                             N * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream_));

  // ── Kernel launches ───────────────────────────────────────────────────────
  // One block per shader; kBlockSize threads per block
  int grid = static_cast<int>(N);

  HashShadersKernel<<<grid, kBlockSize, 0, stream_>>>(d_pool, d_offsets,
                                                      d_lengths, d_hashes, N);

  ClassifyShadersKernel<<<grid, kBlockSize, 0, stream_>>>(
      reinterpret_cast<const uint32_t *>(d_pool), d_offsets_u32, d_lengths_u32,
      d_flags, N);

  // ── Async D→H copies ──────────────────────────────────────────────────────
  CUDA_CHECK(cudaMemcpyAsync(out_hashes.data(), d_hashes, N * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(out_flags.data(), d_flags, N * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost, stream_));

  // Synchronise stream
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // ── Free device memory ────────────────────────────────────────────────────
  cudaFree(d_pool);
  cudaFree(d_offsets);
  cudaFree(d_lengths);
  cudaFree(d_offsets_u32);
  cudaFree(d_lengths_u32);
  cudaFree(d_hashes);
  cudaFree(d_flags);

  return true;
}

// ---------------------------------------------------------------------------
//  PrintDeviceInfo — diagnostic helper
// ---------------------------------------------------------------------------

void CudaShaderProcessor::PrintDeviceInfo() const {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  printf("[CudaShaderProcessor] %-24s | SM %d.%d | %d SMs | "
         "%.0f MB global | L2 %.0f KB\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.totalGlobalMem / 1024.0 / 1024.0, prop.l2CacheSize / 1024.0);
}

} // namespace cuda
} // namespace gpu
} // namespace xe
