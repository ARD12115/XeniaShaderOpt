#pragma once
// =============================================================================
//  shader_hash_cuda.h
//  GPU-Parallel Shader Hashing + Classification — Header
//  Integrates into: cuda/shader_hash_cuda.h
// =============================================================================

#ifndef XENIA_GPU_SHADER_HASH_CUDA_H_
#define XENIA_GPU_SHADER_HASH_CUDA_H_

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace xe {
namespace gpu {
namespace cuda {

// ── Per-shader input descriptor ──────────────────────────────────────────────

struct ShaderMicrocodeDesc {
  uint64_t shader_hash;           // pre-computed or 0 (will be filled)
  std::vector<uint8_t> microcode; // raw Xbox 360 microcode bytes
};

// ── Main host-side interface
// ──────────────────────────────────────────────────

class CudaShaderProcessor {
public:
  CudaShaderProcessor();
  ~CudaShaderProcessor();

  // Non-copyable
  CudaShaderProcessor(const CudaShaderProcessor &) = delete;
  CudaShaderProcessor &operator=(const CudaShaderProcessor &) = delete;

  // Hash and classify a batch of shaders on the GPU.
  // out_hashes[i] — FNV-1a 64-bit hash of shaders[i].microcode
  // out_flags[i]  — bit0=MSAA  bit1=Blur  bit2=ShadowImpostor
  bool BatchHashAndClassify(const std::vector<ShaderMicrocodeDesc> &shaders,
                            std::vector<uint64_t> &out_hashes,
                            std::vector<uint8_t> &out_flags);

  void PrintDeviceInfo() const;

private:
  cudaStream_t stream_ = nullptr;
  int sm_count_ = 0;
};

} // namespace cuda
} // namespace gpu
} // namespace xe

#endif // XENIA_GPU_SHADER_HASH_CUDA_H_
