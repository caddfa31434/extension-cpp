#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                               \
                     VLLM_DISPATCH_CASE_FLOATING_AND_BYTE_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_INTEGRAL_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define VLLM_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

namespace extension_cpp {


__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}

// Grid: (num_layers)
template <typename scalar_t>
__global__ void custom_copy_kv_caches_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ src_slot_mapping,
                                   const int64_t* __restrict__ dst_slot_mapping,
                                   const int num_heads, const int head_size, 
                                   const int block_size, const int x) {
  const int layer_idx = blockIdx.x;
  const int64_t token_idx = blockIdx.y;
  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);     // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]); // [num_blocks, num_heads, head_size, block_size]

  const int64_t src_slot_idx = src_slot_mapping[token_idx];
  const int64_t dst_slot_idx = dst_slot_mapping[token_idx];

  const int64_t src_block_idx = src_slot_idx / block_size;
  const int64_t src_block_offset = src_slot_idx % block_size;
  const int64_t dst_block_idx = dst_slot_idx / block_size;
  const int64_t dst_block_offset = dst_slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t src_key_idx =
        src_block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        src_block_offset * x + x_offset;

    const int64_t src_value_idx =
        src_block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        src_block_offset;

    const int64_t dst_key_idx =
        dst_block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        dst_block_offset * x + x_offset;
    const int64_t dst_value_idx =
        dst_block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        dst_block_offset;

    scalar_t dst_key = key_cache[src_key_idx];
    scalar_t dst_value = value_cache[src_value_idx];
    key_cache[dst_key_idx] = dst_key;
    value_cache[dst_value_idx] = dst_value; 
  }

}

  // Note: the key_caches and value_caches vectors are constant but
  // not the Tensors they contain. The vectors need to be const refs
  // in order to satisfy pytorch's C++ operator registration code.
  void custom_copy_kv_caches_cuda(std::vector<torch::Tensor> const &key_caches,
    std::vector<torch::Tensor> const &value_caches,
    const torch::Tensor &src_slot_mapping,
    const torch::Tensor &dst_slot_mapping)
{
  int num_layers = key_caches.size();
  int num_tokens = src_slot_mapping.size(0);
  int num_heads = value_caches[0].size(1);
  int head_size = value_caches[0].size(2);
  int block_size = key_caches[0].size(3);
  int x = key_caches[0].size(4);

  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);

  // Launch the kernel.
  dim3 grid(num_layers, num_tokens);
  dim3 block(std::min(1024, num_heads * head_size));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
      key_caches[0].scalar_type(), "custom_copy_kv_caches_kernel", ([&] {
        custom_copy_kv_caches_kernel<scalar_t><<<grid, block, 0, stream>>>(
            key_cache_ptrs_tensor.data_ptr<int64_t>(),
            value_cache_ptrs_tensor.data_ptr<int64_t>(),
            src_slot_mapping.data_ptr<int64_t>(),
            dst_slot_mapping.data_ptr<int64_t>(),
            num_heads, head_size, block_size, x);
      }));
}


// Registers CUDA implementations for myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("myadd_out", &myadd_out_cuda);
  m.impl("custom_copy_kv_caches", &custom_copy_kv_caches_cuda);
}

}
