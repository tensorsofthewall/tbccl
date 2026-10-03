#pragma once

// The one place the CUDA reduction backends define how two elements are summed, shared by cuda_reduce_backend.cu and
// cuda_external_async_backend.cu (device code lives under benchmarks/tensor only; the core library never sees CUDA headers).
//
// Float16 / BFloat16 follow the documented host semantics (src/collectives/low_precision.hpp): widen both operands to float32, ONE float32
// add, round once to the target format with round-to-nearest-even. The native half/bfloat16 add instructions are deliberately NOT used:
// they round the exact sum once, which can differ from "float32 add, then round" in rare double-rounding cases, and host and device
// must agree bit for bit. Signed integer SUM wraps modulo 2^N through the matching unsigned type (never signed-overflow UB).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace tbccl_bench::tensor::cuda_reduce
{

template <typename T>
__device__ __forceinline__ T sum_elem(T a, T b)
{
    if constexpr (std::is_integral_v<T>)
    {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
    }
    else
    {
        return a + b;
    }
}

template <>
__device__ __forceinline__ __half sum_elem<__half>(__half a, __half b)
{
    return __float2half_rn(__half2float(a) + __half2float(b));
}

template <>
__device__ __forceinline__ __nv_bfloat16 sum_elem<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
}

} // namespace tbccl_bench::tensor::cuda_reduce
