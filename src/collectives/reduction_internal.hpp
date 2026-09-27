#pragma once

// Internal (non-installed) typed element-wise reduction kernel, shared
// by every algorithm variant of reduce()/reduce_scatter()/all_reduce()
// that needs to combine two buffers of the same DataType with a
// ReduceOp. Not part of the public tbccl:: surface.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <tbccl/reduction.hpp>

namespace tbccl::detail
{
namespace reduction_detail
{

    template <typename T>
    struct UnsignedOf;

    template <>
    struct UnsignedOf<std::int32_t>
    {
        using type = std::uint32_t;
    };

    template <>
    struct UnsignedOf<std::int64_t>
    {
        using type = std::uint64_t;
    };

    // Signed Sum/Product go through the corresponding unsigned type so
    // wraparound is two's-complement modular arithmetic, never signed-
    // overflow undefined behavior.
    template <typename T>
    T wrapping_add(T a, T b)
    {
        using U = typename UnsignedOf<T>::type;
        return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
    }

    template <typename T>
    T wrapping_mul(T a, T b)
    {
        using U = typename UnsignedOf<T>::type;
        return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
    }

} // namespace reduction_detail

// dst[i] = dst[i] OP src[i] for i in [0, count). Signed integer
// Sum/Product wrap via the matching unsigned type; Float32/Float64 use
// ordinary C++ arithmetic; Min/Max use std::min/std::max for every
// type. This is the single reduction kernel shared by every
// reduce()-family algorithm variant (reference and ring alike), so
// their element-wise semantics can never diverge.
template <typename T>
void apply_reduction(T *dst, const T *src, std::size_t count, ReduceOp op)
{
    switch (op)
    {
    case ReduceOp::Sum:
        for (std::size_t i = 0; i < count; ++i)
        {
            if constexpr (std::is_integral_v<T>)
            {
                dst[i] = reduction_detail::wrapping_add(dst[i], src[i]);
            }
            else
            {
                dst[i] = dst[i] + src[i];
            }
        }
        break;

    case ReduceOp::Product:
        for (std::size_t i = 0; i < count; ++i)
        {
            if constexpr (std::is_integral_v<T>)
            {
                dst[i] = reduction_detail::wrapping_mul(dst[i], src[i]);
            }
            else
            {
                dst[i] = dst[i] * src[i];
            }
        }
        break;

    case ReduceOp::Min:
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = std::min(dst[i], src[i]);
        }
        break;

    case ReduceOp::Max:
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = std::max(dst[i], src[i]);
        }
        break;
    }
}

} // namespace tbccl::detail
