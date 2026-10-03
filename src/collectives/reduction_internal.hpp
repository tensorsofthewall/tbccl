#pragma once

// Internal (non-installed) typed element-wise reduction kernel, shared
// by every algorithm variant of reduce()/reduce_scatter()/all_reduce()
// that needs to combine two buffers of the same DataType with a
// ReduceOp. Not part of the public tbccl:: surface.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <tbccl/reduction.hpp>

#include "low_precision.hpp"

namespace tbccl::detail
{

// Compile-time tag carrying the element type of a DataType, passed to the visitor's generic lambda.
template <typename T>
struct TypeTag
{
    using type = T;
};

// The single DataType -> element-type dispatcher shared by every reducing algorithm and the host reduce backend, so a new
// reduction type is added in exactly one place. Calls fn(TypeTag<T>{}) for the matching T; throws for a DataType that has
// no reduction arithmetic (callers have normally rejected it earlier with validate_reduction()).
template <typename Fn>
void visit_reduction_type(DataType datatype, Fn &&fn)
{
    switch (datatype)
    {
    case DataType::Int32: fn(TypeTag<std::int32_t>{}); return;
    case DataType::Int64: fn(TypeTag<std::int64_t>{}); return;
    case DataType::Float32: fn(TypeTag<float>{}); return;
    case DataType::Float64: fn(TypeTag<double>{}); return;
    case DataType::Float16: fn(TypeTag<lowp::Half>{}); return;
    case DataType::BFloat16: fn(TypeTag<lowp::BFloat16>{}); return;
    case DataType::Int8: fn(TypeTag<std::int8_t>{}); return;
    case DataType::UInt8: fn(TypeTag<std::uint8_t>{}); return;
    }
    throw std::runtime_error("unsupported: no reduction arithmetic for this DataType");
}

namespace reduction_detail
{

    template <typename T>
    struct UnsignedOf;

    template <>
    struct UnsignedOf<std::int8_t>
    {
        using type = std::uint8_t;
    };

    template <>
    struct UnsignedOf<std::uint8_t>
    {
        using type = std::uint8_t;
    };

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
        // The inner cast narrows the (possibly int-promoted, for 8-bit operands) sum back to U, which is modulo 2^N by definition.
        return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
    }

    template <typename T>
    T wrapping_mul(T a, T b)
    {
        using U = typename UnsignedOf<T>::type;
        // Unsigned 8-bit operands promote to int; multiply through a wide enough unsigned type so the product can never overflow int.
        using W = std::conditional_t<(sizeof(U) < sizeof(unsigned)), unsigned, U>;
        return static_cast<T>(static_cast<U>(static_cast<W>(a) * static_cast<W>(b)));
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
