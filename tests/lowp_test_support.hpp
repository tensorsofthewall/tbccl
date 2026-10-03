#pragma once

// Phase 49 test support shared by the host and CUDA low-precision tests: format traits for Float16 / BFloat16 (independent exact decode,
// NaN predicate, the documented SUM semantics via the private conversion header), deterministic pattern generators and the edge-pattern lists.

#include "low_precision.hpp"

#include <tbccl/reduction.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lowp_test
{

namespace lowp = tbccl::detail::lowp;

// ----- format traits -------------------------------------------------------------------------------------------------------
struct Fp16
{
    static constexpr const char *name = "float16";
    static constexpr tbccl::DataType dt = tbccl::DataType::Float16;
    using Elem = lowp::Half;
    static constexpr std::uint16_t kInf = 0x7C00, kLastFinite = 0x7BFF;
    static float dec(std::uint16_t b) { return lowp::fp16_bits_to_float(b); }
    static std::uint16_t enc(float f) { return lowp::float_to_fp16_bits_rne(f); }
    static bool is_nan(std::uint16_t b) { return (b & 0x7C00) == 0x7C00 && (b & 0x03FF) != 0; }
    // Independent exact value of a non-NaN pattern.
    static double exact(std::uint16_t b)
    {
        const double sign = (b & 0x8000) ? -1.0 : 1.0;
        const int e = (b >> 10) & 0x1F, m = b & 0x3FF;
        if (e == 0x1F) return sign * std::numeric_limits<double>::infinity();
        return sign * (e == 0 ? std::ldexp(static_cast<double>(m), -24) : std::ldexp(static_cast<double>(1024 + m), e - 25));
    }
};

struct Bf16
{
    static constexpr const char *name = "bfloat16";
    static constexpr tbccl::DataType dt = tbccl::DataType::BFloat16;
    using Elem = lowp::BFloat16;
    static constexpr std::uint16_t kInf = 0x7F80, kLastFinite = 0x7F7F;
    static float dec(std::uint16_t b) { return lowp::bf16_bits_to_float(b); }
    static std::uint16_t enc(float f) { return lowp::float_to_bf16_bits_rne(f); }
    static bool is_nan(std::uint16_t b) { return (b & 0x7F80) == 0x7F80 && (b & 0x007F) != 0; }
    static double exact(std::uint16_t b)
    {
        const double sign = (b & 0x8000) ? -1.0 : 1.0;
        const int e = (b >> 7) & 0xFF, m = b & 0x7F;
        if (e == 0xFF) return sign * std::numeric_limits<double>::infinity();
        return sign * (e == 0 ? std::ldexp(static_cast<double>(m), -133) : std::ldexp(static_cast<double>(128 + m), e - 134));
    }
};

inline std::uint32_t next_random(std::uint32_t &state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Equal bits, or both NaN (NaN payloads are not compared; see low_precision.hpp).
template <class F>
inline bool same(std::uint16_t got, std::uint16_t want)
{
    if (F::is_nan(want)) return F::is_nan(got);
    return got == want;
}

template <class F>
inline std::uint16_t expected_sum(std::uint16_t a, std::uint16_t b)
{
    return F::enc(F::dec(a) + F::dec(b));
}


template <class F>
std::vector<std::uint16_t> edge_patterns();

template <>
inline std::vector<std::uint16_t> edge_patterns<Fp16>()
{
    return {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x0401, 0x3555, 0x3C00, 0xBC00, 0x3C01, 0x4000, 0x5640, 0x7BFF, 0xFBFF,
            0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x0200, 0x0800, 0x6400, 0x6401, 0xE400};
}

template <>
inline std::vector<std::uint16_t> edge_patterns<Bf16>()
{
    return {0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x0081, 0x3F80, 0xBF80, 0x3F81, 0x4000, 0x4B00, 0x7F7F, 0xFF7F,
            0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x3F00, 0x4380, 0x4381, 0x3C00, 0xC3C0, 0x3FC0};
}


template <class F>
inline std::vector<std::uint16_t> random_patterns(std::size_t count, std::uint32_t seed)
{
    std::vector<std::uint16_t> v(count);
    std::uint32_t state = seed;
    for (auto &x : v) x = static_cast<std::uint16_t>(next_random(state) >> 8);
    return v;
}


} // namespace lowp_test
