#pragma once

// Phase 49 (internal, not installed): host arithmetic for the 16-bit floating-point reduction types.
//
// Float16 (IEEE 754 binary16) and BFloat16 travel as raw 16-bit words; no compiler _Float16/__bf16 type appears in any
// header (the public API only names DataType::Float16 / DataType::BFloat16). A SUM is defined as
//
//     widen both operands to float32  ->  one float32 add  ->  round once to the target format, ties-to-even
//
// for every element. For N = 2 that is exactly one rounding per element, and it is the same arithmetic the CUDA kernels perform,
// so host and device results agree bit for bit. (For N > 2 a sequential algorithm re-rounds after each pairwise add; N > 2
// low-precision semantics are not defined by Phase 49.)
//
// NaN: any NaN operand produces a NaN result; payloads are not preserved bit for bit across host and CUDA (callers and tests
// assert "is NaN", never a payload). +/-inf and overflow follow IEEE round-to-nearest-even.

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tbccl::detail::lowp
{

inline float bits_to_float(std::uint32_t bits) noexcept
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline std::uint32_t float_to_bits(float value) noexcept
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Exact IEEE binary16 -> float32 widening (every half value is representable in float32).
inline float fp16_bits_to_float(std::uint16_t h) noexcept
{
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            return bits_to_float(sign); // +/-0
        }
        // Subnormal: normalize. The value is mantissa * 2^-24.
        int shifts = 0;
        while ((mantissa & 0x400u) == 0)
        {
            mantissa <<= 1;
            ++shifts;
        }
        mantissa &= 0x3FFu;
        const std::uint32_t float_exponent = static_cast<std::uint32_t>(127 - 15 + 1 - shifts);
        return bits_to_float(sign | (float_exponent << 23) | (mantissa << 13));
    }
    if (exponent == 0x1Fu)
    {
        return bits_to_float(sign | 0x7F800000u | (mantissa << 13)); // inf, or NaN with its payload
    }
    return bits_to_float(sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13));
}

// float32 -> IEEE binary16, round to nearest, ties to even; overflow -> +/-inf; NaN -> a quiet NaN keeping the sign.
inline std::uint16_t float_to_fp16_bits_rne(float value) noexcept
{
    const std::uint32_t bits = float_to_bits(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t magnitude = bits & 0x7FFFFFFFu;

    if (magnitude >= 0x7F800000u)
    {
        if (magnitude == 0x7F800000u)
        {
            return static_cast<std::uint16_t>(sign | 0x7C00u); // inf
        }
        return static_cast<std::uint16_t>(sign | 0x7C00u | 0x0200u | ((magnitude >> 13) & 0x3FFu)); // quiet NaN
    }
    if (magnitude >= 0x477FF000u)
    {
        return static_cast<std::uint16_t>(sign | 0x7C00u); // >= 65520 rounds (tie to even) to infinity
    }
    if (magnitude < 0x38800000u)
    {
        // Result is subnormal or zero (|value| < 2^-14).
        const std::uint32_t exponent = magnitude >> 23;
        if (exponent < 102)
        {
            return static_cast<std::uint16_t>(sign); // below half of the smallest subnormal (2^-25) rounds to zero
        }
        const std::uint32_t mantissa = (magnitude & 0x7FFFFFu) | 0x800000u;
        const std::uint32_t shift = 126 - exponent; // 14..24
        std::uint32_t quotient = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t half = 1u << (shift - 1);
        if (remainder > half || (remainder == half && (quotient & 1u) != 0))
        {
            ++quotient; // may carry into the smallest normal (0x0400), which is the correct encoding
        }
        return static_cast<std::uint16_t>(sign | quotient);
    }
    const std::uint32_t mantissa = magnitude & 0x7FFFFFu;
    std::uint32_t half_bits = (((magnitude >> 23) - 112u) << 10) | (mantissa >> 13);
    const std::uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_bits & 1u) != 0))
    {
        ++half_bits; // a carry out of the mantissa bumps the exponent (up to infinity), as IEEE requires
    }
    return static_cast<std::uint16_t>(sign | half_bits);
}

// bfloat16 is the top 16 bits of a float32: widening is a shift.
inline float bf16_bits_to_float(std::uint16_t b) noexcept
{
    return bits_to_float(static_cast<std::uint32_t>(b) << 16);
}

// float32 -> bfloat16, round to nearest, ties to even (not truncation); NaN stays NaN (quiet bit forced).
inline std::uint16_t float_to_bf16_bits_rne(float value) noexcept
{
    std::uint32_t bits = float_to_bits(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
    {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040u); // NaN: keep sign/payload top bits, force quiet
    }
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u); // ties to even on the retained low bit
    bits += rounding_bias;
    return static_cast<std::uint16_t>(bits >> 16);
}

// Element types the reduction templates (apply_reduction, ring/pipelined/reference algorithms) are instantiated with.
// Trivially copyable 2-byte wrappers: the wire format is the raw 16 bits.
struct Half
{
    std::uint16_t bits;
};

struct BFloat16
{
    std::uint16_t bits;
};

static_assert(sizeof(Half) == 2 && std::is_trivially_copyable_v<Half>, "Half must be a raw 16-bit word");
static_assert(sizeof(BFloat16) == 2 && std::is_trivially_copyable_v<BFloat16>, "BFloat16 must be a raw 16-bit word");

inline Half operator+(Half a, Half b) noexcept
{
    return Half{float_to_fp16_bits_rne(fp16_bits_to_float(a.bits) + fp16_bits_to_float(b.bits))};
}

inline BFloat16 operator+(BFloat16 a, BFloat16 b) noexcept
{
    return BFloat16{float_to_bf16_bits_rne(bf16_bits_to_float(a.bits) + bf16_bits_to_float(b.bits))};
}

// Product/Min/Max are not supported for these types (reduction_supported() rejects them before any communication); these
// exist only so the shared apply_reduction template compiles for every ReduceOp.
inline Half operator*(Half a, Half b) noexcept
{
    return Half{float_to_fp16_bits_rne(fp16_bits_to_float(a.bits) * fp16_bits_to_float(b.bits))};
}
inline BFloat16 operator*(BFloat16 a, BFloat16 b) noexcept
{
    return BFloat16{float_to_bf16_bits_rne(bf16_bits_to_float(a.bits) * bf16_bits_to_float(b.bits))};
}
inline bool operator<(Half a, Half b) noexcept { return fp16_bits_to_float(a.bits) < fp16_bits_to_float(b.bits); }
inline bool operator<(BFloat16 a, BFloat16 b) noexcept { return bf16_bits_to_float(a.bits) < bf16_bits_to_float(b.bits); }

} // namespace tbccl::detail::lowp
