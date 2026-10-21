#pragma once

// Shared by the two-rank all_reduce tests: per-rank input values and their exact sums for every supported datatype, written and compared as raw element bytes.
// Small integers are exact in every supported floating-point type (Float16 and BFloat16 included), so the expected sums are exact: every element is compared bit for bit.

#include <tbccl/reduction.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace redtest
{
using tbccl::DataType;

inline std::uint16_t to_half(unsigned v)
{
    if (v == 0) return 0;
    unsigned e = 0;
    while ((2u << e) <= v) ++e; // v in [2^e, 2^(e+1))
    const unsigned mantissa = static_cast<unsigned>((static_cast<double>(v) / static_cast<double>(1u << e) - 1.0) * 1024.0);
    return static_cast<std::uint16_t>(((e + 15) << 10) | mantissa);
}
inline std::uint16_t to_bf16(unsigned v)
{
    const float f = static_cast<float>(v);
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

inline unsigned value(std::size_t rank, std::size_t i) { return static_cast<unsigned>((i * 7 + rank * 3) % 13 + 1); } // 1..13; the sum of two ranks is at most 26

template <typename T>
void store(std::vector<std::uint8_t> &bytes, std::size_t i, T v)
{
    std::memcpy(bytes.data() + i * sizeof(T), &v, sizeof(T));
}
template <typename T>
T load(const std::vector<std::uint8_t> &bytes, std::size_t i)
{
    T v;
    std::memcpy(&v, bytes.data() + i * sizeof(T), sizeof(T));
    return v;
}

inline std::size_t element_size(DataType t)
{
    switch (t)
    {
    case DataType::Float32: case DataType::Int32: return 4;
    case DataType::Float64: case DataType::Int64: return 8;
    case DataType::Float16: case DataType::BFloat16: return 2;
    default: return 1;
    }
}

// Writes rank `rank`'s element `i` (and nothing else) as the datatype's own bit pattern for the integer value `v`.
inline void put(std::vector<std::uint8_t> &b, DataType t, std::size_t i, unsigned v)
{
    switch (t)
    {
    case DataType::Float32: store<float>(b, i, static_cast<float>(v)); break;
    case DataType::Float64: store<double>(b, i, static_cast<double>(v)); break;
    case DataType::Int32: store<std::int32_t>(b, i, static_cast<std::int32_t>(v)); break;
    case DataType::Int64: store<std::int64_t>(b, i, static_cast<std::int64_t>(v)); break;
    case DataType::Int8: store<std::int8_t>(b, i, static_cast<std::int8_t>(v)); break;
    case DataType::UInt8: store<std::uint8_t>(b, i, static_cast<std::uint8_t>(v)); break;
    case DataType::Float16: store<std::uint16_t>(b, i, to_half(v)); break;
    case DataType::BFloat16: store<std::uint16_t>(b, i, to_bf16(v)); break;
    default: throw std::runtime_error("unsupported datatype in the test");
    }
}
inline bool same_element(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b, DataType t, std::size_t i)
{
    return std::memcmp(a.data() + i * element_size(t), b.data() + i * element_size(t), element_size(t)) == 0;
}

inline const char *name_of(DataType t)
{
    switch (t)
    {
    case DataType::Float32: return "Float32";
    case DataType::Float64: return "Float64";
    case DataType::Int32: return "Int32";
    case DataType::Int64: return "Int64";
    case DataType::Int8: return "Int8";
    case DataType::UInt8: return "UInt8";
    case DataType::Float16: return "Float16";
    case DataType::BFloat16: return "BFloat16";
    default: return "?";
    }
}
} // namespace redtest
