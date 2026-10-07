#pragma once

#include <cstddef>

namespace tbccl
{

// Element types for ARITHMETIC collectives (reduce / all_reduce / reduce_scatter) only. Byte-generic operations
// (send/recv, broadcast, all_gather) move opaque bytes and never need one of these; payloads such as FP8 or packed
// 4-bit weights are deliberately NOT members (they are transported, not reduced).
//
// Numeric values are fixed (an installed consumer may store them): Int32..Float64 are the original Phase 1 set;
// Int8, UInt8, Float16 and BFloat16 were appended in low-precision datatype. Never reorder or reuse a value.
enum class DataType
{
    Int32 = 0,
    Int64 = 1,
    Float32 = 2,
    Float64 = 3,
    Int8 = 4,
    UInt8 = 5,
    Float16 = 6,
    BFloat16 = 7,
};

// Deliberately closed — no user-defined/callback operators yet.
enum class ReduceOp
{
    Sum,
    Product,
    Min,
    Max,
};

// Size in bytes of one element of `datatype`. Throws on an unrecognized
// enum value rather than silently returning 0. The single authoritative
// size table: every collective path calls this, none keeps its own.
std::size_t datatype_size(DataType datatype);

// Throws on an unrecognized ReduceOp value; returns normally otherwise.
void validate_reduce_op(ReduceOp op);

// Stable lowercase names for diagnostics ("float16", "sum"); "unknown" for an out-of-range value.
const char *datatype_label(DataType datatype) noexcept;
const char *reduce_op_label(ReduceOp op) noexcept;

// True if this library can reduce `datatype` with `op`. Int32/Int64/Float32/Float64 support every ReduceOp;
// Int8/UInt8/Float16/BFloat16 support Sum only. This says nothing about memory kinds (see
// Capabilities::supports_collective_all_reduce) and involves no peer: it is a property of the build.
bool reduction_supported(DataType datatype, ReduceOp op) noexcept;

// Throws std::runtime_error "unsupported: reduction dtype=<d> op=<o> (supported ops for this dtype: <list>)" if
// !reduction_supported(). Every public reducing entry point calls this before any communication, so an
// unsupported pair fails immediately and identically on every rank instead of hanging a peer.
void validate_reduction(DataType datatype, ReduceOp op);

} // namespace tbccl
