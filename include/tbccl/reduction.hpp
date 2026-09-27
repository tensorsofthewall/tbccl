#pragma once

#include <cstddef>

namespace tbccl
{

// Deliberately small initial set — no Float16/BFloat16, no unsigned or
// complex types, until real GPU buffer integration motivates them.
enum class DataType
{
    Int32,
    Int64,
    Float32,
    Float64,
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
// enum value rather than silently returning 0.
std::size_t datatype_size(DataType datatype);

// Throws on an unrecognized ReduceOp value; returns normally otherwise.
void validate_reduce_op(ReduceOp op);

} // namespace tbccl
