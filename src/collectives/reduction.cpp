#include <tbccl/reduction.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tbccl
{

    std::size_t datatype_size(DataType datatype)
    {
        switch (datatype)
        {
        case DataType::Int8:
        case DataType::UInt8:
            return 1;

        case DataType::Float16:
        case DataType::BFloat16:
            return 2;

        case DataType::Int32:
            return sizeof(std::int32_t);

        case DataType::Int64:
            return sizeof(std::int64_t);

        case DataType::Float32:
            return sizeof(float);

        case DataType::Float64:
            return sizeof(double);
        }

        throw std::runtime_error("datatype_size: unknown DataType");
    }

    void validate_reduce_op(ReduceOp op)
    {
        switch (op)
        {
        case ReduceOp::Sum:
        case ReduceOp::Product:
        case ReduceOp::Min:
        case ReduceOp::Max:
            return;
        }

        throw std::runtime_error("validate_reduce_op: unknown ReduceOp");
    }

    const char *datatype_label(DataType datatype) noexcept
    {
        switch (datatype)
        {
        case DataType::Int8: return "int8";
        case DataType::UInt8: return "uint8";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::Float16: return "float16";
        case DataType::BFloat16: return "bfloat16";
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        }
        return "unknown";
    }

    const char *reduce_op_label(ReduceOp op) noexcept
    {
        switch (op)
        {
        case ReduceOp::Sum: return "sum";
        case ReduceOp::Product: return "product";
        case ReduceOp::Min: return "min";
        case ReduceOp::Max: return "max";
        }
        return "unknown";
    }

    bool reduction_supported(DataType datatype, ReduceOp op) noexcept
    {
        switch (datatype)
        {
        case DataType::Int32:
        case DataType::Int64:
        case DataType::Float32:
        case DataType::Float64:
            return op == ReduceOp::Sum || op == ReduceOp::Product || op == ReduceOp::Min || op == ReduceOp::Max;

        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Float16:
        case DataType::BFloat16:
            return false; // enabled together with their arithmetic (Phase 49 follow-up commits)
        }
        return false;
    }

    void validate_reduction(DataType datatype, ReduceOp op)
    {
        if (reduction_supported(datatype, op))
        {
            return;
        }

        std::string supported;
        for (ReduceOp candidate : {ReduceOp::Sum, ReduceOp::Product, ReduceOp::Min, ReduceOp::Max})
        {
            if (reduction_supported(datatype, candidate))
            {
                supported += supported.empty() ? "" : ", ";
                supported += reduce_op_label(candidate);
            }
        }
        if (supported.empty()) supported = "none";
        throw std::runtime_error(
            std::string("unsupported: reduction dtype=") + datatype_label(datatype) + " op=" + reduce_op_label(op) +
            " (supported ops for this dtype: " + supported + ")");
    }

} // namespace tbccl
