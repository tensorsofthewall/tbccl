#include <tbccl/reduction.hpp>

#include <cstdint>
#include <stdexcept>

namespace tbccl
{

    std::size_t datatype_size(DataType datatype)
    {
        switch (datatype)
        {
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

} // namespace tbccl
