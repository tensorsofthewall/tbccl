#include <tbccl/error.hpp>
#include <tbccl/buffer.hpp>

#include <limits>
#include <stdexcept>

namespace tbccl
{

void validate_buffer_view(
    const BufferView &view,
    std::size_t count,
    DataType datatype)
{
    if (count == 0)
    {
        return; // zero-count is always valid, regardless of bytes (Part 31).
    }

    if (view.data == nullptr)
    {
        throw Error(ErrorCode::InvalidArgument, "validate_buffer_view: buffer.data is null for a non-zero count");
    }

    const std::size_t element_size = datatype_size(datatype);
    if (element_size != 0 && count > std::numeric_limits<std::size_t>::max() / element_size)
    {
        throw Error(ErrorCode::InvalidArgument, "validate_buffer_view: count * datatype_size overflows size_t");
    }

    const std::size_t required_bytes = count * element_size;
    if (required_bytes > view.bytes)
    {
        throw Error(ErrorCode::InvalidArgument, 
            "validate_buffer_view: count * datatype_size exceeds buffer.bytes");
    }
}

} // namespace tbccl
