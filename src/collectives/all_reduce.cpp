#include <tbccl/collectives.hpp>

#include <limits>
#include <stdexcept>

namespace tbccl
{

    // AllReduce is deliberately just reduce(root=0) followed by
    // broadcast(root=0) — see collectives.hpp for why send_buffer ==
    // recv_buffer aliasing falls out of that composition for free. No
    // internal barrier is needed between the two calls: rank 0 cannot
    // start broadcasting until reduce() has received every peer's
    // contribution, and every non-root rank has already completed its
    // reduce() send before it reaches broadcast()'s recv, so collective
    // ordering alone provides the synchronization.
    void all_reduce(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        DataType datatype,
        ReduceOp op)
    {
        const std::size_t element_size = datatype_size(datatype);
        validate_reduce_op(op);

        if (count > std::numeric_limits<std::size_t>::max() / element_size)
        {
            throw std::overflow_error(
                "all_reduce: total buffer size overflows size_t");
        }

        if (count > 0 && send_buffer == nullptr)
        {
            throw std::runtime_error(
                "all_reduce: send buffer is null for non-zero element "
                "count");
        }

        if (count > 0 && recv_buffer == nullptr)
        {
            throw std::runtime_error(
                "all_reduce: receive buffer is null for non-zero element "
                "count");
        }

        if (count == 0)
        {
            return;
        }

        constexpr std::size_t kRoot = 0;
        const std::size_t total_bytes = count * element_size;

        reduce(world, send_buffer, recv_buffer, count, datatype, op, kRoot);
        broadcast(world, recv_buffer, total_bytes, kRoot);
    }

} // namespace tbccl
