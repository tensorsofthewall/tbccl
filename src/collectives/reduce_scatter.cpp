#include <tbccl/collectives.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tbccl
{
namespace
{

    // Rank 0 reduces the full N*recv_count input via the existing
    // reduce() (no reduction arithmetic duplicated here), then hands
    // out one rank-ordered segment per peer; every other rank sends its
    // full contribution through that same reduce() call and then
    // receives exactly its own segment. No internal barrier is needed:
    // rank 0 cannot start scattering until reduce() has consumed every
    // contribution, and every non-root rank has already returned from
    // its reduce() send before it reaches the segment recv() below, so
    // collective ordering alone provides the synchronization. `reduced`
    // is typed storage (not a byte buffer reinterpreted later), so it
    // is correctly aligned for T by construction.
    template <typename T>
    void reduce_scatter_typed(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t recv_count,
        DataType datatype,
        ReduceOp op)
    {
        constexpr std::size_t kRoot = 0;

        const std::size_t rank = world.rank();
        const std::size_t size = world.size();
        const std::size_t total_count = size * recv_count;
        const std::size_t segment_bytes = recv_count * sizeof(T);

        if (rank == kRoot)
        {
            std::vector<T> reduced(total_count);

            reduce(
                world, send_buffer, reduced.data(), total_count, datatype,
                op, kRoot);

            std::memcpy(recv_buffer, reduced.data(), segment_bytes);

            for (std::size_t peer = 1; peer < size; ++peer)
            {
                world.send(
                    peer, reduced.data() + peer * recv_count, segment_bytes);
            }
        }
        else
        {
            reduce(
                world, send_buffer, nullptr, total_count, datatype, op,
                kRoot);

            world.recv(kRoot, recv_buffer, segment_bytes);
        }
    }

} // namespace

    void reduce_scatter(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t recv_count,
        DataType datatype,
        ReduceOp op)
    {
        const std::size_t size = world.size();

        validate_reduce_op(op);
        const std::size_t element_size = datatype_size(datatype);

        if (recv_count != 0 &&
            size > std::numeric_limits<std::size_t>::max() / recv_count)
        {
            throw std::overflow_error(
                "reduce_scatter: total element count overflows size_t");
        }

        const std::size_t total_count = size * recv_count;

        if (recv_count > std::numeric_limits<std::size_t>::max() / element_size)
        {
            throw std::overflow_error(
                "reduce_scatter: receive segment size overflows size_t");
        }

        if (total_count >
            std::numeric_limits<std::size_t>::max() / element_size)
        {
            throw std::overflow_error(
                "reduce_scatter: total buffer size overflows size_t");
        }

        if (recv_count > 0 && send_buffer == nullptr)
        {
            throw std::runtime_error(
                "reduce_scatter: send buffer is null for non-zero receive "
                "count");
        }

        if (recv_count > 0 && recv_buffer == nullptr)
        {
            throw std::runtime_error(
                "reduce_scatter: receive buffer is null for non-zero "
                "receive count");
        }

        if (recv_count == 0)
        {
            return;
        }

        switch (datatype)
        {
        case DataType::Int32:
            reduce_scatter_typed<std::int32_t>(
                world, send_buffer, recv_buffer, recv_count, datatype, op);
            break;

        case DataType::Int64:
            reduce_scatter_typed<std::int64_t>(
                world, send_buffer, recv_buffer, recv_count, datatype, op);
            break;

        case DataType::Float32:
            reduce_scatter_typed<float>(
                world, send_buffer, recv_buffer, recv_count, datatype, op);
            break;

        case DataType::Float64:
            reduce_scatter_typed<double>(
                world, send_buffer, recv_buffer, recv_count, datatype, op);
            break;
        }
    }

} // namespace tbccl
