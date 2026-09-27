#include <tbccl/collectives.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tbccl
{
namespace
{

    template <typename T>
    struct UnsignedOf;

    template <>
    struct UnsignedOf<std::int32_t>
    {
        using type = std::uint32_t;
    };

    template <>
    struct UnsignedOf<std::int64_t>
    {
        using type = std::uint64_t;
    };

    // Signed Sum/Product go through the corresponding unsigned type so
    // wraparound is two's-complement modular arithmetic, never signed-
    // overflow undefined behavior.
    template <typename T>
    T wrapping_add(T a, T b)
    {
        using U = typename UnsignedOf<T>::type;
        return static_cast<T>(
            static_cast<U>(a) + static_cast<U>(b));
    }

    template <typename T>
    T wrapping_mul(T a, T b)
    {
        using U = typename UnsignedOf<T>::type;
        return static_cast<T>(
            static_cast<U>(a) * static_cast<U>(b));
    }

    template <typename T>
    void apply_reduction(
        T *dst,
        const T *src,
        std::size_t count,
        ReduceOp op)
    {
        switch (op)
        {
        case ReduceOp::Sum:
            for (std::size_t i = 0; i < count; ++i)
            {
                if constexpr (std::is_integral_v<T>)
                {
                    dst[i] = wrapping_add(dst[i], src[i]);
                }
                else
                {
                    dst[i] = dst[i] + src[i];
                }
            }
            break;

        case ReduceOp::Product:
            for (std::size_t i = 0; i < count; ++i)
            {
                if constexpr (std::is_integral_v<T>)
                {
                    dst[i] = wrapping_mul(dst[i], src[i]);
                }
                else
                {
                    dst[i] = dst[i] * src[i];
                }
            }
            break;

        case ReduceOp::Min:
            for (std::size_t i = 0; i < count; ++i)
            {
                dst[i] = std::min(dst[i], src[i]);
            }
            break;

        case ReduceOp::Max:
            for (std::size_t i = 0; i < count; ++i)
            {
                dst[i] = std::max(dst[i], src[i]);
            }
            break;
        }
    }

    // Non-root ranks send their whole contribution to root in one shot.
    // Root copies its own contribution into recv_buffer (memcpy skipped
    // if send_buffer == recv_buffer, so root aliasing is harmless), then
    // receives and reduces each other rank's contribution in ascending
    // rank order into one reusable typed temporary buffer — this is what
    // makes results reproducible for a given input and root regardless
    // of which rank is root. When world.size() == 1, root has no peers
    // to receive from, so this degenerates to exactly the local copy.
    template <typename T>
    void reduce_typed(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        ReduceOp op,
        std::size_t root)
    {
        const std::size_t rank = world.rank();
        const std::size_t size = world.size();
        const std::size_t total_bytes = count * sizeof(T);

        if (rank != root)
        {
            world.send(root, send_buffer, total_bytes);
            return;
        }

        const auto *typed_send = static_cast<const T *>(send_buffer);
        auto *typed_recv = static_cast<T *>(recv_buffer);

        if (typed_send != typed_recv)
        {
            std::memcpy(recv_buffer, send_buffer, total_bytes);
        }

        std::vector<T> temp(count);

        for (std::size_t peer = 0; peer < size; ++peer)
        {
            if (peer == root)
            {
                continue;
            }

            world.recv(peer, temp.data(), total_bytes);
            apply_reduction(typed_recv, temp.data(), count, op);
        }
    }

} // namespace

    void reduce(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        DataType datatype,
        ReduceOp op,
        std::size_t root)
    {
        const std::size_t size = world.size();

        if (root >= size)
        {
            throw std::runtime_error(
                "reduce: root rank " + std::to_string(root) +
                " is invalid for world size " + std::to_string(size));
        }

        const std::size_t element_size = datatype_size(datatype);

        if (count > std::numeric_limits<std::size_t>::max() / element_size)
        {
            throw std::overflow_error(
                "reduce: total buffer size overflows size_t");
        }

        if (count > 0 && send_buffer == nullptr)
        {
            throw std::runtime_error(
                "reduce: send buffer is null for non-zero element count");
        }

        if (count > 0 && world.rank() == root && recv_buffer == nullptr)
        {
            throw std::runtime_error(
                "reduce: root receive buffer is null for non-zero element "
                "count");
        }

        if (count == 0)
        {
            return;
        }

        switch (datatype)
        {
        case DataType::Int32:
            reduce_typed<std::int32_t>(
                world, send_buffer, recv_buffer, count, op, root);
            break;

        case DataType::Int64:
            reduce_typed<std::int64_t>(
                world, send_buffer, recv_buffer, count, op, root);
            break;

        case DataType::Float32:
            reduce_typed<float>(
                world, send_buffer, recv_buffer, count, op, root);
            break;

        case DataType::Float64:
            reduce_typed<double>(
                world, send_buffer, recv_buffer, count, op, root);
            break;
        }
    }

} // namespace tbccl
