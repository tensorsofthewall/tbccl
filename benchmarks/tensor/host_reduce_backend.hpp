#pragma once

// Phase 38: the simplest-correct LocalReduceBackend (tbccl/
// hetero_allreduce.hpp) -- a plain CPU loop over two caller-supplied
// pointers. Used for BOTH the host backend and the Metal-shared backend
// (Part R of the Phase 38 plan): Metal-shared memory is already
// CPU-addressable, so no separate GPU-side Metal reduction kernel is
// needed unless Part S's gate (CPU reduction > 10% of end-to-end time)
// is actually tripped.

#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/reduction.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace tbccl_bench::tensor
{

class HostReduceBackend final : public tbccl::LocalReduceBackend
{
public:
    // `local_and_output` += `peer`, both `count`-element buffers of
    // `datatype` passed to reduce_sum(). Neither pointer is read/written
    // until reduce_sum() is actually called -- construction does no work.
    // `local_and_output` must be mutable host-visible memory; `peer` need
    // only be readable host-visible memory.
    HostReduceBackend(void *local_and_output, const void *peer)
        : local_and_output_(local_and_output), peer_(peer)
    {
    }

    void reduce_sum(std::size_t count, tbccl::DataType datatype) override
    {
        switch (datatype)
        {
        case tbccl::DataType::Float32:
            sum_typed<float>(count);
            return;
        case tbccl::DataType::Float64:
            sum_typed<double>(count);
            return;
        case tbccl::DataType::Int32:
            sum_typed<std::int32_t>(count);
            return;
        case tbccl::DataType::Int64:
            sum_typed<std::int64_t>(count);
            return;
        default:
            break;
        }
        throw std::runtime_error("HostReduceBackend: reduction not implemented for this DataType");
    }

private:
    template <typename T>
    void sum_typed(std::size_t count)
    {
        auto *dst = static_cast<T *>(local_and_output_);
        const auto *src = static_cast<const T *>(peer_);
        for (std::size_t i = 0; i < count; ++i)
        {
            if constexpr (std::is_integral_v<T>)
            {
                // Phase 39 Part AC: bucketed AllReduce sums arbitrary
                // CUDA-compute-kernel output bytes reinterpreted as
                // Int32 words, which overflows routinely -- go through
                // the matching unsigned type so wraparound is
                // well-defined two's-complement modular arithmetic,
                // never signed-overflow UB (same convention as
                // src/collectives/reduction_internal.hpp's
                // wrapping_add, which this isn't allowed to depend on
                // directly since that header is private to the tbccl
                // library target).
                using U = std::make_unsigned_t<T>;
                dst[i] = static_cast<T>(static_cast<U>(dst[i]) + static_cast<U>(src[i]));
            }
            else
            {
                dst[i] = dst[i] + src[i];
            }
        }
    }

    void *local_and_output_;
    const void *peer_;
};

} // namespace tbccl_bench::tensor
