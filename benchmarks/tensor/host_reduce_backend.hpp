#pragma once

// The simplest-correct LocalReduceBackend (tbccl/ hetero_allreduce.hpp)
// -- a plain CPU loop over two caller-supplied pointers. Used for BOTH
// the host backend and the Metal-shared backend: Metal-shared memory is
// already CPU-addressable, so no separate GPU-side Metal reduction
// kernel is needed unless the gate (CPU reduction > 10% of end-to-end
// time) is actually tripped.

#include <tbccl/reduction.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

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
        }
        throw std::runtime_error("HostReduceBackend: unrecognized DataType");
    }

private:
    template <typename T>
    void sum_typed(std::size_t count)
    {
        auto *dst = static_cast<T *>(local_and_output_);
        const auto *src = static_cast<const T *>(peer_);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = dst[i] + src[i];
        }
    }

    void *local_and_output_;
    const void *peer_;
};

} // namespace tbccl_bench::tensor
