#include <tbccl/cuda_support.hpp>

#include "cuda_external_async_backend.hpp"

#include <tbccl/communicator.hpp>

#include <memory>

namespace tbccl_bench::tensor
{

class CudaMemoryProvider final : public tbccl::ExternalMemoryProvider
{
public:
    CudaMemoryProvider(
        const tbccl::BufferView &buffer,
        const tbccl::ExecutionContext &context,
        std::shared_ptr<CudaStagingResources> resources)
        : primary_(buffer.data, buffer.bytes, nullptr, resources), bytes_(buffer.bytes), resources_(std::move(resources))
    {
        // Part L: source readiness -- if the caller supplied a CUDA
        // stream, make our copy stream wait on it before any D2H starts
        // reading the external buffer, instead of requiring the caller
        // to cudaDeviceSynchronize() first.
        if (context.kind == tbccl::ExecutionContextKind::CudaStream)
        {
            primary_.wait_for_producer_stream(context.native_handle);
        }
    }

    tbccl::AsyncMemoryBackend &primary_backend() override { return primary_; }

    tbccl::AsyncMemoryBackend &scratch_backend() override
    {
        if (!scratch_)
        {
            // nullptr external pointer -> this instance allocates and
            // owns `bytes_` of device memory itself (Communicator-owned
            // scratch, not the caller's tensor -- Part 25).
            scratch_ = std::make_unique<CudaExternalAsyncBackend>(nullptr, bytes_, nullptr, resources_);
        }
        return *scratch_;
    }

    tbccl::LocalReduceBackend &reduce_backend() override
    {
        scratch_backend();
        if (!reduce_)
        {
            reduce_ = std::make_unique<CudaExternalReduceBackend>(
                primary_.device_ptr(), scratch_->device_ptr(), nullptr);
        }
        return *reduce_;
    }

private:
    CudaExternalAsyncBackend primary_;
    std::size_t bytes_;
    std::shared_ptr<CudaStagingResources> resources_;
    std::unique_ptr<CudaExternalAsyncBackend> scratch_;
    std::unique_ptr<CudaExternalReduceBackend> reduce_;
};

} // namespace tbccl_bench::tensor

namespace tbccl
{

void register_cuda_support()
{
    register_memory_provider_factory_ex(
        MemoryKind::Cuda,
        [](const BufferView &buffer, const ExecutionContext &context, ProviderResourceSlot &slot)
            -> std::unique_ptr<ExternalMemoryProvider> {
            // One persistent staging owner per Communicator, created on first CUDA use.
            if (!slot) slot = std::make_shared<tbccl_bench::tensor::CudaStagingResources>();
            return std::make_unique<tbccl_bench::tensor::CudaMemoryProvider>(
                buffer, context, std::static_pointer_cast<tbccl_bench::tensor::CudaStagingResources>(slot));
        });
}

bool cuda_staging_stats(const Communicator &comm, CudaStagingStats &out)
{
    auto res = std::static_pointer_cast<tbccl_bench::tensor::CudaStagingResources>(comm.provider_resources(MemoryKind::Cuda));
    if (!res) return false;
    const auto s = res->stats();
    out = CudaStagingStats{s.pinned_alloc_count, s.pinned_free_count, s.pinned_allocated_bytes_total, s.pinned_capacity,
                           s.pinned_peak_capacity, s.device_scratch_alloc_count, s.device_scratch_free_count, s.device_scratch_capacity};
    return true;
}

void cuda_staging_test_fail_next_pinned_growth(const Communicator &comm)
{
    auto res = std::static_pointer_cast<tbccl_bench::tensor::CudaStagingResources>(comm.provider_resources(MemoryKind::Cuda));
    if (res) res->fail_next_pinned_growth();
}

} // namespace tbccl
