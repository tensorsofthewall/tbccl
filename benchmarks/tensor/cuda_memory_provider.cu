#include "cuda_memory_provider.hpp"

#include "cuda_external_async_backend.hpp"

#include <tbccl/communicator.hpp>

#include <memory>

namespace tbccl_bench::tensor
{

namespace
{

class CudaMemoryProvider final : public tbccl::ExternalMemoryProvider
{
public:
    CudaMemoryProvider(const tbccl::BufferView &buffer, const tbccl::ExecutionContext &context)
        : primary_(buffer.data, buffer.bytes), bytes_(buffer.bytes)
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
            scratch_ = std::make_unique<CudaExternalAsyncBackend>(nullptr, bytes_);
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
    std::unique_ptr<CudaExternalAsyncBackend> scratch_;
    std::unique_ptr<CudaExternalReduceBackend> reduce_;
};

} // namespace

void register_cuda_memory_provider()
{
    tbccl::register_memory_provider_factory(
        tbccl::MemoryKind::Cuda,
        [](const tbccl::BufferView &buffer, const tbccl::ExecutionContext &context) -> std::unique_ptr<tbccl::ExternalMemoryProvider> {
            return std::make_unique<CudaMemoryProvider>(buffer, context);
        });
}

} // namespace tbccl_bench::tensor
