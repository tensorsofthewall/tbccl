#pragma once

// A plain-host AsyncMemoryBackend (tbccl/async_transfer.hpp) over a
// non-owning raw buffer. Used directly for host<->host benchmarking,
// and as the staging-side implementation other backends (CUDA/Metal)
// can compose with for their own host-visible staging buffer once
// GPU-side work has already completed (Commit 4).

#include <tbccl/async_transfer.hpp>

#include <cstddef>
#include <cstring>

namespace tbccl_bench::tensor
{

class HostAsyncBackend final : public tbccl::AsyncMemoryBackend
{
public:
    // `buffer` must outlive this backend and remain valid (and, for a
    // source buffer, already fully populated) for as long as any
    // TransferRequest referencing this backend is outstanding.
    HostAsyncBackend(void *buffer, std::size_t capacity)
        : buffer_(static_cast<std::byte *>(buffer)), capacity_(capacity)
    {
    }

    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        std::memcpy(staging, buffer_ + chunk.offset, chunk.size);
    }

    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        std::memcpy(buffer_ + chunk.offset, staging, chunk.size);
    }

    // Plain host CPU memory is already directly readable/writable by
    // TcpTransport -- no staging copy is needed at all.
    // stage_source_chunk()/commit_destination_chunk() above remain
    // correct and are still used whenever a caller explicitly
    // requests chunking/pipelining via chunk_hint != 0 (e.g. to
    // overlap transfer with compute) -- TensorCommWorker only takes
    // the direct path when chunk_hint == 0 (see async_transfer.hpp).
    bool supports_direct_transport_access() const noexcept override { return true; }
    const void *direct_source_data() const noexcept override { return buffer_; }
    void *direct_destination_data() noexcept override { return buffer_; }

    std::size_t capacity() const noexcept { return capacity_; }

private:
    std::byte *buffer_;
    std::size_t capacity_;
};

} // namespace tbccl_bench::tensor
