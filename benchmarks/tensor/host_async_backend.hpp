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

    std::size_t capacity() const noexcept { return capacity_; }

private:
    std::byte *buffer_;
    std::size_t capacity_;
};

} // namespace tbccl_bench::tensor
