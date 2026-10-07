#pragma once

// The built-in host-pointer AsyncMemoryBackend, shared by the Host/MetalShared provider and the N-rank collective control messages
// (descriptors, verdicts). Direct transport access: the transport reads/writes the buffer in place, no staging.

#include <tbccl/async_transfer.hpp>

#include <cstddef>
#include <cstring>

namespace tbccl::detail
{

class HostPointerAsyncBackend final : public AsyncMemoryBackend
{
public:
    HostPointerAsyncBackend(void *buffer, std::size_t capacity)
        : buffer_(static_cast<std::byte *>(buffer)), capacity_(capacity)
    {
    }

    void stage_source_chunk(const Chunk &chunk, void *staging) override
    {
        std::memcpy(staging, buffer_ + chunk.offset, chunk.size);
    }

    void commit_destination_chunk(const Chunk &chunk, const void *staging) override
    {
        std::memcpy(buffer_ + chunk.offset, staging, chunk.size);
    }

    bool supports_direct_transport_access() const noexcept override { return true; }
    const void *direct_source_data() const noexcept override { return buffer_; }
    void *direct_destination_data() noexcept override { return buffer_; }

    void *data() const noexcept { return buffer_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    std::byte *buffer_;
    std::size_t capacity_;
};

} // namespace tbccl::detail
