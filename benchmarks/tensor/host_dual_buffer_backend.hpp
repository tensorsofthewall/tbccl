#pragma once

// Phase 38: a plain-host AsyncMemoryBackend with two SEPARATE buffers
// (source and destination), unlike HostAsyncBackend (host_async_backend.hpp)
// which deliberately aliases both to the same buffer for its Phase 32
// in-place round-trip use case. n2_all_reduce_tensor() needs a backend
// where receiving a peer's contribution (into "destination") never
// overwrites this rank's own local input (in "source") -- see
// docs/phase38_collective_design.md Part 3's root/non-root completion
// asymmetry, which depends on source and destination being genuinely
// distinct storage.

#include <tbccl/async_transfer.hpp>

#include <cstddef>
#include <cstring>
#include <vector>

namespace tbccl_bench::tensor
{

class HostDualBufferBackend final : public tbccl::AsyncMemoryBackend
{
public:
    explicit HostDualBufferBackend(std::size_t capacity)
        : source_(capacity), destination_(capacity)
    {
    }

    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        std::memcpy(staging, source_.data() + chunk.offset, chunk.size);
    }

    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        std::memcpy(destination_.data() + chunk.offset, staging, chunk.size);
    }

    std::size_t capacity() const noexcept { return source_.size(); }

    void *source_data() noexcept { return source_.data(); }
    const void *source_data() const noexcept { return source_.data(); }
    void *destination_data() noexcept { return destination_.data(); }
    const void *destination_data() const noexcept { return destination_.data(); }

private:
    std::vector<std::uint8_t> source_;
    std::vector<std::uint8_t> destination_;
};

} // namespace tbccl_bench::tensor
