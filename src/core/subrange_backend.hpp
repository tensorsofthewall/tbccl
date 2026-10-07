#pragma once

// A view of [offset, offset + size) of another AsyncMemoryBackend, so a ring step can move one chunk of a provider's buffer through the unchanged transfer engine
// (staged or direct). Private to libtbccl.

#include <tbccl/async_transfer.hpp>

#include <cstddef>

namespace tbccl::detail
{

class SubRangeBackend final : public AsyncMemoryBackend
{
public:
    SubRangeBackend(AsyncMemoryBackend &base, std::size_t offset, std::size_t size) : base_(base), offset_(offset), size_(size) {}

    void stage_source_chunk(const Chunk &chunk, void *staging) override { base_.stage_source_chunk(Chunk{offset_ + chunk.offset, chunk.size}, staging); }
    void commit_destination_chunk(const Chunk &chunk, const void *staging) override { base_.commit_destination_chunk(Chunk{offset_ + chunk.offset, chunk.size}, staging); }

    bool supports_direct_transport_access() const noexcept override { return base_.supports_direct_transport_access(); }
    const void *direct_source_data() const noexcept override { return static_cast<const char *>(base_.direct_source_data()) + offset_; }
    void *direct_destination_data() noexcept override { return static_cast<char *>(base_.direct_destination_data()) + offset_; }

    std::size_t size() const noexcept { return size_; }

private:
    AsyncMemoryBackend &base_;
    std::size_t offset_;
    std::size_t size_;
};

} // namespace tbccl::detail
