#include <tbccl/async_transfer.hpp>

#include <algorithm>

namespace tbccl
{

std::vector<Chunk> plan_chunks(
    std::size_t total_bytes,
    std::size_t chunk_hint,
    std::size_t alignment)
{
    std::vector<Chunk> chunks;

    if (total_bytes == 0)
    {
        return chunks;
    }

    if (alignment == 0)
    {
        alignment = 1;
    }

    if (chunk_hint == 0 || chunk_hint >= total_bytes)
    {
        chunks.push_back(Chunk{0, total_bytes});
        return chunks;
    }

    // Round chunk_hint down to a multiple of alignment where that
    // still leaves at least one full alignment unit -- never rounds a
    // small chunk_hint down to zero, and never skips or duplicates a
    // byte: any leftover from rounding simply becomes part of the
    // following chunk (or the final partial chunk), not a gap.
    std::size_t aligned_hint = (chunk_hint / alignment) * alignment;
    if (aligned_hint == 0)
    {
        aligned_hint = chunk_hint;
    }

    std::size_t offset = 0;
    while (offset < total_bytes)
    {
        const std::size_t remaining = total_bytes - offset;
        const std::size_t size = std::min(aligned_hint, remaining);
        chunks.push_back(Chunk{offset, size});
        offset += size;
    }

    return chunks;
}

} // namespace tbccl
