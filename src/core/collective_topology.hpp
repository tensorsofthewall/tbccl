#pragma once

// Phase 51: the internal collective topology: pure functions of (rank, root, world_size). There is NO public topology API and no machine/model knowledge: with no
// outside information the rank order is [0, 1, ..., N-1], the ring is r-1 -> r -> r+1 (mod N) and trees are binomial trees over logical ranks relative to the root.
// A future phase may build these from measured link properties; the algorithms only consume the functions below.

#include <cstddef>
#include <vector>

namespace tbccl::detail
{

// Ring neighbours in the default order.
inline std::size_t ring_next(std::size_t rank, std::size_t world) { return (rank + 1) % world; }
inline std::size_t ring_prev(std::size_t rank, std::size_t world) { return (rank + world - 1) % world; }

// Binomial tree rooted at `root`. Logical rank lr = (rank - root) mod N. The parent of lr (lr != 0) is lr with its lowest set bit cleared; the children of lr are
// lr + 2^j for every 2^j below lr's lowest set bit (all j for the root) with lr + 2^j < N. Depth is ceil(log2 N); a node with k children sends to them in
// DESCENDING j (the largest subtree first), which is the standard binomial schedule.
inline std::size_t tree_logical(std::size_t rank, std::size_t root, std::size_t world) { return (rank + world - root) % world; }
inline std::size_t tree_physical(std::size_t logical, std::size_t root, std::size_t world) { return (logical + root) % world; }

inline std::size_t tree_parent(std::size_t rank, std::size_t root, std::size_t world)
{
    const std::size_t lr = tree_logical(rank, root, world);
    return tree_physical(lr & (lr - 1), root, world); // only meaningful for lr != 0
}

inline std::vector<std::size_t> tree_children(std::size_t rank, std::size_t root, std::size_t world)
{
    const std::size_t lr = tree_logical(rank, root, world);
    const std::size_t limit = lr == 0 ? world : (lr & (~lr + 1)); // lowest set bit of lr (the root has no limit)
    std::size_t top = 1;
    while ((top << 1) < limit && (top << 1) < world) top <<= 1;
    std::vector<std::size_t> out;
    for (std::size_t bit = top; bit >= 1; bit >>= 1)
    {
        if (bit < limit && lr + bit < world) out.push_back(tree_physical(lr + bit, root, world));
        if (bit == 1) break;
    }
    return out;
}

// Element-count chunking for ring reduce-scatter / all-gather: `count` elements into `world` contiguous chunks whose sizes differ by at most one (the first
// count % world chunks are one element longer). Chunks may be empty when count < world.
inline std::size_t chunk_elements(std::size_t count, std::size_t world, std::size_t chunk) { return count / world + (chunk < count % world ? 1 : 0); }
inline std::size_t chunk_offset_elements(std::size_t count, std::size_t world, std::size_t chunk)
{
    const std::size_t base = count / world, rem = count % world;
    return chunk * base + (chunk < rem ? chunk : rem);
}

} // namespace tbccl::detail
