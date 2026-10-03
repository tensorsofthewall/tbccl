// Phase 51: ring / binomial-tree / chunk-partition helper functions. Pure.

#include "collective_topology.hpp"

#include "test_utils.hpp"

#include <algorithm>
#include <iostream>
#include <set>

using tbccl_test::expect;
using namespace tbccl::detail;

int main()
{
    // Trees: for every N and root, each non-root rank has exactly one parent, the parent lists it as a child, the tree spans all ranks, and the depth is ceil(log2 N).
    for (std::size_t n = 2; n <= 64; ++n)
    {
        std::size_t log2n = 0;
        while ((std::size_t{1} << log2n) < n) ++log2n;
        for (std::size_t root = 0; root < n; ++root)
        {
            std::size_t child_edges = 0;
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t c : tree_children(r, root, n))
                {
                    expect(c != root && c < n, "child is a valid non-root rank");
                    expect(tree_parent(c, root, n) == r, "parent(child(r)) == r for N=" + std::to_string(n));
                    ++child_edges;
                }
            }
            expect(child_edges == n - 1, "a spanning tree has N-1 edges (N=" + std::to_string(n) + ")");
            // depth and reachability by walking up to the root
            for (std::size_t r = 0; r < n; ++r)
            {
                std::size_t depth = 0, cur = r;
                while (cur != root)
                {
                    cur = tree_parent(cur, root, n);
                    expect(++depth <= log2n, "depth <= ceil(log2 N) for N=" + std::to_string(n) + " root=" + std::to_string(root));
                }
            }
            expect(tree_children(root, root, n).size() == log2n, "the root has ceil(log2 N) children");
        }
    }
    // a worked example: N=8 root 0 is the classic binomial tree
    {
        auto c0 = tree_children(0, 0, 8);
        expect((c0 == std::vector<std::size_t>{4, 2, 1}), "root children descend");
        expect((tree_children(4, 0, 8) == std::vector<std::size_t>{6, 5}) && (tree_children(2, 0, 8) == std::vector<std::size_t>{3}) && tree_children(1, 0, 8).empty(), "N=8 tree shape");
        expect(tree_parent(7, 0, 8) == 6 && tree_parent(6, 0, 8) == 4 && tree_parent(4, 0, 8) == 0, "N=8 parents");
        // rotated root
        expect(tree_children(3, 3, 5).size() == 3 && tree_parent(4, 3, 5) == 3, "rotation by the root");
    }
    // Ring
    for (std::size_t n = 2; n <= 16; ++n)
        for (std::size_t r = 0; r < n; ++r) expect(ring_prev(ring_next(r, n), n) == r && ring_next(r, n) < n, "ring neighbours");

    // Chunking
    for (std::size_t n = 1; n <= 9; ++n)
    {
        for (std::size_t count : {std::size_t{0}, std::size_t{1}, n - 1, n, n + 1, std::size_t{17}, std::size_t{1000}, std::size_t{1000003}})
        {
            std::size_t total = 0, max_size = 0, min_size = ~std::size_t{0};
            for (std::size_t c = 0; c < n; ++c)
            {
                expect(chunk_offset_elements(count, n, c) == total, "chunks are contiguous");
                const std::size_t sz = chunk_elements(count, n, c);
                total += sz;
                max_size = std::max(max_size, sz);
                min_size = std::min(min_size, sz);
            }
            expect(total == count, "chunks cover the buffer exactly (N=" + std::to_string(n) + ", count=" + std::to_string(count) + ")");
            expect(max_size - min_size <= 1, "chunk sizes differ by at most one");
        }
    }
    std::cout << "collective_topology_test passed\n";
    return 0;
}
