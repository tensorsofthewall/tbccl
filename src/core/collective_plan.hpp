#pragma once

// The internal collective planner. Communicator::all_reduce()/broadcast()/all_gather()/barrier() keep their the N-rank runtime work semantics; BEHIND them rank 0
// maps {collective kind, world size, payload bytes, datatype, op} to a CollectivePlan, and the verdict carries the chosen algorithm id to every rank, so all
// ranks execute exactly the same plan regardless of rank-local environment. Private to libtbccl: nothing here is public API, and it contains no framework,
// model or device knowledge.
//
// world_size 1 is Local and world_size 2 is the N2FastPath (the framework-independent runtime-50 specialised engine, never routed through the generic algorithms). For world_size > 2
// the N-rank runtime root-based algorithms stay available as `Reference` (correctness fallback, oracle, benchmark baseline).
//
// A rank may carry a debug override (TBCCL_BARRIER_ALGORITHM / TBCCL_BROADCAST_ALGORITHM / TBCCL_ALLGATHER_ALGORITHM / TBCCL_ALLREDUCE_ALGORITHM =
// reference | tree | recursive | ring | dissemination). Overrides exist to A/B algorithms and force regression paths, are not a stable interface, and are
// part of the collective descriptor: ranks that carry different overrides are a collective mismatch, never a silent disagreement.

#include "collective_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace tbccl::detail
{

enum class CommAlgorithm : std::uint32_t
{
    Unspecified = 0, // wire value meaning "no override" in a descriptor; never a plan
    Local = 1,
    N2FastPath = 2,
    Reference = 3,
    Ring = 4,
    BinomialTree = 5,
    RecursiveDoubling = 6,
    Dissemination = 7,
};

const char *comm_algorithm_name(CommAlgorithm algorithm) noexcept;

struct CollectivePlan
{
    CommAlgorithm algorithm = CommAlgorithm::Reference;
};

// Selection thresholds (bytes of the collective's payload per rank). Generic local-loopback heuristics measured earlier, NOT
// Thunderbolt-tuned constants; a future topology/autotuning layer may replace them.
struct PlannerThresholds
{
    std::size_t all_reduce_ring_min_bytes = 0;   // N>2: bytes >= this -> Ring, below -> BinomialTree (0 until measured: see planner)
    std::size_t broadcast_tree_min_bytes = 0;    // N>2: bytes >= this -> BinomialTree, below -> Reference
};

struct PlannerOverrides
{
    CommAlgorithm barrier = CommAlgorithm::Unspecified;
    CommAlgorithm broadcast = CommAlgorithm::Unspecified;
    CommAlgorithm all_gather = CommAlgorithm::Unspecified;
    CommAlgorithm all_reduce = CommAlgorithm::Unspecified;

    CommAlgorithm for_kind(CollectiveKind kind) const noexcept;
};

// Reads the four TBCCL_*_ALGORITHM variables. An unrecognized value throws std::runtime_error("invalid_argument: ...") so a typo cannot silently select the default.
PlannerOverrides planner_overrides_from_environment();

// True if `algorithm` can run `kind` at `world_size` (e.g. recursive doubling needs a power-of-two world; the tree/ring/dissemination/reference algorithms need
// world_size > 2 here because 1 and 2 have their own paths). `why_not` receives a short reason when false.
bool algorithm_supported(CollectiveKind kind, CommAlgorithm algorithm, std::size_t world_size, std::string *why_not = nullptr);

// The plan for one collective. `forced` is the agreed override (Unspecified = choose). Throws std::runtime_error("unsupported: ...") if `forced` cannot run the
// collective at this world size. Pure function of its arguments: identical inputs give identical plans.
CollectivePlan plan_collective(
    CollectiveKind kind, std::size_t world_size, std::size_t bytes, CommAlgorithm forced, const PlannerThresholds &thresholds);

} // namespace tbccl::detail
