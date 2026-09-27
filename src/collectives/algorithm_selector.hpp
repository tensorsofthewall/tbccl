#pragma once

// Internal (non-installed) algorithm-selection layer for collectives
// that have both a reference and a ring implementation. Not part of
// the public tbccl:: surface — kept under src/ and exposed to tests/
// benchmark code only via a private include path, not the public
// include/ tree.
//
// Selection is deliberately transport-agnostic: the pure policy
// functions below depend only on parameters guaranteed identical
// across every rank in a World (world size, message/segment/tensor
// size, and the resolved AlgorithmMode) — never host identity,
// interface name, OS, or measured local latency. This guarantees
// every rank reaches the same decision independently, with no
// algorithm-agreement network round trip. See resolve_algorithm_mode()
// for the environment-override contract, which requires identical
// override values on every rank in a World; the current synchronous
// World protocol has no channel to detect or negotiate a mismatch.

#include <cstddef>
#include <string>

namespace tbccl::detail
{

enum class CollectiveAlgorithm
{
    Reference,
    Ring,
};

enum class AlgorithmMode
{
    Auto,
    Reference,
    Ring,
};

enum class CollectiveKind
{
    AllGather,
    ReduceScatter,
    AllReduce,
};

enum class SelectionReason
{
    ForcedReference,
    ForcedRing,
    AutoRingThreshold,
    AutoBelowThreshold,
    AutoRingUnsupported,
    AutoUnmeasuredWorldSize,
};

struct AlgorithmDecision
{
    CollectiveAlgorithm algorithm;
    SelectionReason reason;
};

// "reference" / "ring". Used for benchmark CSV, error messages, and
// tests.
const char *algorithm_name(CollectiveAlgorithm algorithm);

const char *selection_reason_name(SelectionReason reason);

// Parses `value` as "auto"/"reference"/"ring", ASCII case-insensitive.
// `env_var_name` is embedded in the exception message on an
// unrecognized value and is not otherwise interpreted — callers pass
// whichever environment variable name they read `value` from.
AlgorithmMode parse_algorithm_mode(
    const std::string &env_var_name,
    const std::string &value);

// Resolves the algorithm mode for `kind` from environment variables,
// reading the environment fresh at every call (not cached, and never
// mutated by library code):
//
//   per-collective override (TBCCL_ALL_GATHER_ALGORITHM /
//   TBCCL_REDUCE_SCATTER_ALGORITHM / TBCCL_ALL_REDUCE_ALGORITHM)
//       > TBCCL_ALGORITHM
//       > AlgorithmMode::Auto (default, if neither is set)
//
// Throws if a variable that IS set holds an unrecognized value — an
// explicitly-provided invalid override is never silently ignored.
//
// TBCCL_ALGORITHM and the per-collective override variables must have
// identical values on every rank in a World. Different algorithm
// selections across ranks are unsupported and may deadlock, since the
// current synchronous World protocol has no collective-operation
// negotiation channel.
AlgorithmMode resolve_algorithm_mode(CollectiveKind kind);

// Pure selection policy, callable without touching process
// environment variables (see algorithm_selector.cpp for the exact
// threshold table backing "auto" for each collective — kept
// centralized there rather than scattered through the collective
// implementations). Thresholds are an initial, deliberately
// conservative policy, not an API/ABI guarantee; they may change
// between versions. An unmeasured world size always resolves to
// Reference under Auto, never accidentally bucketed into a
// threshold. mode == Reference/Ring always returns that algorithm,
// with no threshold able to override an explicit request.
AlgorithmDecision select_all_gather_algorithm(
    std::size_t world_size,
    std::size_t bytes_per_rank,
    AlgorithmMode mode);

AlgorithmDecision select_reduce_scatter_algorithm(
    std::size_t world_size,
    std::size_t segment_bytes,
    AlgorithmMode mode);

// Ring AllReduce additionally requires `element_count % world_size ==
// 0` (equal segments). Under Auto, a tensor at/above threshold whose
// element_count does not divide world_size falls back to Reference
// (AutoRingUnsupported) rather than selecting an algorithm that would
// throw. Under forced Ring, this function still returns Ring
// regardless of divisibility — forced Ring must never be silently
// downgraded — and the caller's subsequent call into
// detail::all_reduce_ring() is what surfaces the existing clear
// divisibility error before any communication.
AlgorithmDecision select_all_reduce_algorithm(
    std::size_t world_size,
    std::size_t tensor_bytes,
    std::size_t element_count,
    AlgorithmMode mode);

} // namespace tbccl::detail
