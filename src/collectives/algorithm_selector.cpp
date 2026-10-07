#include "algorithm_selector.hpp"

#include <cstdlib>
#include <stdexcept>

namespace tbccl::detail
{
namespace
{

    std::string to_lower_ascii(const std::string &text)
    {
        std::string result = text;

        for (char &c : result)
        {
            if (c >= 'A' && c <= 'Z')
            {
                c = static_cast<char>(c - 'A' + 'a');
            }
        }

        return result;
    }

    const char *env_var_name_for(CollectiveKind kind)
    {
        switch (kind)
        {
        case CollectiveKind::AllGather:
            return "TBCCL_ALL_GATHER_ALGORITHM";
        case CollectiveKind::ReduceScatter:
            return "TBCCL_REDUCE_SCATTER_ALGORITHM";
        case CollectiveKind::AllReduce:
            return "TBCCL_ALL_REDUCE_ALGORITHM";
        }

        return "";
    }

    struct WorldSizeThreshold
    {
        std::size_t world_size;
        std::size_t threshold_bytes;
    };

    // Algorithm selection thresholds. The persistent ring worker substantially cut ring's fixed per-invocation overhead,
    // which made the thresholds of the earlier per-invocation-thread ring badly stale: most
    // crossover points had moved down by one to two orders of
    // magnitude. These values come from a two-stage process, not a
    // single sweep: a broad loopback A/B sweep (2 independent
    // runs) identified rough crossover *neighborhoods*, then each
    // neighborhood was re-measured directly (3 independent runs,
    // checking both median AND p95, at the candidate size and its
    // half/double neighbors) before a value was adopted. Several
    // candidates did not survive that re-check
    // and were moved to a
    // larger, cleanly-supported size instead of being adopted as-is;
    // AllReduce's candidates in particular turned out to be
    // computed against the wrong size unit (segment bytes instead of
    // the tensor bytes the selector and CLI actually use) and, once
    // corrected and re-measured, did not support lowering any of its
    // thresholds at all.
    //
    // Kept centralized here, not scattered through the collective
    // implementations. Every world size absent from a table falls back
    // to Reference under Auto (AutoUnmeasuredWorldSize) — a size is
    // never bucketed into ring policy unless it was actually
    // characterized, and never interpolated between neighboring world
    // sizes.

    constexpr WorldSizeThreshold kAllGatherThresholds[] = {
        // N=2 is a v2 policy change, not a threshold adjustment: the
        // earlier implementation always used Reference here
        // (loopback and real-TB4 data conflicted at the time). The
        // dedicated N=2 investigation — alternating (ABBA) real
        // TB4 A/B runs at 64 B/4 KiB/64 KiB/1 MiB, both physical rank
        // assignments, and both busy_poll settings — found Ring
        // consistently faster (10-40% median improvement) with
        // *better*, not worse, p95 at every tested size, unlike some
        // of the small-message p95 regressions seen for other
        // collectives. 64 (bytes) is the smallest size actually
        // tested; not extrapolated below it.
        {2, 64},
        {3, 64ull * 1024},
        // N=4/N=8: the loopback sweep's
        // raw crossover point (16 KiB) did not hold up under the
        // 3-run re-check — Ring was still consistently worse at
        // 16 KiB, only becoming cleanly and consistently favorable at
        // 32 KiB — so the threshold was moved up to the larger,
        // actually-supported size rather than adopted as originally
        // proposed.
        {4, 32ull * 1024},
        {8, 32ull * 1024},
    };

    constexpr WorldSizeThreshold kReduceScatterThresholds[] = {
        // N=2's loopback candidate (64 KiB)
        // was contradicted by a non-monotonic re-check:
        // Ring lost consistently at 64 KiB across 3 reps despite
        // winning at both the smaller (32 KiB) and larger (128 KiB)
        // neighboring sizes tested. Moved up to 128 KiB, the
        // smallest size where the win was clean and consistent
        // across every rep.
        {2, 128ull * 1024},
        {3, 64ull * 1024},
        // N=4/N=8: same pattern as AllGather — the loopback
        // algorithm-sweep raw crossover (16 KiB) did not survive
        // re-measurement (Ring consistently worse at 16 KiB,
        // sometimes still worse even at 32 KiB for N=4); moved to the
        // smallest size with a clean, repeatable win in all 3 reps.
        {4, 64ull * 1024},
        {8, 32ull * 1024},
    };

    constexpr WorldSizeThreshold kAllReduceThresholds[] = {
        {4, 2ull * 1024 * 1024},
        {8, 512ull * 1024},
        // N=2 and N=3 deliberately absent, and N=4/N=8 deliberately
        // UNCHANGED from the earlier ring implementation — this is a real (evidence-based)
        // finding, not an oversight. The loopback sweep's AllReduce candidates
        // were computed against segment bytes but the selector
        // threshold (and the benchmark's --sizes) is total tensor
        // bytes per rank; once that was corrected and
        // re-measured directly in tensor-byte terms: N=3 showed Ring
        // consistently *worse* than Reference at every tensor size
        // tested, up to 768 KiB; N=4 showed no consistent direction
        // (median improvement flipped sign between repeated runs at
        // both the candidate size and 2x it); N=8's candidate (256
        // KiB tensor) was marginal/inconsistent, and the size where
        // Ring became a clean, repeatable win (512 KiB tensor) turned
        // out to already match the earlier threshold almost
        // exactly. AllReduce composes two ring phases (reduce-scatter
        // + all-gather) per call, roughly doubling the fixed cost
        // Ring must amortize versus AllGather/ReduceScatter alone,
        // which is a plausible physical explanation for why its
        // crossovers sit so much higher. Forced overrides remain
        // available for platform-specific tuning.
    };

    template <std::size_t N>
    const WorldSizeThreshold *find_threshold(
        const WorldSizeThreshold (&table)[N],
        std::size_t world_size)
    {
        for (const auto &entry : table)
        {
            if (entry.world_size == world_size)
            {
                return &entry;
            }
        }

        return nullptr;
    }

} // namespace

    const char *algorithm_name(CollectiveAlgorithm algorithm)
    {
        switch (algorithm)
        {
        case CollectiveAlgorithm::Reference:
            return "reference";
        case CollectiveAlgorithm::Ring:
            return "ring";
        }

        return "unknown";
    }

    const char *selection_reason_name(SelectionReason reason)
    {
        switch (reason)
        {
        case SelectionReason::ForcedReference:
            return "forced_reference";
        case SelectionReason::ForcedRing:
            return "forced_ring";
        case SelectionReason::AutoRingThreshold:
            return "auto_ring_threshold";
        case SelectionReason::AutoBelowThreshold:
            return "auto_below_threshold";
        case SelectionReason::AutoRingUnsupported:
            return "auto_ring_unsupported";
        case SelectionReason::AutoUnmeasuredWorldSize:
            return "auto_unmeasured_world_size";
        }

        return "unknown";
    }

    AlgorithmMode parse_algorithm_mode(
        const std::string &env_var_name,
        const std::string &value)
    {
        const std::string lowered = to_lower_ascii(value);

        if (lowered == "auto")
        {
            return AlgorithmMode::Auto;
        }

        if (lowered == "reference")
        {
            return AlgorithmMode::Reference;
        }

        if (lowered == "ring")
        {
            return AlgorithmMode::Ring;
        }

        throw std::runtime_error(
            "invalid " + env_var_name + " value \"" + value +
            "\"; expected auto, reference, or ring");
    }

    AlgorithmMode resolve_algorithm_mode(CollectiveKind kind)
    {
        const char *per_collective_name = env_var_name_for(kind);

        if (const char *value = std::getenv(per_collective_name))
        {
            return parse_algorithm_mode(per_collective_name, value);
        }

        if (const char *value = std::getenv("TBCCL_ALGORITHM"))
        {
            return parse_algorithm_mode("TBCCL_ALGORITHM", value);
        }

        return AlgorithmMode::Auto;
    }

    AlgorithmDecision select_all_gather_algorithm(
        std::size_t world_size,
        std::size_t bytes_per_rank,
        AlgorithmMode mode)
    {
        if (mode == AlgorithmMode::Reference)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::ForcedReference};
        }

        if (mode == AlgorithmMode::Ring)
        {
            return {CollectiveAlgorithm::Ring, SelectionReason::ForcedRing};
        }

        const WorldSizeThreshold *entry =
            find_threshold(kAllGatherThresholds, world_size);

        if (entry == nullptr)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::AutoUnmeasuredWorldSize};
        }

        if (bytes_per_rank >= entry->threshold_bytes)
        {
            return {CollectiveAlgorithm::Ring,
                    SelectionReason::AutoRingThreshold};
        }

        return {CollectiveAlgorithm::Reference,
                SelectionReason::AutoBelowThreshold};
    }

    AlgorithmDecision select_reduce_scatter_algorithm(
        std::size_t world_size,
        std::size_t segment_bytes,
        AlgorithmMode mode)
    {
        if (mode == AlgorithmMode::Reference)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::ForcedReference};
        }

        if (mode == AlgorithmMode::Ring)
        {
            return {CollectiveAlgorithm::Ring, SelectionReason::ForcedRing};
        }

        const WorldSizeThreshold *entry =
            find_threshold(kReduceScatterThresholds, world_size);

        if (entry == nullptr)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::AutoUnmeasuredWorldSize};
        }

        if (segment_bytes >= entry->threshold_bytes)
        {
            return {CollectiveAlgorithm::Ring,
                    SelectionReason::AutoRingThreshold};
        }

        return {CollectiveAlgorithm::Reference,
                SelectionReason::AutoBelowThreshold};
    }

    AlgorithmDecision select_all_reduce_algorithm(
        std::size_t world_size,
        std::size_t tensor_bytes,
        std::size_t element_count,
        AlgorithmMode mode)
    {
        if (mode == AlgorithmMode::Reference)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::ForcedReference};
        }

        if (mode == AlgorithmMode::Ring)
        {
            return {CollectiveAlgorithm::Ring, SelectionReason::ForcedRing};
        }

        const WorldSizeThreshold *entry =
            find_threshold(kAllReduceThresholds, world_size);

        if (entry == nullptr)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::AutoUnmeasuredWorldSize};
        }

        if (tensor_bytes < entry->threshold_bytes)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::AutoBelowThreshold};
        }

        if (world_size == 0 || element_count % world_size != 0)
        {
            return {CollectiveAlgorithm::Reference,
                    SelectionReason::AutoRingUnsupported};
        }

        return {CollectiveAlgorithm::Ring,
                SelectionReason::AutoRingThreshold};
    }

} // namespace tbccl::detail
