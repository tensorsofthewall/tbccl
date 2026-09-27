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

    // Initial conservative thresholds, gathered from local loopback and
    // real TB4 A/B benchmark data (see Phase 9-11 final reports). Kept
    // centralized here, not scattered through the collective
    // implementations. Every world size absent from a table falls back
    // to Reference under Auto (AutoUnmeasuredWorldSize) — a size is
    // never bucketed into ring policy unless it was actually
    // characterized.

    constexpr WorldSizeThreshold kAllGatherThresholds[] = {
        {3, 16ull * 1024 * 1024}, // N=3: ring only at very large payloads
        {4, 1ull * 1024 * 1024},
        {8, 512ull * 1024},
    };

    constexpr WorldSizeThreshold kReduceScatterThresholds[] = {
        {2, 512ull * 1024},
        {3, 128ull * 1024},
        {4, 512ull * 1024},
        {8, 128ull * 1024},
    };

    constexpr WorldSizeThreshold kAllReduceThresholds[] = {
        {4, 2ull * 1024 * 1024},
        {8, 512ull * 1024},
        // N=2 and N=3 deliberately absent: local N=2 showed no ring
        // crossover, N=3 had no crossover in the tested range, and
        // real TB4 N=2 results were noisy/non-monotonic — that noise
        // is not encoded into generic policy. Forced overrides remain
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
