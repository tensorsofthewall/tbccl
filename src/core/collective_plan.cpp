// Phase 51: CollectivePlanner (collective_plan.hpp).

#include "collective_plan.hpp"

#include <cstdlib>
#include <stdexcept>

namespace tbccl::detail
{

const char *comm_algorithm_name(CommAlgorithm a) noexcept
{
    switch (a)
    {
    case CommAlgorithm::Unspecified: return "unspecified";
    case CommAlgorithm::Local: return "local";
    case CommAlgorithm::N2FastPath: return "n2-fast-path";
    case CommAlgorithm::Reference: return "reference";
    case CommAlgorithm::Ring: return "ring";
    case CommAlgorithm::BinomialTree: return "tree";
    case CommAlgorithm::RecursiveDoubling: return "recursive-doubling";
    case CommAlgorithm::Dissemination: return "dissemination";
    }
    return "unknown";
}

CommAlgorithm PlannerOverrides::for_kind(CollectiveKind kind) const noexcept
{
    switch (kind)
    {
    case CollectiveKind::Barrier: return barrier;
    case CollectiveKind::Broadcast: return broadcast;
    case CollectiveKind::AllGather: return all_gather;
    case CollectiveKind::AllReduce: return all_reduce;
    }
    return CommAlgorithm::Unspecified;
}

namespace
{

CommAlgorithm parse_override(const char *variable)
{
    const char *value = std::getenv(variable);
    if (value == nullptr || *value == '\0') return CommAlgorithm::Unspecified;
    const std::string v(value);
    if (v == "reference") return CommAlgorithm::Reference;
    if (v == "tree") return CommAlgorithm::BinomialTree;
    if (v == "recursive") return CommAlgorithm::RecursiveDoubling;
    if (v == "ring") return CommAlgorithm::Ring;
    if (v == "dissemination") return CommAlgorithm::Dissemination;
    if (v == "auto") return CommAlgorithm::Unspecified;
    throw std::runtime_error(
        std::string("invalid_argument: ") + variable + "='" + v + "' is not one of reference, tree, recursive, ring, dissemination, auto");
}

bool is_power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

} // namespace

PlannerOverrides planner_overrides_from_environment()
{
    PlannerOverrides o;
    o.barrier = parse_override("TBCCL_BARRIER_ALGORITHM");
    o.broadcast = parse_override("TBCCL_BROADCAST_ALGORITHM");
    o.all_gather = parse_override("TBCCL_ALLGATHER_ALGORITHM");
    o.all_reduce = parse_override("TBCCL_ALLREDUCE_ALGORITHM");
    return o;
}

bool algorithm_supported(CollectiveKind kind, CommAlgorithm algorithm, std::size_t world_size, std::string *why_not)
{
    auto no = [&](const char *why) {
        if (why_not) *why_not = why;
        return false;
    };
    if (algorithm == CommAlgorithm::Unspecified) return true;
    if (world_size <= 2) return no("only world_size > 2 has selectable algorithms (1 is local, 2 is the specialised fast path)");
    switch (algorithm)
    {
    case CommAlgorithm::Reference:
        return true;
    case CommAlgorithm::Dissemination:
        return kind == CollectiveKind::Barrier ? true : no("dissemination only implements barrier");
    case CommAlgorithm::BinomialTree:
        return (kind == CollectiveKind::Broadcast || kind == CollectiveKind::AllReduce) ? true : no("tree implements broadcast and all_reduce");
    case CommAlgorithm::Ring:
        return (kind == CollectiveKind::AllGather || kind == CollectiveKind::AllReduce) ? true : no("ring implements all_gather and all_reduce");
    case CommAlgorithm::RecursiveDoubling:
        if (kind != CollectiveKind::AllReduce) return no("recursive doubling implements all_reduce only");
        return is_power_of_two(world_size) ? true : no("recursive doubling needs a power-of-two world size");
    default:
        return no("not a selectable algorithm");
    }
}

CollectivePlan plan_collective(
    CollectiveKind kind, std::size_t world_size, std::size_t bytes, CommAlgorithm forced, const PlannerThresholds &thresholds)
{
    CollectivePlan plan;
    if (world_size <= 1)
    {
        plan.algorithm = CommAlgorithm::Local;
        return plan;
    }
    if (world_size == 2)
    {
        plan.algorithm = CommAlgorithm::N2FastPath;
        return plan;
    }
    if (forced != CommAlgorithm::Unspecified)
    {
        std::string why;
        if (!algorithm_supported(kind, forced, world_size, &why))
        {
            throw std::runtime_error(
                std::string("unsupported: forced algorithm '") + comm_algorithm_name(forced) + "' cannot run " + collective_kind_name(kind) + " at world_size " +
                std::to_string(world_size) + ": " + why);
        }
        plan.algorithm = forced;
        return plan;
    }
    // Defaults, from the Phase 51 measurements (see PlannerThresholds). bytes is the per-rank payload of the collective.
    switch (kind)
    {
    case CollectiveKind::Barrier:
        plan.algorithm = world_size >= thresholds.barrier_dissemination_min_world ? CommAlgorithm::Dissemination : CommAlgorithm::Reference;
        break;
    case CollectiveKind::Broadcast:
        plan.algorithm = CommAlgorithm::BinomialTree; // beat the root fan-out at every measured size, N=3,4,8
        break;
    case CollectiveKind::AllGather:
        plan.algorithm = CommAlgorithm::Ring; // beat gather+fan-out at every measured size, N=3,4,8
        break;
    case CollectiveKind::AllReduce:
        if (bytes >= thresholds.all_reduce_ring_bytes_per_extra_rank * (world_size - 2)) plan.algorithm = CommAlgorithm::Ring;
        else if (is_power_of_two(world_size) && world_size <= thresholds.recursive_doubling_max_world) plan.algorithm = CommAlgorithm::RecursiveDoubling;
        else plan.algorithm = CommAlgorithm::BinomialTree;
        break;
    }
    return plan;
}

} // namespace tbccl::detail
