// The internal CollectivePlanner and the algorithm id carried in the verdict. Pure: no sockets.

#include "collective_plan.hpp"

#include "test_utils.hpp"

#include <cstdlib>
#include <iostream>

using tbccl_test::expect;
using namespace tbccl;
using namespace tbccl::detail;

namespace
{
    bool throws_with(const std::function<void()> &f, const std::string &needle)
    {
        try
        {
            f();
        }
        catch (const std::runtime_error &e)
        {
            return std::string(e.what()).find(needle) != std::string::npos;
        }
        return false;
    }
} // namespace

int main()
{
    const PlannerThresholds th;
    for (auto kind : {CollectiveKind::Barrier, CollectiveKind::Broadcast, CollectiveKind::AllGather, CollectiveKind::AllReduce})
    {
        expect(plan_collective(kind, 1, 4096, CommAlgorithm::Unspecified, th).algorithm == CommAlgorithm::Local, "world 1 is local");
        expect(plan_collective(kind, 2, 4096, CommAlgorithm::Unspecified, th).algorithm == CommAlgorithm::N2FastPath, "world 2 is the fast path");
        expect(plan_collective(kind, 2, 4096, CommAlgorithm::Ring, th).algorithm == CommAlgorithm::N2FastPath, "an override never reroutes the N=2 fast path");
    }

    // support matrix
    std::string why;
    expect(algorithm_supported(CollectiveKind::Barrier, CommAlgorithm::Dissemination, 3, &why), "dissemination barrier");
    expect(!algorithm_supported(CollectiveKind::AllReduce, CommAlgorithm::Dissemination, 3, &why) && !why.empty(), "dissemination is a barrier only");
    expect(algorithm_supported(CollectiveKind::Broadcast, CommAlgorithm::BinomialTree, 5), "tree broadcast at 5");
    expect(!algorithm_supported(CollectiveKind::AllGather, CommAlgorithm::BinomialTree, 5), "no tree all_gather");
    expect(algorithm_supported(CollectiveKind::AllGather, CommAlgorithm::Ring, 5), "ring all_gather");
    for (std::size_t n : {std::size_t{4}, std::size_t{8}, std::size_t{16}}) expect(algorithm_supported(CollectiveKind::AllReduce, CommAlgorithm::RecursiveDoubling, n), "recursive doubling at a power of two");
    for (std::size_t n : {std::size_t{3}, std::size_t{5}, std::size_t{6}, std::size_t{7}}) expect(!algorithm_supported(CollectiveKind::AllReduce, CommAlgorithm::RecursiveDoubling, n, &why) && why.find("power-of-two") != std::string::npos, "recursive doubling refuses non-powers of two");

    // forcing
    expect(plan_collective(CollectiveKind::AllReduce, 4, 64, CommAlgorithm::Ring, th).algorithm == CommAlgorithm::Ring, "forced ring");
    expect(throws_with([&] { plan_collective(CollectiveKind::AllReduce, 5, 64, CommAlgorithm::RecursiveDoubling, th); }, "unsupported: forced algorithm 'recursive-doubling'"), "forcing recursive doubling at N=5 is an unsupported error");
    expect(throws_with([&] { plan_collective(CollectiveKind::Barrier, 4, 0, CommAlgorithm::Ring, th); }, "cannot run barrier"), "forcing ring for a barrier is an unsupported error");

    // default selection (the measured heuristics)
    auto def = [&](CollectiveKind k, std::size_t n, std::size_t bytes) { return plan_collective(k, n, bytes, CommAlgorithm::Unspecified, th).algorithm; };
    for (std::size_t n = 3; n <= 8; ++n)
    {
        expect(def(CollectiveKind::Broadcast, n, 64) == CommAlgorithm::BinomialTree && def(CollectiveKind::Broadcast, n, 16 << 20) == CommAlgorithm::BinomialTree, "broadcast: tree");
        expect(def(CollectiveKind::AllGather, n, 64) == CommAlgorithm::Ring && def(CollectiveKind::AllGather, n, 16 << 20) == CommAlgorithm::Ring, "all_gather: ring");
        expect(def(CollectiveKind::Barrier, n, 0) == CommAlgorithm::Reference, "barrier below the dissemination threshold: the control-plane gather/release");
        expect(def(CollectiveKind::AllReduce, n, 16 << 20) == CommAlgorithm::Ring, "large all_reduce: ring");
    }
    expect(def(CollectiveKind::Barrier, 9, 0) == CommAlgorithm::Dissemination && def(CollectiveKind::Barrier, 16, 0) == CommAlgorithm::Dissemination, "barrier: dissemination from the threshold on");
    expect(def(CollectiveKind::AllReduce, 3, 4096) == CommAlgorithm::BinomialTree && def(CollectiveKind::AllReduce, 3, 64 << 10) == CommAlgorithm::BinomialTree && def(CollectiveKind::AllReduce, 3, 96 << 10) == CommAlgorithm::Ring, "N=3 all_reduce: tree below 96 KiB, ring from there");
    expect(def(CollectiveKind::AllReduce, 4, 64) == CommAlgorithm::RecursiveDoubling && def(CollectiveKind::AllReduce, 4, 128 << 10) == CommAlgorithm::RecursiveDoubling && def(CollectiveKind::AllReduce, 4, 192 << 10) == CommAlgorithm::Ring, "N=4 all_reduce: recursive doubling below 192 KiB, ring from there");
    expect(def(CollectiveKind::AllReduce, 8, 4096) == CommAlgorithm::BinomialTree && def(CollectiveKind::AllReduce, 8, 512 << 10) == CommAlgorithm::BinomialTree && def(CollectiveKind::AllReduce, 8, 576 << 10) == CommAlgorithm::Ring, "N=8 all_reduce: tree (recursive doubling lost at 8), ring from 576 KiB");
    for (std::size_t n : {std::size_t{3}, std::size_t{5}, std::size_t{6}, std::size_t{7}})
        expect(def(CollectiveKind::AllReduce, n, 64) == CommAlgorithm::BinomialTree, "recursive doubling is never chosen for a non-power-of-two world");
    for (std::size_t n = 3; n <= 16; ++n)
        for (std::size_t bytes : {std::size_t{0}, std::size_t{1} << 10, std::size_t{1} << 16, std::size_t{1} << 20, std::size_t{1} << 24})
            for (auto kind : {CollectiveKind::Broadcast, CollectiveKind::AllGather, CollectiveKind::AllReduce, CollectiveKind::Barrier})
                expect(algorithm_supported(kind, def(kind, n, bytes), n), "the default plan is always a supported algorithm");

    // determinism: identical inputs -> identical plans
    for (std::size_t n = 3; n <= 8; ++n)
        for (std::size_t bytes : {std::size_t{0}, std::size_t{64}, std::size_t{4096}, std::size_t{1} << 20, std::size_t{16} << 20})
            expect(plan_collective(CollectiveKind::AllReduce, n, bytes, CommAlgorithm::Unspecified, th).algorithm == plan_collective(CollectiveKind::AllReduce, n, bytes, CommAlgorithm::Unspecified, th).algorithm, "deterministic");

    // environment parsing
    setenv("TBCCL_ALLREDUCE_ALGORITHM", "ring", 1);
    setenv("TBCCL_BARRIER_ALGORITHM", "dissemination", 1);
    auto o = planner_overrides_from_environment();
    expect(o.all_reduce == CommAlgorithm::Ring && o.barrier == CommAlgorithm::Dissemination && o.broadcast == CommAlgorithm::Unspecified && o.for_kind(CollectiveKind::AllReduce) == CommAlgorithm::Ring, "overrides read from the environment");
    setenv("TBCCL_ALLREDUCE_ALGORITHM", "rnig", 1);
    expect(throws_with([] { planner_overrides_from_environment(); }, "invalid_argument: TBCCL_ALLREDUCE_ALGORITHM='rnig'"), "a typo is rejected, never silently defaulted");
    unsetenv("TBCCL_ALLREDUCE_ALGORITHM");
    unsetenv("TBCCL_BARRIER_ALGORITHM");

    // wire: the verdict carries the algorithm, the descriptor carries the override
    {
        CollectiveVerdict v;
        v.status = VerdictStatus::Ok;
        v.algorithm = static_cast<std::uint32_t>(CommAlgorithm::Ring);
        v.sequence = 7;
        const auto w = decode_verdict(encode_verdict(v));
        expect(w.algorithm == static_cast<std::uint32_t>(CommAlgorithm::Ring) && w.sequence == 7, "verdict algorithm round trip");
        CollectiveDescriptor d;
        d.forced_algorithm = static_cast<std::uint32_t>(CommAlgorithm::BinomialTree);
        d.note = std::string(80, 'n'); // the note must not run into the forced-algorithm field
        expect(decode_descriptor(encode_descriptor(d)).forced_algorithm == static_cast<std::uint32_t>(CommAlgorithm::BinomialTree), "descriptor forced algorithm round trip");
    }

    // two ranks forcing DIFFERENT algorithms is a collective mismatch (never a silent disagreement); one rank forcing and the rest not is fine
    {
        std::vector<CollectiveDescriptor> by_rank(3);
        for (std::size_t r = 0; r < 3; ++r)
        {
            by_rank[r].sequence = 4;
            by_rank[r].kind = CollectiveKind::AllReduce;
            by_rank[r].rank = static_cast<std::uint32_t>(r);
            by_rank[r].bytes = 64;
            by_rank[r].count = 16;
        }
        by_rank[0].forced_algorithm = static_cast<std::uint32_t>(CommAlgorithm::Ring);
        by_rank[2].forced_algorithm = static_cast<std::uint32_t>(CommAlgorithm::BinomialTree);
        auto v = judge_collective(by_rank);
        expect(v.status == VerdictStatus::Mismatch && v.text.find("force different algorithms") != std::string::npos, "different forced algorithms: " + v.text);
        by_rank[2].forced_algorithm = 0;
        expect(judge_collective(by_rank).status == VerdictStatus::Ok, "an override on only some ranks is not a mismatch");
        by_rank[2].forced_algorithm = by_rank[0].forced_algorithm;
        expect(judge_collective(by_rank).status == VerdictStatus::Ok, "equal overrides agree");
    }

    std::cout << "collective_plan_test passed\n";
    return 0;
}
