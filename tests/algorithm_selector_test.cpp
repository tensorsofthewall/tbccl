#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "algorithm_selector.hpp"
#include "test_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using tbccl_test::deterministic_buffer;
using tbccl_test::expect;
using tbccl_test::join_and_check;
using tbccl_test::make_local_peers;
using tbccl_test::make_options;
using tbccl_test::run_rank;

namespace
{

    // Fixed port ranges, one block per distributed test, continuing on
    // from every other test file's range (up through
    // all_reduce_ring_test's 32100-32560) and — importantly — kept
    // below 32768, the default Linux ephemeral port range floor
    // (`/proc/sys/net/ipv4/ip_local_port_range`). Every other test
    // file in this repo already stays below that floor; ports at or
    // above it can transiently collide with any outbound connection's
    // OS-assigned source port anywhere on the machine, which causes
    // intermittent "Address already in use" bind failures unrelated to
    // this test's own logic.
    constexpr std::uint16_t kAllGatherAutoRingBase = 32600;
    constexpr std::uint16_t kAllGatherAutoRingN2Base = 32605;
    constexpr std::uint16_t kReduceScatterAutoRingBase = 32610;
    constexpr std::uint16_t kAllReduceAutoRingBase = 32620;
    constexpr std::uint16_t kAllReduceAutoIneligibleBase = 32630;
    constexpr std::uint16_t kForcedRefAllGatherBase = 32640;
    constexpr std::uint16_t kForcedRefReduceScatterBase = 32650;
    constexpr std::uint16_t kForcedRefAllReduceBase = 32660;
    constexpr std::uint16_t kForcedRingAllGatherBase = 32670;
    constexpr std::uint16_t kForcedRingReduceScatterBase = 32680;
    constexpr std::uint16_t kForcedRingAllReduceBase = 32690;
    constexpr std::uint16_t kForcedRingNondivisibleBase = 32700;

    // -------------------------------------------------------------------
    // Environment RAII helpers. Every distributed test in this file
    // runs every rank as a std::thread inside this single process, so
    // setting an env var once before spawning the rank threads exactly
    // models "every rank in the World has the identical override" —
    // the distributed-override contract these variables require.
    // -------------------------------------------------------------------

    void set_env(const std::string &name, const std::string &value)
    {
#if defined(_WIN32)
        _putenv_s(name.c_str(), value.c_str());
#else
        setenv(name.c_str(), value.c_str(), 1);
#endif
    }

    void unset_env(const std::string &name)
    {
#if defined(_WIN32)
        _putenv_s(name.c_str(), "");
#else
        unsetenv(name.c_str());
#endif
    }

    // Sets `name` to `value` for this object's lifetime, restoring
    // whatever value (or absence) `name` held beforehand on
    // destruction. Never leaks configuration into later tests.
    class EnvOverride
    {
    public:
        EnvOverride(std::string name, const std::string &value)
            : name_(std::move(name))
        {
            const char *existing = std::getenv(name_.c_str());
            had_previous_ = existing != nullptr;

            if (had_previous_)
            {
                previous_ = existing;
            }

            set_env(name_, value);
        }

        ~EnvOverride()
        {
            if (had_previous_)
            {
                set_env(name_, previous_);
            }
            else
            {
                unset_env(name_);
            }
        }

        EnvOverride(const EnvOverride &) = delete;
        EnvOverride &operator=(const EnvOverride &) = delete;

    private:
        std::string name_;
        std::string previous_;
        bool had_previous_ = false;
    };

    // Ensures `name` is unset for this object's lifetime (so tests are
    // deterministic even if the ambient shell happens to export one of
    // these variables), restoring its previous value on destruction.
    class EnvUnset
    {
    public:
        explicit EnvUnset(std::string name) : name_(std::move(name))
        {
            const char *existing = std::getenv(name_.c_str());
            had_previous_ = existing != nullptr;

            if (had_previous_)
            {
                previous_ = existing;
            }

            unset_env(name_);
        }

        ~EnvUnset()
        {
            if (had_previous_)
            {
                set_env(name_, previous_);
            }
        }

        EnvUnset(const EnvUnset &) = delete;
        EnvUnset &operator=(const EnvUnset &) = delete;

    private:
        std::string name_;
        std::string previous_;
        bool had_previous_ = false;
    };

    void expect_decision(
        tbccl::detail::AlgorithmDecision decision,
        tbccl::detail::CollectiveAlgorithm expected_algorithm,
        tbccl::detail::SelectionReason expected_reason,
        const std::string &context)
    {
        expect(
            decision.algorithm == expected_algorithm,
            context + ": expected algorithm " +
                tbccl::detail::algorithm_name(expected_algorithm) +
                ", got " + tbccl::detail::algorithm_name(decision.algorithm));

        expect(
            decision.reason == expected_reason,
            context + ": expected reason " +
                tbccl::detail::selection_reason_name(expected_reason) +
                ", got " +
                tbccl::detail::selection_reason_name(decision.reason));
    }

    // -------------------------------------------------------------------
    // Pure selector tests — no networking.
    // -------------------------------------------------------------------

    void test_algorithm_and_reason_names()
    {
        using tbccl::detail::algorithm_name;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::selection_reason_name;
        using tbccl::detail::SelectionReason;

        expect(
            std::string(algorithm_name(CollectiveAlgorithm::Reference)) ==
                "reference",
            "algorithm_name(Reference)");
        expect(
            std::string(algorithm_name(CollectiveAlgorithm::Ring)) == "ring",
            "algorithm_name(Ring)");

        expect(
            std::string(
                selection_reason_name(SelectionReason::ForcedReference)) ==
                "forced_reference",
            "selection_reason_name(ForcedReference)");
        expect(
            std::string(selection_reason_name(SelectionReason::ForcedRing)) ==
                "forced_ring",
            "selection_reason_name(ForcedRing)");
        expect(
            std::string(selection_reason_name(
                SelectionReason::AutoRingThreshold)) ==
                "auto_ring_threshold",
            "selection_reason_name(AutoRingThreshold)");
        expect(
            std::string(selection_reason_name(
                SelectionReason::AutoBelowThreshold)) ==
                "auto_below_threshold",
            "selection_reason_name(AutoBelowThreshold)");
        expect(
            std::string(selection_reason_name(
                SelectionReason::AutoRingUnsupported)) ==
                "auto_ring_unsupported",
            "selection_reason_name(AutoRingUnsupported)");
        expect(
            std::string(selection_reason_name(
                SelectionReason::AutoUnmeasuredWorldSize)) ==
                "auto_unmeasured_world_size",
            "selection_reason_name(AutoUnmeasuredWorldSize)");

        std::cout << "[PASS] test_algorithm_and_reason_names\n";
    }

    void test_mode_reference_always_reference()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        for (std::size_t world_size : {1, 2, 3, 4, 5, 8, 16})
        {
            for (std::size_t bytes :
                 {std::size_t{0}, std::size_t{1}, std::size_t{1048576},
                  std::size_t{16777216}, std::size_t{67108864}})
            {
                expect_decision(
                    tbccl::detail::select_all_gather_algorithm(
                        world_size, bytes, AlgorithmMode::Reference),
                    CollectiveAlgorithm::Reference,
                    SelectionReason::ForcedReference,
                    "all_gather Reference mode world_size=" +
                        std::to_string(world_size) +
                        " bytes=" + std::to_string(bytes));

                expect_decision(
                    tbccl::detail::select_reduce_scatter_algorithm(
                        world_size, bytes, AlgorithmMode::Reference),
                    CollectiveAlgorithm::Reference,
                    SelectionReason::ForcedReference,
                    "reduce_scatter Reference mode world_size=" +
                        std::to_string(world_size) +
                        " bytes=" + std::to_string(bytes));

                expect_decision(
                    tbccl::detail::select_all_reduce_algorithm(
                        world_size, bytes, world_size,
                        AlgorithmMode::Reference),
                    CollectiveAlgorithm::Reference,
                    SelectionReason::ForcedReference,
                    "all_reduce Reference mode world_size=" +
                        std::to_string(world_size) +
                        " bytes=" + std::to_string(bytes));
            }
        }

        std::cout << "[PASS] test_mode_reference_always_reference\n";
    }

    void test_mode_ring_always_ring()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        for (std::size_t world_size : {1, 2, 3, 4, 5, 8, 16})
        {
            for (std::size_t bytes :
                 {std::size_t{0}, std::size_t{1}, std::size_t{1048576}})
            {
                expect_decision(
                    tbccl::detail::select_all_gather_algorithm(
                        world_size, bytes, AlgorithmMode::Ring),
                    CollectiveAlgorithm::Ring, SelectionReason::ForcedRing,
                    "all_gather Ring mode world_size=" +
                        std::to_string(world_size));

                expect_decision(
                    tbccl::detail::select_reduce_scatter_algorithm(
                        world_size, bytes, AlgorithmMode::Ring),
                    CollectiveAlgorithm::Ring, SelectionReason::ForcedRing,
                    "reduce_scatter Ring mode world_size=" +
                        std::to_string(world_size));
            }
        }

        // AllReduce forced Ring: the selector always returns Ring
        // regardless of divisibility — forced Ring must never be
        // silently downgraded. The eventual call into
        // detail::all_reduce_ring() is what surfaces the divisibility
        // error; see test_forced_ring_all_reduce_nondivisible_rejects().
        expect_decision(
            tbccl::detail::select_all_reduce_algorithm(
                4, 1048576, 257, AlgorithmMode::Ring),
            CollectiveAlgorithm::Ring, SelectionReason::ForcedRing,
            "all_reduce Ring mode, non-divisible count");

        expect_decision(
            tbccl::detail::select_all_reduce_algorithm(
                4, 1048576, 256, AlgorithmMode::Ring),
            CollectiveAlgorithm::Ring, SelectionReason::ForcedRing,
            "all_reduce Ring mode, divisible count");

        std::cout << "[PASS] test_mode_ring_always_ring\n";
    }

    void test_all_gather_threshold_boundaries()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        struct Case
        {
            std::size_t world_size;
            std::size_t threshold;
        };

        for (Case c :
             {Case{2, 64}, Case{3, 64ull * 1024}, Case{4, 32ull * 1024},
              Case{8, 32ull * 1024}})
        {
            expect_decision(
                tbccl::detail::select_all_gather_algorithm(
                    c.world_size, c.threshold - 1, AlgorithmMode::Auto),
                CollectiveAlgorithm::Reference,
                SelectionReason::AutoBelowThreshold,
                "all_gather below threshold N=" +
                    std::to_string(c.world_size));

            expect_decision(
                tbccl::detail::select_all_gather_algorithm(
                    c.world_size, c.threshold, AlgorithmMode::Auto),
                CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
                "all_gather at threshold N=" + std::to_string(c.world_size));

            expect_decision(
                tbccl::detail::select_all_gather_algorithm(
                    c.world_size, c.threshold + 1, AlgorithmMode::Auto),
                CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
                "all_gather above threshold N=" +
                    std::to_string(c.world_size));
        }

        std::cout << "[PASS] test_all_gather_threshold_boundaries\n";
    }

    void test_reduce_scatter_threshold_boundaries()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        struct Case
        {
            std::size_t world_size;
            std::size_t threshold;
        };

        for (Case c :
             {Case{2, 128ull * 1024}, Case{3, 64ull * 1024},
              Case{4, 64ull * 1024}, Case{8, 32ull * 1024}})
        {
            expect_decision(
                tbccl::detail::select_reduce_scatter_algorithm(
                    c.world_size, c.threshold - 1, AlgorithmMode::Auto),
                CollectiveAlgorithm::Reference,
                SelectionReason::AutoBelowThreshold,
                "reduce_scatter below threshold N=" +
                    std::to_string(c.world_size));

            expect_decision(
                tbccl::detail::select_reduce_scatter_algorithm(
                    c.world_size, c.threshold, AlgorithmMode::Auto),
                CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
                "reduce_scatter at threshold N=" +
                    std::to_string(c.world_size));

            expect_decision(
                tbccl::detail::select_reduce_scatter_algorithm(
                    c.world_size, c.threshold + 1, AlgorithmMode::Auto),
                CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
                "reduce_scatter above threshold N=" +
                    std::to_string(c.world_size));
        }

        std::cout << "[PASS] test_reduce_scatter_threshold_boundaries\n";
    }

    void test_all_reduce_threshold_boundaries()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        struct Case
        {
            std::size_t world_size;
            std::size_t threshold;
            std::size_t divisible_count;
        };

        for (Case c :
             {Case{4, 2ull * 1024 * 1024, 400}, Case{8, 512ull * 1024, 800}})
        {
            expect_decision(
                tbccl::detail::select_all_reduce_algorithm(
                    c.world_size, c.threshold - 1, c.divisible_count,
                    AlgorithmMode::Auto),
                CollectiveAlgorithm::Reference,
                SelectionReason::AutoBelowThreshold,
                "all_reduce below threshold N=" +
                    std::to_string(c.world_size));

            expect_decision(
                tbccl::detail::select_all_reduce_algorithm(
                    c.world_size, c.threshold, c.divisible_count,
                    AlgorithmMode::Auto),
                CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
                "all_reduce at threshold N=" + std::to_string(c.world_size));

            expect_decision(
                tbccl::detail::select_all_reduce_algorithm(
                    c.world_size, c.threshold + 1, c.divisible_count,
                    AlgorithmMode::Auto),
                CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
                "all_reduce above threshold N=" +
                    std::to_string(c.world_size));
        }

        std::cout << "[PASS] test_all_reduce_threshold_boundaries\n";
    }

    void test_unmeasured_world_sizes()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        const std::vector<std::size_t> sample_bytes = {
            0, 1024, 1ull << 30};

        // N=2 is deliberately excluded here: Phase 15 gave it its own
        // measured threshold (see test_all_gather_threshold_boundaries),
        // so it is no longer "unmeasured" for AllGather.
        for (std::size_t world_size : {1, 5, 6, 7, 9, 16})
        {
            for (std::size_t bytes : sample_bytes)
            {
                expect_decision(
                    tbccl::detail::select_all_gather_algorithm(
                        world_size, bytes, AlgorithmMode::Auto),
                    CollectiveAlgorithm::Reference,
                    SelectionReason::AutoUnmeasuredWorldSize,
                    "all_gather unmeasured N=" + std::to_string(world_size));
            }
        }

        for (std::size_t world_size : {1, 5, 6, 7, 9, 16})
        {
            for (std::size_t bytes : sample_bytes)
            {
                expect_decision(
                    tbccl::detail::select_reduce_scatter_algorithm(
                        world_size, bytes, AlgorithmMode::Auto),
                    CollectiveAlgorithm::Reference,
                    SelectionReason::AutoUnmeasuredWorldSize,
                    "reduce_scatter unmeasured N=" +
                        std::to_string(world_size));
            }
        }

        for (std::size_t world_size : {1, 2, 3, 5, 6, 7, 9, 16})
        {
            for (std::size_t bytes : sample_bytes)
            {
                expect_decision(
                    tbccl::detail::select_all_reduce_algorithm(
                        world_size, bytes, world_size, AlgorithmMode::Auto),
                    CollectiveAlgorithm::Reference,
                    SelectionReason::AutoUnmeasuredWorldSize,
                    "all_reduce unmeasured N=" + std::to_string(world_size));
            }
        }

        std::cout << "[PASS] test_unmeasured_world_sizes\n";
    }

    void test_all_reduce_auto_eligibility()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveAlgorithm;
        using tbccl::detail::SelectionReason;

        constexpr std::size_t kWorldSize = 4;
        constexpr std::size_t kTensorBytes = 4ull * 1024 * 1024;

        expect_decision(
            tbccl::detail::select_all_reduce_algorithm(
                kWorldSize, kTensorBytes, 1024, AlgorithmMode::Auto),
            CollectiveAlgorithm::Ring, SelectionReason::AutoRingThreshold,
            "all_reduce auto, above threshold, divisible count");

        expect_decision(
            tbccl::detail::select_all_reduce_algorithm(
                kWorldSize, kTensorBytes, 1025, AlgorithmMode::Auto),
            CollectiveAlgorithm::Reference,
            SelectionReason::AutoRingUnsupported,
            "all_reduce auto, above threshold, non-divisible count");

        std::cout << "[PASS] test_all_reduce_auto_eligibility\n";
    }

    // -------------------------------------------------------------------
    // Environment parsing tests.
    // -------------------------------------------------------------------

    void test_parse_algorithm_mode_direct()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::parse_algorithm_mode;

        for (const char *literal : {"auto", "AUTO", "Auto", "aUtO"})
        {
            const std::string text = literal;

            expect(
                parse_algorithm_mode("TEST_VAR", text) == AlgorithmMode::Auto,
                "parse_algorithm_mode should accept \"" + text + "\" as auto");
        }

        for (const char *literal :
             {"reference", "REFERENCE", "Reference"})
        {
            const std::string text = literal;

            expect(
                parse_algorithm_mode("TEST_VAR", text) ==
                    AlgorithmMode::Reference,
                "parse_algorithm_mode should accept \"" + text +
                    "\" as reference");
        }

        for (const char *literal : {"ring", "RING", "Ring"})
        {
            const std::string text = literal;

            expect(
                parse_algorithm_mode("TEST_VAR", text) == AlgorithmMode::Ring,
                "parse_algorithm_mode should accept \"" + text + "\" as ring");
        }

        for (const char *literal : {"tree", "", "auto2", "ringed"})
        {
            const std::string text = literal;
            bool threw = false;

            try
            {
                parse_algorithm_mode("TBCCL_ALL_REDUCE_ALGORITHM", text);
            }
            catch (const std::exception &error)
            {
                threw = true;

                const std::string message = error.what();

                expect(
                    message.find("TBCCL_ALL_REDUCE_ALGORITHM") !=
                        std::string::npos,
                    "error message should name the variable: " + message);
                expect(
                    message.find("auto, reference, or ring") !=
                        std::string::npos,
                    "error message should list valid values: " + message);
            }

            expect(
                threw,
                "parse_algorithm_mode should reject \"" + text + "\"");
        }

        std::cout << "[PASS] test_parse_algorithm_mode_direct\n";
    }

    void test_resolve_algorithm_mode_unset()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveKind;
        using tbccl::detail::resolve_algorithm_mode;

        EnvUnset unset_global("TBCCL_ALGORITHM");
        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");
        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        expect(
            resolve_algorithm_mode(CollectiveKind::AllGather) ==
                AlgorithmMode::Auto,
            "no overrides set -> AllGather resolves Auto");
        expect(
            resolve_algorithm_mode(CollectiveKind::ReduceScatter) ==
                AlgorithmMode::Auto,
            "no overrides set -> ReduceScatter resolves Auto");
        expect(
            resolve_algorithm_mode(CollectiveKind::AllReduce) ==
                AlgorithmMode::Auto,
            "no overrides set -> AllReduce resolves Auto");

        std::cout << "[PASS] test_resolve_algorithm_mode_unset\n";
    }

    void test_resolve_algorithm_mode_global()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveKind;
        using tbccl::detail::resolve_algorithm_mode;

        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");
        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        {
            EnvOverride global("TBCCL_ALGORITHM", "reference");

            expect(
                resolve_algorithm_mode(CollectiveKind::AllGather) ==
                    AlgorithmMode::Reference,
                "TBCCL_ALGORITHM=reference -> AllGather Reference");
            expect(
                resolve_algorithm_mode(CollectiveKind::ReduceScatter) ==
                    AlgorithmMode::Reference,
                "TBCCL_ALGORITHM=reference -> ReduceScatter Reference");
            expect(
                resolve_algorithm_mode(CollectiveKind::AllReduce) ==
                    AlgorithmMode::Reference,
                "TBCCL_ALGORITHM=reference -> AllReduce Reference");
        }

        {
            EnvOverride global("TBCCL_ALGORITHM", "RING");

            expect(
                resolve_algorithm_mode(CollectiveKind::AllGather) ==
                    AlgorithmMode::Ring,
                "TBCCL_ALGORITHM=RING (uppercase) -> AllGather Ring");
        }

        {
            EnvOverride global("TBCCL_ALGORITHM", "Auto");

            expect(
                resolve_algorithm_mode(CollectiveKind::AllReduce) ==
                    AlgorithmMode::Auto,
                "TBCCL_ALGORITHM=Auto -> AllReduce Auto");
        }

        std::cout << "[PASS] test_resolve_algorithm_mode_global\n";
    }

    void test_resolve_algorithm_mode_invalid()
    {
        using tbccl::detail::CollectiveKind;
        using tbccl::detail::resolve_algorithm_mode;

        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");
        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        {
            EnvOverride global("TBCCL_ALGORITHM", "tree");
            bool threw = false;

            try
            {
                resolve_algorithm_mode(CollectiveKind::AllGather);
            }
            catch (const std::exception &error)
            {
                threw = true;
                expect(
                    std::string(error.what()).find("TBCCL_ALGORITHM") !=
                        std::string::npos,
                    "invalid TBCCL_ALGORITHM error names itself");
            }

            expect(threw, "invalid TBCCL_ALGORITHM should throw");
        }

        {
            EnvUnset unset_global("TBCCL_ALGORITHM");
            EnvOverride per_collective(
                "TBCCL_ALL_REDUCE_ALGORITHM", "bogus");
            bool threw = false;

            try
            {
                resolve_algorithm_mode(CollectiveKind::AllReduce);
            }
            catch (const std::exception &error)
            {
                threw = true;
                expect(
                    std::string(error.what())
                            .find("TBCCL_ALL_REDUCE_ALGORITHM") !=
                        std::string::npos,
                    "invalid TBCCL_ALL_REDUCE_ALGORITHM error names itself");
            }

            expect(
                threw, "invalid TBCCL_ALL_REDUCE_ALGORITHM should throw");
        }

        std::cout << "[PASS] test_resolve_algorithm_mode_invalid\n";
    }

    void test_resolve_algorithm_mode_precedence()
    {
        using tbccl::detail::AlgorithmMode;
        using tbccl::detail::CollectiveKind;
        using tbccl::detail::resolve_algorithm_mode;

        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        {
            EnvOverride global("TBCCL_ALGORITHM", "reference");
            EnvOverride per_collective("TBCCL_ALL_GATHER_ALGORITHM", "ring");

            expect(
                resolve_algorithm_mode(CollectiveKind::AllGather) ==
                    AlgorithmMode::Ring,
                "per-collective override wins: AllGather -> Ring");
            expect(
                resolve_algorithm_mode(CollectiveKind::ReduceScatter) ==
                    AlgorithmMode::Reference,
                "global fallback: ReduceScatter -> Reference");
            expect(
                resolve_algorithm_mode(CollectiveKind::AllReduce) ==
                    AlgorithmMode::Reference,
                "global fallback: AllReduce -> Reference");
        }

        {
            EnvOverride global("TBCCL_ALGORITHM", "ring");
            EnvOverride per_collective(
                "TBCCL_ALL_GATHER_ALGORITHM", "reference");

            expect(
                resolve_algorithm_mode(CollectiveKind::AllGather) ==
                    AlgorithmMode::Reference,
                "reversed: per-collective override wins: AllGather -> "
                "Reference");
            expect(
                resolve_algorithm_mode(CollectiveKind::ReduceScatter) ==
                    AlgorithmMode::Ring,
                "reversed: global fallback: ReduceScatter -> Ring");
            expect(
                resolve_algorithm_mode(CollectiveKind::AllReduce) ==
                    AlgorithmMode::Ring,
                "reversed: global fallback: AllReduce -> Ring");
        }

        std::cout << "[PASS] test_resolve_algorithm_mode_precedence\n";
    }

    // -------------------------------------------------------------------
    // Distributed integration tests — prove the public wrappers
    // dispatch correctly. Exhaustive reduction/gather correctness is
    // already covered by all_gather_test/reduce_scatter_test/
    // all_reduce_test and their _ring_test counterparts; this file's
    // job is only to prove no dispatch-integration bug exists between
    // the selector and the public API, so a simple constant-fill
    // pattern (value = rank+1) is used throughout.
    // -------------------------------------------------------------------

    void run_all_gather_dispatch(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t bytes_per_rank)
    {
        auto peers = make_local_peers(base_port, size);
        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, 15000),
                [rank, bytes_per_rank](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();
                    const auto send = deterministic_buffer(
                        bytes_per_rank,
                        static_cast<std::uint32_t>(rank) + 0x7000u);
                    std::vector<std::uint8_t> recv(
                        size2 * bytes_per_rank, 0);

                    tbccl::all_gather(
                        world, send.data(), recv.data(), bytes_per_rank);

                    for (std::size_t r = 0; r < size2; ++r)
                    {
                        const auto expected = deterministic_buffer(
                            bytes_per_rank,
                            static_cast<std::uint32_t>(r) + 0x7000u);

                        expect(
                            std::memcmp(
                                recv.data() + r * bytes_per_rank,
                                expected.data(), bytes_per_rank) == 0,
                            "public all_gather() mismatch on rank " +
                                std::to_string(rank) + " slot " +
                                std::to_string(r));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    void run_reduce_scatter_dispatch(
        std::uint16_t base_port, std::size_t size, std::size_t recv_count)
    {
        auto peers = make_local_peers(base_port, size);
        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, 15000),
                [rank, recv_count](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();
                    const std::size_t total_count = size2 * recv_count;

                    // Every element of this rank's whole contribution
                    // is (rank+1), so every output segment reduces to
                    // the same sum regardless of which segment it is.
                    std::vector<std::int32_t> send(
                        total_count, static_cast<std::int32_t>(rank + 1));

                    std::vector<std::int32_t> recv(recv_count, 0);

                    tbccl::reduce_scatter(
                        world, send.data(), recv.data(), recv_count,
                        tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                    const auto expected = static_cast<std::int32_t>(
                        size2 * (size2 + 1) / 2);

                    for (std::size_t i = 0; i < recv_count; ++i)
                    {
                        expect(
                            recv[i] == expected,
                            "public reduce_scatter() mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    void run_all_reduce_dispatch(
        std::uint16_t base_port, std::size_t size, std::size_t count)
    {
        auto peers = make_local_peers(base_port, size);
        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, 15000),
                [rank, count](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();

                    std::vector<std::int32_t> send(count);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        send[i] = static_cast<std::int32_t>(rank + 1);
                    }

                    std::vector<std::int32_t> recv(count, 0);

                    tbccl::all_reduce(
                        world, send.data(), recv.data(), count,
                        tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                    const auto expected = static_cast<std::int32_t>(
                        size2 * (size2 + 1) / 2);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        expect(
                            recv[i] == expected,
                            "public all_reduce() mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    void test_public_all_gather_auto_ring()
    {
        EnvUnset unset_global("TBCCL_ALGORITHM");
        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");

        // N=4, bytes_per_rank well above the v2 32 KiB threshold ->
        // Auto resolves Ring.
        run_all_gather_dispatch(kAllGatherAutoRingBase, 4, 1048576);

        std::cout << "[PASS] test_public_all_gather_auto_ring\n";
    }

    void test_public_all_gather_auto_ring_n2()
    {
        EnvUnset unset_global("TBCCL_ALGORITHM");
        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");

        // N=2 is a Phase 15 policy change (was always Reference under
        // the per-invocation-thread ring work): Ring now resolves at
        // any tested size, so even a small,
        // otherwise-Reference-favoring-under-the-old-policy 4 KiB
        // contribution must dispatch through ring_all_gather() and
        // still produce correct output.
        run_all_gather_dispatch(kAllGatherAutoRingN2Base, 2, 4096);

        std::cout << "[PASS] test_public_all_gather_auto_ring_n2\n";
    }

    void test_public_reduce_scatter_auto_ring()
    {
        EnvUnset unset_global("TBCCL_ALGORITHM");
        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");

        // N=2, recv_count=131072 Int32 elements -> segment_bytes =
        // 524288 B, well above the v2 128 KiB threshold -> Auto
        // resolves Ring.
        run_reduce_scatter_dispatch(kReduceScatterAutoRingBase, 2, 131072);

        std::cout << "[PASS] test_public_reduce_scatter_auto_ring\n";
    }

    void test_public_all_reduce_auto_ring()
    {
        EnvUnset unset_global("TBCCL_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        // N=4, count=524292 Int32 elements -> tensor_bytes = 2097168 B
        // (just above the 2 MiB threshold) and divisible by 4 -> Auto
        // resolves Ring.
        run_all_reduce_dispatch(kAllReduceAutoRingBase, 4, 524292);

        std::cout << "[PASS] test_public_all_reduce_auto_ring\n";
    }

    void test_public_all_reduce_auto_ineligible_fallback()
    {
        EnvUnset unset_global("TBCCL_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        // N=4, count=524293 Int32 elements -> tensor_bytes = 2097172 B,
        // still above the 2 MiB threshold, but NOT divisible by 4 ->
        // Auto must fall back to Reference instead of throwing.
        run_all_reduce_dispatch(kAllReduceAutoIneligibleBase, 4, 524293);

        std::cout
            << "[PASS] test_public_all_reduce_auto_ineligible_fallback\n";
    }

    void test_forced_reference_integration()
    {
        EnvOverride global("TBCCL_ALGORITHM", "reference");
        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");
        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        // Same inputs that resolve Ring under Auto in the tests above
        // — TBCCL_ALGORITHM=reference must force Reference instead,
        // and results must still be correct.
        run_all_gather_dispatch(kForcedRefAllGatherBase, 4, 1048576);
        run_reduce_scatter_dispatch(
            kForcedRefReduceScatterBase, 2, 131072);
        run_all_reduce_dispatch(kForcedRefAllReduceBase, 4, 524292);

        std::cout << "[PASS] test_forced_reference_integration\n";
    }

    void test_forced_ring_integration()
    {
        EnvOverride global("TBCCL_ALGORITHM", "ring");
        EnvUnset unset_ag("TBCCL_ALL_GATHER_ALGORITHM");
        EnvUnset unset_rs("TBCCL_REDUCE_SCATTER_ALGORITHM");
        EnvUnset unset_ar("TBCCL_ALL_REDUCE_ALGORITHM");

        // Small sizes that would resolve Reference under Auto —
        // forced Ring must still be selected and still produce correct
        // results.
        run_all_gather_dispatch(kForcedRingAllGatherBase, 4, 64);
        run_reduce_scatter_dispatch(kForcedRingReduceScatterBase, 4, 8);
        run_all_reduce_dispatch(kForcedRingAllReduceBase, 4, 16);

        std::cout << "[PASS] test_forced_ring_integration\n";
    }

    void test_forced_ring_all_reduce_nondivisible_rejects()
    {
        EnvOverride per_collective("TBCCL_ALL_REDUCE_ALGORITHM", "ring");

        constexpr std::size_t kSize = 3;
        constexpr std::size_t kCount = 10; // not divisible by 3

        auto peers =
            make_local_peers(kForcedRingNondivisibleBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers),
                [](tbccl::World &world)
                {
                    std::vector<std::int32_t> send(kCount, 1);
                    std::vector<std::int32_t> recv(kCount, 0);
                    bool threw = false;

                    try
                    {
                        tbccl::all_reduce(
                            world, send.data(), recv.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;

                        const std::string message = error.what();

                        expect(
                            message.find("not divisible") !=
                                std::string::npos,
                            "wrong error for forced-ring non-divisible "
                            "count: " +
                                message);
                    }

                    expect(
                        threw,
                        "public all_reduce() with "
                        "TBCCL_ALL_REDUCE_ALGORITHM=ring should reject a "
                        "non-divisible count");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_forced_ring_all_reduce_nondivisible_rejects\n";
    }

} // namespace

int main()
{
    try
    {
        test_algorithm_and_reason_names();
        test_mode_reference_always_reference();
        test_mode_ring_always_ring();
        test_all_gather_threshold_boundaries();
        test_reduce_scatter_threshold_boundaries();
        test_all_reduce_threshold_boundaries();
        test_unmeasured_world_sizes();
        test_all_reduce_auto_eligibility();

        test_parse_algorithm_mode_direct();
        test_resolve_algorithm_mode_unset();
        test_resolve_algorithm_mode_global();
        test_resolve_algorithm_mode_invalid();
        test_resolve_algorithm_mode_precedence();

        test_public_all_gather_auto_ring();
        test_public_all_gather_auto_ring_n2();
        test_public_reduce_scatter_auto_ring();
        test_public_all_reduce_auto_ring();
        test_public_all_reduce_auto_ineligible_fallback();
        test_forced_reference_integration();
        test_forced_ring_integration();
        test_forced_ring_all_reduce_nondivisible_rejects();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
