// N>2 all_reduce algorithms, one forced algorithm per run (argv[1] = reference | tree | recursive | ring), Host loopback, world_size 3, 4, 5, 8 (recursive: 4, 8).
//
// Numerics follow docs/reference/numerical-semantics.md. Float32/Float64: every rank must receive IDENTICAL bits (NaN lanes compared as "is NaN"), the result must be
// deterministic across repeated runs of the same algorithm, and it must agree with a high-precision reference (compensated summation in double) within
// (N-1) * epsilon * sum(|x_i|): huge/tiny magnitudes, cancellation, subnormals, +-inf and NaN included. NO bitwise equality between algorithms is required or checked.
// Int32/Int64/Int8/UInt8: exact modular arithmetic, no tolerance, whatever the algorithm and order.

#include "mesh_test_support.hpp"

#include "collective_topology.hpp"
#include "communicator_debug.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

namespace
{
    bool has(const std::string &t, const std::string &n) { return t.find(n) != std::string::npos; }
    BufferView view(void *p, std::size_t bytes) { return BufferView{MemoryKind::Host, p, bytes, 0}; }

    template <typename T> DataType dtype_of();
    template <> DataType dtype_of<float>() { return DataType::Float32; }
    template <> DataType dtype_of<double>() { return DataType::Float64; }
    template <> DataType dtype_of<std::int32_t>() { return DataType::Int32; }
    template <> DataType dtype_of<std::int64_t>() { return DataType::Int64; }
    template <> DataType dtype_of<std::int8_t>() { return DataType::Int8; }
    template <> DataType dtype_of<std::uint8_t>() { return DataType::UInt8; }

    enum class Kind { Random, Positive, Wide, Cancellation, Subnormal, Specials };
    const char *kind_name(Kind k)
    {
        switch (k)
        {
        case Kind::Random: return "random";
        case Kind::Positive: return "positive";
        case Kind::Wide: return "wide-range";
        case Kind::Cancellation: return "cancellation";
        case Kind::Subnormal: return "subnormal";
        case Kind::Specials: return "inf/nan";
        }
        return "?";
    }

    template <typename T> std::vector<T> make_inputs(Kind kind, std::size_t rank, std::size_t count, std::uint32_t seed)
    {
        std::mt19937_64 gen(seed * 7919u + rank * 104729u + count * 13u + static_cast<std::uint64_t>(kind) * 31u);
        std::vector<T> v(count);
        if constexpr (std::is_floating_point_v<T>)
        {
            std::uniform_real_distribution<double> u(-1000.0, 1000.0), pos(1.0, 2.0), mant(0.5, 1.0);
            std::uniform_int_distribution<int> expo(std::is_same_v<T, float> ? -60 : -200, std::is_same_v<T, float> ? 60 : 200), pick(0, 7), coin(0, 1);
            // a shared random stream for cancellation so every rank can reproduce rank 0's values
            std::mt19937_64 shared(seed * 31337u + count * 17u);
            for (std::size_t i = 0; i < count; ++i)
            {
                switch (kind)
                {
                case Kind::Random: v[i] = static_cast<T>(u(gen)); break;
                case Kind::Positive: v[i] = static_cast<T>(pos(gen)); break;
                case Kind::Wide: v[i] = static_cast<T>(std::ldexp(mant(gen) * (coin(gen) ? 1 : -1), expo(gen))); break;
                case Kind::Cancellation:
                {
                    const T x = static_cast<T>(std::uniform_real_distribution<double>(-1e6, 1e6)(shared));
                    v[i] = rank == 0 ? x : rank == 1 ? -x : static_cast<T>(u(gen) * 1e-3);
                    break;
                }
                case Kind::Subnormal: v[i] = std::numeric_limits<T>::denorm_min() * static_cast<T>(1 + static_cast<int>(gen() % 1000)) * (coin(gen) ? 1 : -1); break;
                case Kind::Specials:
                {
                    const int p = pick(gen);
                    v[i] = p == 0 ? std::numeric_limits<T>::infinity() : p == 1 ? -std::numeric_limits<T>::infinity() : p == 2 ? std::numeric_limits<T>::quiet_NaN()
                         : p == 3 ? static_cast<T>(-0.0) : static_cast<T>(u(gen));
                    // make specials rare enough that finite lanes exist, but present in every size
                    if (gen() % 5 != 0) v[i] = static_cast<T>(u(gen));
                    break;
                }
                }
            }
        }
        else
        {
            for (auto &x : v) x = static_cast<T>(gen());
        }
        return v;
    }

    // Reference for one lane: returns false if the lane is special (inf/nan) with the expected classification in `special`.
    template <typename T> struct LaneRef
    {
        bool is_nan = false, is_pos_inf = false, is_neg_inf = false;
        double sum = 0, abs_sum = 0;
    };

    template <typename T> LaneRef<T> lane_reference(const std::vector<std::vector<T>> &inputs, std::size_t i)
    {
        LaneRef<T> r;
        bool pos = false, neg = false;
        double s = 0, c = 0; // Neumaier compensated sum in double
        for (const auto &in : inputs)
        {
            const double x = static_cast<double>(in[i]);
            if (std::isnan(x)) r.is_nan = true;
            else if (std::isinf(x)) (x > 0 ? pos : neg) = true;
            else
            {
                const double t = s + x;
                c += (std::fabs(s) >= std::fabs(x)) ? (s - t) + x : (x - t) + s;
                s = t;
                r.abs_sum += std::fabs(x);
            }
        }
        r.sum = s + c;
        if (pos && neg) r.is_nan = true;
        r.is_pos_inf = pos && !r.is_nan;
        r.is_neg_inf = neg && !r.is_nan;
        return r;
    }

    template <typename T> bool same_bits_modulo_nan(T a, T b)
    {
        if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }

    struct Config
    {
        std::string algorithm;
        std::vector<std::size_t> worlds;
    };

    // One world per size is reused by every case of the matrix (see mesh_test::World); the edge check below builds its own fresh one.
    std::map<std::size_t, std::unique_ptr<mesh_test::World>> g_worlds;

    mesh_test::World &pooled_world(std::size_t world)
    {
        auto &w = g_worlds[world];
        if (!w) w = std::make_unique<mesh_test::World>(world);
        return *w;
    }

    // Runs one all_reduce and returns every rank's result.
    template <typename T> std::vector<std::vector<T>> run_all_reduce(std::size_t world, const std::vector<std::vector<T>> &inputs, bool in_place, std::set<std::size_t> *edges_out = nullptr)
    {
        std::vector<std::vector<T>> results(world);
        std::vector<std::set<std::size_t>> edges(world);
        const std::size_t count = inputs[0].size();
        const auto body = [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<T> mine = inputs[rank], out(count, T{});
            const std::size_t bytes = count * sizeof(T);
            tbccl::Work w = in_place ? comm.all_reduce(view(mine.data(), bytes), view(mine.data(), bytes), count, dtype_of<T>(), ReduceOp::Sum)
                                     : comm.all_reduce(view(mine.data(), bytes), view(out.data(), bytes), count, dtype_of<T>(), ReduceOp::Sum);
            w.wait();
            expect(!w.has_error(), "all_reduce: " + w.error());
            if (!in_place) expect(mine == inputs[rank] || std::is_floating_point_v<T>, "the input of an out-of-place all_reduce is unchanged");
            results[rank] = in_place ? mine : out;
            auto v = tbccl::detail::debug_connected_data_peers(comm);
            edges[rank] = std::set<std::size_t>(v.begin(), v.end());
        };
        if (edges_out) run_world(world, body);
        else pooled_world(world).run(body);
        if (edges_out) for (std::size_t r = 0; r < world; ++r) edges_out[r] = edges[r];
        return results;
    }

    template <typename T> void check_float_case(const Config &cfg, std::size_t world, std::size_t count, Kind kind, bool in_place)
    {
        std::vector<std::vector<T>> inputs;
        for (std::size_t r = 0; r < world; ++r) inputs.push_back(make_inputs<T>(kind, r, count, 11));
        const auto first = run_all_reduce<T>(world, inputs, in_place);
        const auto again = run_all_reduce<T>(world, inputs, in_place);
        const std::string label = std::string(std::is_same_v<T, float> ? "float32" : "float64") + " " + kind_name(kind) + " world=" + std::to_string(world) + " count=" + std::to_string(count) +
                                  (in_place ? " in-place" : " out-of-place") + " algorithm=" + cfg.algorithm;
        for (std::size_t i = 0; i < count; ++i)
        {
            for (std::size_t r = 1; r < world; ++r) expect(same_bits_modulo_nan(first[0][i], first[r][i]), label + ": ranks 0 and " + std::to_string(r) + " disagree at element " + std::to_string(i));
            for (std::size_t r = 0; r < world; ++r) expect(same_bits_modulo_nan(first[r][i], again[r][i]), label + ": not deterministic at element " + std::to_string(i));
            const auto ref = lane_reference<T>(inputs, i);
            const T got = first[0][i];
            if (ref.is_nan) { expect(std::isnan(got), label + ": expected NaN at element " + std::to_string(i)); continue; }
            if (ref.is_pos_inf) { expect(std::isinf(got) && got > 0, label + ": expected +inf at element " + std::to_string(i)); continue; }
            if (ref.is_neg_inf) { expect(std::isinf(got) && got < 0, label + ": expected -inf at element " + std::to_string(i)); continue; }
            const double eps = std::numeric_limits<T>::epsilon();
            const double tol = static_cast<double>(world - 1) * eps * ref.abs_sum + 4.0 * static_cast<double>(std::numeric_limits<T>::denorm_min()) * static_cast<double>(world);
            expect(std::isfinite(static_cast<double>(got)) && std::fabs(static_cast<double>(got) - ref.sum) <= tol,
                   label + ": element " + std::to_string(i) + " got " + std::to_string(static_cast<double>(got)) + " reference " + std::to_string(ref.sum) + " tolerance " + std::to_string(tol));
        }
    }

    template <typename T> void check_int_case(const Config &cfg, std::size_t world, std::size_t count, bool in_place)
    {
        std::vector<std::vector<T>> inputs;
        for (std::size_t r = 0; r < world; ++r) inputs.push_back(make_inputs<T>(Kind::Random, r, count, 3));
        const auto got = run_all_reduce<T>(world, inputs, in_place);
        using U = std::make_unsigned_t<T>;
        for (std::size_t i = 0; i < count; ++i)
        {
            U acc = 0;
            for (const auto &in : inputs) acc = static_cast<U>(acc + static_cast<U>(in[i]));
            for (std::size_t r = 0; r < world; ++r)
                expect(static_cast<U>(got[r][i]) == acc, std::string("integer all_reduce must be exact: dtype=") + tbccl::datatype_label(dtype_of<T>()) + " world=" + std::to_string(world) + " count=" + std::to_string(count) +
                       " element " + std::to_string(i) + " rank " + std::to_string(r) + " algorithm=" + cfg.algorithm);
        }
    }

    void run_matrix(const Config &cfg)
    {
        for (std::size_t world : cfg.worlds)
        {
            for (std::size_t count : {std::size_t{1}, std::size_t{2}, world - 1, world, world + 1, std::size_t{7}, std::size_t{17}, std::size_t{1000}, std::size_t{16384}, std::size_t{262144}})
            {
                for (bool in_place : {true, false})
                {
                    if (count > 16384 && !in_place) continue;
                    check_int_case<std::int32_t>(cfg, world, count, in_place);
                    check_int_case<std::int64_t>(cfg, world, count, in_place);
                    check_int_case<std::int8_t>(cfg, world, count, in_place);
                    check_int_case<std::uint8_t>(cfg, world, count, in_place);
                    for (Kind kind : {Kind::Random, Kind::Positive, Kind::Wide, Kind::Cancellation, Kind::Subnormal, Kind::Specials})
                    {
                        if (count > 16384 && kind != Kind::Random && kind != Kind::Positive) continue;
                        check_float_case<float>(cfg, world, count, kind, in_place);
                        check_float_case<double>(cfg, world, count, kind, in_place);
                    }
                }
            }
            g_worlds.erase(world);
            std::cout << "[PASS] all_reduce algorithm=" << cfg.algorithm << " world_size=" << world << ": Float32/Float64 vs the high-precision reference (random, positive, wide range, cancellation, subnormal, inf/NaN), "
                      << "rank agreement and determinism; Int32/Int64/Int8/UInt8 exact; in-place and out-of-place\n";
        }
    }

    // The data edges a run actually uses are exactly what the algorithm needs (descriptors and verdicts are on the control plane).
    void test_edges(const Config &cfg)
    {
        if (cfg.algorithm == "reference") return;
        for (std::size_t world : cfg.worlds)
        {
            std::vector<std::set<std::size_t>> edges(world);
            std::vector<std::vector<float>> inputs;
            for (std::size_t r = 0; r < world; ++r) inputs.push_back(make_inputs<float>(Kind::Random, r, 5000, 1));
            run_all_reduce<float>(world, inputs, true, edges.data());
            for (std::size_t r = 0; r < world; ++r)
            {
                std::set<std::size_t> want;
                if (cfg.algorithm == "ring") { want.insert(tbccl::detail::ring_next(r, world)); want.insert(tbccl::detail::ring_prev(r, world)); }
                else if (cfg.algorithm == "tree")
                {
                    for (std::size_t c : tbccl::detail::tree_children(r, 0, world)) want.insert(c);
                    if (r != 0) want.insert(tbccl::detail::tree_parent(r, 0, world));
                }
                else for (std::size_t m = 1; m < world; m <<= 1) want.insert(r ^ m);
                expect(edges[r] == want, "all_reduce " + cfg.algorithm + " world=" + std::to_string(world) + " rank " + std::to_string(r) + " uses exactly its algorithm's data edges");
            }
        }
        std::cout << "[PASS] all_reduce " << cfg.algorithm << ": only the algorithm's own data edges are connected\n";
    }

    void test_low_precision_still_rejected(const Config &cfg)
    {
        for (std::size_t world : {std::size_t{3}, std::size_t{4}})
        {
            run_world(world, [&](std::size_t, tbccl::Communicator &comm) {
                std::vector<std::uint16_t> half(64, 0x3c00);
                for (auto dt : {DataType::Float16, DataType::BFloat16})
                {
                    bool threw = false;
                    try
                    {
                        comm.all_reduce(view(half.data(), 128), view(half.data(), 128), 64, dt, ReduceOp::Sum);
                    }
                    catch (const std::exception &e)
                    {
                        threw = has(e.what(), "N>2 reduction semantics are not defined");
                    }
                    expect(threw, std::string(tbccl::datatype_label(dt)) + " is still rejected at N>2 with algorithm " + cfg.algorithm + " forced");
                }
            });
        }
        std::cout << "[PASS] Float16/BFloat16 SUM still rejected for N>2 (algorithm " << cfg.algorithm << " forced)\n";
    }

    void test_unsupported_forcing(const Config &cfg)
    {
        if (cfg.algorithm != "recursive") return;
        for (std::size_t world : {std::size_t{3}, std::size_t{5}})
        {
            run_world(world, [&](std::size_t, tbccl::Communicator &comm) {
                std::vector<float> v(64, 1.0f);
                auto w = comm.all_reduce(view(v.data(), 256), view(v.data(), 256), 64, DataType::Float32, ReduceOp::Sum);
                w.wait();
                expect(w.has_error() && has(w.error(), "power-of-two"), "recursive doubling at a non-power-of-two world is rejected by every rank: " + w.error());
                expect(!comm.failed(), "a rejected forced algorithm poisons nothing");
            });
        }
        std::cout << "[PASS] recursive doubling forced at a non-power-of-two world is rejected cleanly\n";
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: communicator_allreduce_algorithms_test reference|tree|recursive|ring\n";
        return 2;
    }
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(900));
        std::cerr << "[FAIL] watchdog: an all_reduce algorithm test hung\n";
        std::_Exit(2);
    }).detach();
    Config cfg;
    cfg.algorithm = argv[1];
    cfg.worlds = cfg.algorithm == "recursive" ? std::vector<std::size_t>{4, 8} : std::vector<std::size_t>{3, 4, 5, 8};
    setenv("TBCCL_ALLREDUCE_ALGORITHM", cfg.algorithm.c_str(), 1);
    try
    {
        run_matrix(cfg);
        test_edges(cfg);
        test_low_precision_still_rejected(cfg);
        test_unsupported_forcing(cfg);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All all_reduce algorithm tests passed (" << cfg.algorithm << ").\n";
    return 0;
}
