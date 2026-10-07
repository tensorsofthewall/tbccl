// Host correctness of Int8 / UInt8 SUM. Overflow semantics are explicit: SUM is addition modulo 256 (two's complement for Int8),
// computed through unsigned arithmetic so no signed overflow ever occurs (UBSan must stay silent on every case here, including the overflow ones).
//
//   1. Every one of the 65536 (a, b) pairs of each type against an independent reference written with unsigned arithmetic only.
//   2. The overflow cases from the plan, by hand: 120 + 100, 127 + 1, -128 + -1, -100 + -100, 255 + 1, 200 + 100, 255 + 255.
//   3. Deterministic random vectors of assorted lengths through the shared apply_reduction kernel.
//   4. The same arithmetic through the real N=2 collectives on TCP loopback: World reference/ring/pipelined all_reduce and the production
//      Communicator::all_reduce on host buffers (in place and out of place).

#include <tbccl/collectives.hpp>
#include <tbccl/communicator.hpp>
#include <tbccl/reduction.hpp>

#include "all_reduce_internal.hpp"
#include "reduction_internal.hpp"
#include "test_utils.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using tbccl_test::expect;

namespace
{

    template <class T>
    struct Traits;

    template <>
    struct Traits<std::int8_t>
    {
        static constexpr const char *name = "int8";
        static constexpr tbccl::DataType dt = tbccl::DataType::Int8;
    };

    template <>
    struct Traits<std::uint8_t>
    {
        static constexpr const char *name = "uint8";
        static constexpr tbccl::DataType dt = tbccl::DataType::UInt8;
    };

    // Independent reference: modulo-256 addition on the raw byte, nothing signed involved.
    template <class T>
    T reference_sum(T a, T b)
    {
        const unsigned wide = static_cast<unsigned>(static_cast<std::uint8_t>(a)) + static_cast<unsigned>(static_cast<std::uint8_t>(b));
        return static_cast<T>(static_cast<std::uint8_t>(wide & 0xFFu));
    }

    template <class T>
    T sum_one(T a, T b)
    {
        tbccl::detail::apply_reduction(&a, &b, 1, tbccl::ReduceOp::Sum);
        return a;
    }

    std::uint32_t next_random(std::uint32_t &state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    template <class T>
    std::vector<T> random_values(std::size_t count, std::uint32_t seed)
    {
        std::vector<T> v(count);
        std::uint32_t state = seed;
        for (auto &x : v) x = static_cast<T>(next_random(state) >> 11);
        return v;
    }

    template <class T>
    void test_all_pairs()
    {
        std::vector<T> a(65536), b(65536);
        for (std::uint32_t i = 0; i < 65536; ++i)
        {
            a[i] = static_cast<T>(static_cast<std::uint8_t>(i >> 8));
            b[i] = static_cast<T>(static_cast<std::uint8_t>(i & 0xFF));
        }
        std::vector<T> got = a;
        tbccl::detail::apply_reduction(got.data(), b.data(), got.size(), tbccl::ReduceOp::Sum);
        for (std::size_t i = 0; i < got.size(); ++i)
        {
            expect(got[i] == reference_sum(a[i], b[i]), std::string(Traits<T>::name) + ": all-pairs sum mismatch at pair " + std::to_string(i));
        }
        std::cout << "[PASS] test_all_pairs<" << Traits<T>::name << "> (65536 pairs)\n";
    }

    void test_overflow_cases()
    {
        using I = std::int8_t;
        using U = std::uint8_t;
        expect(sum_one<I>(120, 100) == static_cast<I>(-36), "int8 120 + 100 wraps to -36");
        expect(sum_one<I>(127, 1) == static_cast<I>(-128), "int8 127 + 1 wraps to -128");
        expect(sum_one<I>(-128, -1) == static_cast<I>(127), "int8 -128 + -1 wraps to 127");
        expect(sum_one<I>(-100, -100) == static_cast<I>(56), "int8 -100 + -100 wraps to 56");
        expect(sum_one<I>(-128, -128) == static_cast<I>(0), "int8 -128 + -128 wraps to 0");
        expect(sum_one<I>(127, 127) == static_cast<I>(-2), "int8 127 + 127 wraps to -2");
        expect(sum_one<U>(255, 1) == static_cast<U>(0), "uint8 255 + 1 wraps to 0");
        expect(sum_one<U>(200, 100) == static_cast<U>(44), "uint8 200 + 100 wraps to 44");
        expect(sum_one<U>(255, 255) == static_cast<U>(254), "uint8 255 + 255 wraps to 254");
        expect(sum_one<U>(128, 128) == static_cast<U>(0), "uint8 128 + 128 wraps to 0");
        std::cout << "[PASS] test_overflow_cases\n";
    }

    template <class T>
    void test_random_vectors()
    {
        for (std::size_t count : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{17}, std::size_t{255}, std::size_t{1023}, std::size_t{1025}, std::size_t{1} << 20})
        {
            const auto x = random_values<T>(count, 17u + static_cast<std::uint32_t>(count));
            const auto y = random_values<T>(count, 90001u + static_cast<std::uint32_t>(count));
            std::vector<T> acc = x;
            tbccl::detail::apply_reduction(acc.data(), y.data(), count, tbccl::ReduceOp::Sum);
            for (std::size_t i = 0; i < count; ++i)
            {
                expect(acc[i] == reference_sum(x[i], y[i]), std::string(Traits<T>::name) + ": random sum mismatch, count " + std::to_string(count) + " index " + std::to_string(i));
            }
        }
        std::cout << "[PASS] test_random_vectors<" << Traits<T>::name << ">\n";
    }

    // Port zones used by no other test: World API cases take base, base+1 from 31200-31399 (the 16-bit float host test uses 31000-31199);
    // Communicator cases cycle through control ports 29240-29251 (data plane 30240-30251): everything is below the kernel's ephemeral range (32768+).
    // ctest serializes the low-precision tests against each other.
    std::uint16_t g_next_world_port = 31200;
    std::uint16_t g_next_comm_port = 29240;

    std::uint16_t take_world_port()
    {
        const std::uint16_t p = g_next_world_port;
        g_next_world_port = static_cast<std::uint16_t>(g_next_world_port + 4);
        expect(g_next_world_port < 31400, "test world port budget exhausted");
        return p;
    }

    std::uint16_t take_comm_port()
    {
        const std::uint16_t p = g_next_comm_port;
        g_next_comm_port = static_cast<std::uint16_t>(g_next_comm_port + 2);
        if (g_next_comm_port >= 29252) g_next_comm_port = 29240; // sequential cases, SO_REUSEADDR listener
        return p;
    }

    using WorldAllReduce = std::function<void(tbccl::World &, const void *, void *, std::size_t, tbccl::DataType)>;

    template <class T>
    void run_world_all_reduce(const char *label, const WorldAllReduce &fn, std::size_t count)
    {
        const auto in0 = random_values<T>(count, 5u + static_cast<std::uint32_t>(count));
        const auto in1 = random_values<T>(count, 2003u + static_cast<std::uint32_t>(count));
        auto peers = tbccl_test::make_local_peers(take_world_port(), 2);
        std::vector<std::vector<T>> out(2, std::vector<T>(count, 0));
        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;
        for (std::size_t rank = 0; rank < 2; ++rank)
        {
            threads.emplace_back(
                tbccl_test::run_rank, tbccl_test::make_options(rank, peers),
                [&, rank](tbccl::World &world) { fn(world, (rank == 0 ? in0 : in1).data(), out[rank].data(), count, Traits<T>::dt); }, std::ref(errors[rank]));
        }
        tbccl_test::join_and_check(threads, errors);
        for (std::size_t i = 0; i < count; ++i)
        {
            const T want = reference_sum(in0[i], in1[i]);
            expect(out[0][i] == want && out[1][i] == want, std::string(Traits<T>::name) + " " + label + ": World all_reduce mismatch, count " + std::to_string(count) + " index " + std::to_string(i));
        }
    }

    template <class T>
    void test_world_all_reduce()
    {
        const WorldAllReduce selected = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) { tbccl::all_reduce(w, s, r, n, t, tbccl::ReduceOp::Sum); };
        const WorldAllReduce reference = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) { tbccl::detail::all_reduce_reference(w, s, r, n, t, tbccl::ReduceOp::Sum); };
        const WorldAllReduce ring = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) { tbccl::detail::all_reduce_ring(w, s, r, n, t, tbccl::ReduceOp::Sum); };
        const WorldAllReduce pipelined = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) {
            tbccl::detail::all_reduce_pipelined(w, s, r, n, t, tbccl::ReduceOp::Sum, 1024);
        };
        for (std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{4096}, std::size_t{100001}})
        {
            run_world_all_reduce<T>("selected", selected, count);
            run_world_all_reduce<T>("reference", reference, count);
        }
        for (std::size_t count : {std::size_t{2}, std::size_t{18}, std::size_t{4096}, std::size_t{100000}})
        {
            run_world_all_reduce<T>("ring", ring, count);
            run_world_all_reduce<T>("pipelined", pipelined, count);
        }
        std::cout << "[PASS] test_world_all_reduce<" << Traits<T>::name << ">\n";
    }

    void run_pair(std::uint16_t port, const std::function<void(tbccl::Communicator &)> &rank0_fn, const std::function<void(tbccl::Communicator &)> &rank1_fn)
    {
        tbccl::CommunicatorOptions opts0;
        opts0.rank = 0;
        opts0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        tbccl::CommunicatorOptions opts1 = opts0;
        opts1.rank = 1;
        std::exception_ptr err0, err1;
        std::thread t1([&] {
            try { auto c = tbccl::Communicator::create(opts1); rank1_fn(*c); } catch (...) { err1 = std::current_exception(); }
        });
        try { auto c = tbccl::Communicator::create(opts0); rank0_fn(*c); } catch (...) { err0 = std::current_exception(); }
        t1.join();
        if (err0) std::rethrow_exception(err0);
        if (err1) std::rethrow_exception(err1);
    }

    template <class T>
    void test_communicator_all_reduce()
    {
        for (std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{4096}, std::size_t{1048576}})
        {
            for (bool in_place : {true, false})
            {
                const auto in0 = random_values<T>(count, 41u + static_cast<std::uint32_t>(count));
                const auto in1 = random_values<T>(count, 1777u + static_cast<std::uint32_t>(count));
                std::vector<T> out0 = in0, out1 = in1, dst0(count, 0), dst1(count, 0);
                auto body = [&](tbccl::Communicator &comm, std::vector<T> &inout, std::vector<T> &dst) {
                    tbccl::BufferView in_view{tbccl::MemoryKind::Host, inout.data(), inout.size(), -1};
                    tbccl::BufferView out_view{tbccl::MemoryKind::Host, dst.data(), dst.size(), -1};
                    tbccl::Work work = in_place ? comm.all_reduce(in_view, in_view, count, Traits<T>::dt, tbccl::ReduceOp::Sum)
                                                : comm.all_reduce(in_view, out_view, count, Traits<T>::dt, tbccl::ReduceOp::Sum);
                    work.wait();
                    expect(!work.has_error(), std::string(Traits<T>::name) + ": Communicator::all_reduce failed: " + work.error());
                };
                run_pair(take_comm_port(), [&](tbccl::Communicator &c) { body(c, out0, dst0); }, [&](tbccl::Communicator &c) { body(c, out1, dst1); });
                const auto &r0 = in_place ? out0 : dst0;
                const auto &r1 = in_place ? out1 : dst1;
                for (std::size_t i = 0; i < count; ++i)
                {
                    const T want = reference_sum(in0[i], in1[i]);
                    expect(r0[i] == want && r1[i] == want, std::string(Traits<T>::name) + ": Communicator all_reduce mismatch, count " + std::to_string(count) +
                                                               (in_place ? " in-place" : " out-of-place") + " index " + std::to_string(i));
                }
            }
        }
        std::cout << "[PASS] test_communicator_all_reduce<" << Traits<T>::name << ">\n";
    }

} // namespace

int main()
{
    try
    {
        test_all_pairs<std::int8_t>();
        test_all_pairs<std::uint8_t>();
        test_overflow_cases();
        test_random_vectors<std::int8_t>();
        test_random_vectors<std::uint8_t>();
        test_world_all_reduce<std::int8_t>();
        test_world_all_reduce<std::uint8_t>();
        test_communicator_all_reduce<std::int8_t>();
        test_communicator_all_reduce<std::uint8_t>();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
    std::cout << "All int8/uint8 reduction tests passed.\n";
    return 0;
}
