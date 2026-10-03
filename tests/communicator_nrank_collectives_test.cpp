// Phase 50: the reference N-rank collectives through the public Communicator, world_size 1..4 (loopback, dynamic ports): barrier, broadcast from every
// root, all_gather (rank order), all_reduce SUM for Float32/Float64/Int32/Int64 (and Int8/UInt8 where the reference reducer supports them), in-place and
// out-of-place, queued back to back; Float16/BFloat16 rejected for N>2; capability failures reported before any data moves; collective mismatches
// (kind, root, count, dtype, byte count) failing every rank instead of hanging; and an oracle comparison against the older World API.

#include "mesh_test_support.hpp"

#include <tbccl/collectives.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/tcp_world.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

namespace
{

    bool has(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

    BufferView view(void *p, std::size_t bytes, MemoryKind kind = MemoryKind::Host) { return BufferView{kind, p, bytes, 0}; }

    void wait_ok(tbccl::Work w, const std::string &what)
    {
        w.wait();
        expect(!w.has_error(), what + " failed: " + w.error());
    }

    std::vector<std::uint8_t> bytes_of(std::size_t seed, std::size_t n)
    {
        std::vector<std::uint8_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(seed * 37 + i * 11 + (i >> 9));
        return v;
    }

    // ---- barrier ----------------------------------------------------------------------------------------------------------------------------------

    void test_barrier(std::size_t world)
    {
        std::atomic<int> entered{0}, leaving_early{0};
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            for (int round = 0; round < 10; ++round)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(3 * (rank + 1)));
                entered.fetch_add(1);
                wait_ok(comm.barrier(), "barrier");
                // Nobody may leave round r before every rank entered it.
                if (entered.load() < static_cast<int>(world) * (round + 1)) leaving_early.fetch_add(1);
                wait_ok(comm.barrier(), "barrier (separates the rounds)");
            }
        });
        expect(leaving_early.load() == 0, "a rank left a barrier before every rank had entered it");
        std::cout << "[PASS] barrier world_size=" << world << "\n";
    }

    // ---- broadcast --------------------------------------------------------------------------------------------------------------------------------

    void test_broadcast(std::size_t world)
    {
        for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{17}, std::size_t{1} << 20, (std::size_t{5} << 20) + 3})
        {
            for (std::size_t root = 0; root < world; ++root)
            {
                run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                    auto buf = rank == root ? bytes_of(root + 1, bytes) : std::vector<std::uint8_t>(bytes, 0xEE);
                    wait_ok(comm.broadcast(view(buf.data(), buf.size()), root), "broadcast");
                    expect(buf == bytes_of(root + 1, bytes), "broadcast payload on rank " + std::to_string(rank) + " root " + std::to_string(root) + " bytes " + std::to_string(bytes));
                });
            }
        }
        std::cout << "[PASS] broadcast every root world_size=" << world << "\n";
    }

    // ---- all_gather -------------------------------------------------------------------------------------------------------------------------------

    void test_all_gather(std::size_t world)
    {
        for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{100}, std::size_t{1} << 20})
        {
            for (bool aliased : {false, true})
            {
                run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                    auto input = bytes_of(rank + 10, bytes);
                    std::vector<std::vector<std::uint8_t>> out(world, std::vector<std::uint8_t>(bytes, 0xEE));
                    if (aliased) out[rank] = input; // outputs[rank] is the same memory as the input
                    std::vector<BufferView> outs;
                    for (std::size_t r = 0; r < world; ++r) outs.push_back(view(out[r].data(), bytes));
                    BufferView in = aliased ? outs[rank] : view(input.data(), bytes);
                    wait_ok(comm.all_gather(in, outs), "all_gather");
                    for (std::size_t r = 0; r < world; ++r)
                        expect(out[r] == bytes_of(r + 10, bytes), "all_gather slot " + std::to_string(r) + " on rank " + std::to_string(rank));
                });
            }
        }
        std::cout << "[PASS] all_gather rank order, in-place and out-of-place, world_size=" << world << "\n";
    }

    // ---- all_reduce -------------------------------------------------------------------------------------------------------------------------------

    template <typename T> DataType dtype_of();
    template <> DataType dtype_of<float>() { return DataType::Float32; }
    template <> DataType dtype_of<double>() { return DataType::Float64; }
    template <> DataType dtype_of<std::int32_t>() { return DataType::Int32; }
    template <> DataType dtype_of<std::int64_t>() { return DataType::Int64; }
    template <> DataType dtype_of<std::int8_t>() { return DataType::Int8; }
    template <> DataType dtype_of<std::uint8_t>() { return DataType::UInt8; }

    template <typename T> std::vector<T> values(std::size_t rank, std::size_t count, std::uint32_t seed)
    {
        std::mt19937_64 gen(seed * 1000003u + rank * 7919u + count);
        std::vector<T> v(count);
        for (auto &x : v)
        {
            if constexpr (std::is_floating_point_v<T>) x = static_cast<T>(std::uniform_real_distribution<double>(-100.0, 100.0)(gen)); // not exactly representable sums
            else x = static_cast<T>(gen());
        }
        return v;
    }

    // Left fold in rank order (x0 + x1) + x2 ...: the documented, reproducible order of the reference reducer.
    template <typename T> std::vector<T> expected_sum(std::size_t world, std::size_t count, std::uint32_t seed)
    {
        std::vector<T> acc = values<T>(0, count, seed);
        for (std::size_t r = 1; r < world; ++r)
        {
            auto next = values<T>(r, count, seed);
            for (std::size_t i = 0; i < count; ++i)
            {
                if constexpr (std::is_floating_point_v<T>) acc[i] = acc[i] + next[i];
                else
                {
                    using U = std::make_unsigned_t<T>;
                    acc[i] = static_cast<T>(static_cast<U>(static_cast<U>(acc[i]) + static_cast<U>(next[i]))); // two's complement wrap, no signed overflow
                }
            }
        }
        return acc;
    }

    template <typename T> void all_reduce_case(std::size_t world, std::size_t count, bool in_place, std::uint32_t seed)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            auto mine = values<T>(rank, count, seed);
            std::vector<T> result(count, T{});
            const std::size_t bytes = count * sizeof(T);
            if (in_place)
            {
                wait_ok(comm.all_reduce(view(mine.data(), bytes), view(mine.data(), bytes), count, dtype_of<T>(), ReduceOp::Sum), "in-place all_reduce");
                result = mine;
            }
            else
            {
                auto untouched = mine;
                wait_ok(comm.all_reduce(view(untouched.data(), bytes), view(result.data(), bytes), count, dtype_of<T>(), ReduceOp::Sum), "out-of-place all_reduce");
                expect(untouched == mine || bytes == 0, "the input of an out-of-place all_reduce is not modified");
            }
            const auto want = expected_sum<T>(world, count, seed);
            expect(count == 0 || std::memcmp(result.data(), want.data(), bytes) == 0,
                   std::string("all_reduce bits differ from the rank-ordered fold: dtype=") + tbccl::datatype_label(dtype_of<T>()) + " count=" + std::to_string(count) +
                       " world=" + std::to_string(world) + (in_place ? " in-place" : " out-of-place") + " rank=" + std::to_string(rank));
        });
    }

    template <typename T> void all_reduce_type(std::size_t world)
    {
        for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{17}, std::size_t{4096}, std::size_t{300000}})
            for (bool in_place : {true, false}) all_reduce_case<T>(world, count, in_place, 5);
    }

    void test_all_reduce(std::size_t world)
    {
        all_reduce_type<float>(world);
        all_reduce_type<double>(world);
        all_reduce_type<std::int32_t>(world);
        all_reduce_type<std::int64_t>(world);
        // Optional in Phase 50: modulo-256 SUM is associative, so it falls out of the same reducer.
        all_reduce_type<std::int8_t>(world);
        all_reduce_type<std::uint8_t>(world);
        std::cout << "[PASS] all_reduce Float32/Float64/Int32/Int64/Int8/UInt8, in-place and out-of-place, world_size=" << world << "\n";
    }

    void test_low_precision_rejected_beyond_two()
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
                        threw = has(e.what(), "unsupported") && has(e.what(), "N>2 reduction semantics are not defined") && has(e.what(), tbccl::datatype_label(dt));
                    }
                    expect(threw, std::string("all_reduce ") + tbccl::datatype_label(dt) + " across " + std::to_string(world) + " ranks is rejected explicitly");
                }
                expect(!comm.failed(), "the rejection happens before any communication and poisons nothing");
                wait_ok(comm.barrier(), "barrier after the rejection");
            });
        }
        // The two-operand semantics of Phase 49 are untouched.
        run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint16_t> half(8, rank == 0 ? 0x3c00 : 0x4000); // 1.0 + 2.0
            wait_ok(comm.all_reduce(view(half.data(), 16), view(half.data(), 16), 8, DataType::Float16, ReduceOp::Sum), "N=2 Float16");
            expect(half[0] == 0x4200, "1.0 + 2.0 == 3.0 in half precision at N=2");
        });
        std::cout << "[PASS] Float16/BFloat16 SUM rejected for N>2 (clear message, nothing poisoned); N=2 unchanged\n";
    }

    // Many collectives of different kinds posted back to back without waiting: the executor runs them in posting order on every rank.
    void test_queued_sequence(std::size_t world)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t n = 1000;
            std::vector<std::vector<std::int32_t>> sums(6, std::vector<std::int32_t>(n));
            std::vector<std::vector<std::uint8_t>> bcast(6);
            std::vector<tbccl::Work> works;
            for (int i = 0; i < 6; ++i)
            {
                works.push_back(comm.barrier());
                bcast[i] = rank == static_cast<std::size_t>(i) % world ? bytes_of(i, 333) : std::vector<std::uint8_t>(333, 0);
                works.push_back(comm.broadcast(view(bcast[i].data(), 333), static_cast<std::size_t>(i) % world));
                for (std::size_t k = 0; k < n; ++k) sums[i][k] = static_cast<std::int32_t>(rank * 1000 + i * 10 + k);
                works.push_back(comm.all_reduce(view(sums[i].data(), n * 4), view(sums[i].data(), n * 4), n, DataType::Int32, ReduceOp::Sum));
            }
            for (auto &w : works) wait_ok(w, "queued collective");
            for (int i = 0; i < 6; ++i)
            {
                expect(bcast[i] == bytes_of(i, 333), "queued broadcast " + std::to_string(i));
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::int32_t want = 0;
                    for (std::size_t r = 0; r < world; ++r) want += static_cast<std::int32_t>(r * 1000 + i * 10 + k);
                    expect(sums[i][k] == want, "queued all_reduce " + std::to_string(i));
                }
            }
        });
        std::cout << "[PASS] 18 queued collectives run in posting order world_size=" << world << "\n";
    }

    // ---- capability failures: before any data, communicator stays usable -----------------------------------------------------------------------------

    void test_unsupported_memory_kind_fails_everywhere()
    {
        // Rank 1 passes a Cuda-kind buffer; this test binary has no CUDA provider registered, so ONLY rank 1 cannot run the collective.
        // No rank may hang waiting for it, and no payload may move.
        run_world(3, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::int32_t> data(256, static_cast<std::int32_t>(rank + 1));
            const auto before = data;
            std::int32_t fake_device = 0;
            const BufferView v = rank == 1 ? view(&fake_device, 1024, MemoryKind::Cuda) : view(data.data(), 1024);
            auto w = comm.all_reduce(v, v, 256, DataType::Int32, ReduceOp::Sum);
            w.wait();
            expect(w.has_error() && has(w.error(), "unsupported") && has(w.error(), "rank 1") && has(w.error(), "cuda"), "every rank gets the unsupported error naming rank 1: " + w.error());
            if (rank != 1) expect(data == before, "no payload moved: the buffer is unchanged");
            expect(!comm.failed(), "an unsupported collective does not poison the communicator");
            // and it is usable again
            std::vector<std::int32_t> again(8, static_cast<std::int32_t>(rank + 1));
            wait_ok(comm.all_reduce(view(again.data(), 32), view(again.data(), 32), 8, DataType::Int32, ReduceOp::Sum), "all_reduce after the rejection");
            expect(again[0] == 6, "1+2+3 after the rejection");
        });
        std::cout << "[PASS] a rank that cannot run a collective fails it on every rank before any data moves; the communicator stays usable\n";
    }

    void test_int8_on_metal_shared_rank_rejected()
    {
        // MetalShared uses the host provider on every platform, but its reducer only covers the original four element types.
        run_world(3, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::int8_t> data(64, static_cast<std::int8_t>(rank + 1));
            const BufferView v = view(data.data(), 64, rank == 2 ? MemoryKind::MetalShared : MemoryKind::Host);
            auto w = comm.all_reduce(v, v, 64, DataType::Int8, ReduceOp::Sum);
            w.wait();
            expect(w.has_error() && has(w.error(), "unsupported") && has(w.error(), "rank 2") && has(w.error(), "metal-shared") && has(w.error(), "int8"), "Int8 on a MetalShared rank is rejected by every rank: " + w.error());
            expect(data[0] == static_cast<std::int8_t>(rank + 1), "no payload moved");
        });
        std::cout << "[PASS] heterogeneous capability: Int8 all_reduce with a MetalShared rank is rejected before the data phase\n";
    }

    // ---- mismatches --------------------------------------------------------------------------------------------------------------------------------

    // Runs `body` on every rank, which returns the Work (or throws). Every rank must end with a mismatch error, promptly, and a poisoned communicator.
    void expect_mismatch(std::size_t world, const std::string &label, const std::string &needle,
                         const std::function<tbccl::Work(std::size_t, tbccl::Communicator &)> &body)
    {
        const auto start = std::chrono::steady_clock::now();
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            auto w = body(rank, comm);
            w.wait();
            expect(w.has_error() && has(w.error(), "protocol_mismatch") && has(w.error(), needle),
                   label + ": rank " + std::to_string(rank) + " got '" + w.error() + "'");
            expect(comm.failed(), label + ": the communicator is poisoned on rank " + std::to_string(rank));
            bool threw = false;
            try
            {
                comm.barrier();
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, label + ": a poisoned communicator rejects new collectives");
        });
        expect(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), label + ": no hang");
        std::cout << "[PASS] mismatch (" << label << ") world_size=" << world << "\n";
    }

    void test_mismatches(std::size_t world)
    {
        std::vector<std::int32_t> scratch(1024, 1);
        expect_mismatch(world, "wrong collective kind", "called", [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::int32_t> v(64, 1);
            // rank 1 calls broadcast while everyone else calls all_reduce
            if (rank == 1) return comm.broadcast(view(v.data(), 256), 0);
            return comm.all_reduce(view(v.data(), 256), view(v.data(), 256), 64, DataType::Int32, ReduceOp::Sum);
        });
        expect_mismatch(world, "wrong root", "root differs", [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> v(100, 1);
            return comm.broadcast(view(v.data(), 100), rank == world - 1 ? 1 : 0);
        });
        expect_mismatch(world, "wrong element count", "element count differs", [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::int32_t> v(64, 1);
            const std::size_t count = rank == 2 ? 32 : 64; // the buffer is big enough for either
            return comm.all_reduce(view(v.data(), 256), view(v.data(), 256), count, DataType::Int32, ReduceOp::Sum);
        });
        expect_mismatch(world, "wrong dtype", "dtype differs", [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::int32_t> v(64, 1);
            return comm.all_reduce(view(v.data(), 256), view(v.data(), 256), 64, rank == 1 ? DataType::Float32 : DataType::Int32, ReduceOp::Sum); // same size, different type
        });
        expect_mismatch(world, "wrong broadcast byte count", "byte count differs", [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> v(100, 1);
            return comm.broadcast(view(v.data(), rank == 2 ? 50 : 100), 0);
        });
        expect_mismatch(world, "wrong all_gather byte count", "byte count differs", [&](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t n = rank == 1 ? 8 : 16;
            std::vector<std::uint8_t> in(n, 1);
            std::vector<std::vector<std::uint8_t>> out(world, std::vector<std::uint8_t>(n));
            std::vector<BufferView> outs;
            for (auto &o : out) outs.push_back(view(o.data(), n));
            return comm.all_gather(view(in.data(), n), outs);
        });
    }

    // ---- the old World API as an oracle -----------------------------------------------------------------------------------------------------------

    std::vector<std::uint16_t> free_ports(std::size_t n)
    {
        std::vector<std::unique_ptr<tbccl::Listener>> hold;
        std::vector<std::uint16_t> ports;
        for (std::size_t i = 0; i < n; ++i)
        {
            hold.push_back(tbccl::tcp_listen("127.0.0.1", 0, {}));
            ports.push_back(hold.back()->local_port());
        }
        return ports;
    }

    // Same inputs through the synchronous World reference collectives and through the Communicator; the outputs must agree byte for byte.
    void test_against_world_oracle(std::size_t world)
    {
        const std::size_t count = 5000;
        auto ports = free_ports(world);
        std::vector<std::vector<std::int32_t>> world_sum(world), world_gather(world);
        std::vector<std::vector<std::uint8_t>> world_bcast(world);
        {
            std::vector<std::thread> threads;
            std::vector<std::string> errors(world);
            for (std::size_t r = 0; r < world; ++r)
            {
                threads.emplace_back([&, r] {
                    try
                    {
                        tbccl::TcpWorldOptions o;
                        o.rank = r;
                        for (auto p : ports) o.peers.push_back({"127.0.0.1", p});
                        auto w = tbccl::create_tcp_world(o);
                        auto mine = values<std::int32_t>(r, count, 9);
                        world_sum[r].resize(count);
                        tbccl::all_reduce(*w, mine.data(), world_sum[r].data(), count, DataType::Int32, ReduceOp::Sum);
                        world_gather[r].resize(count * world);
                        tbccl::all_gather(*w, mine.data(), world_gather[r].data(), count * 4);
                        world_bcast[r] = r == world - 1 ? bytes_of(3, 4096) : std::vector<std::uint8_t>(4096, 0);
                        tbccl::broadcast(*w, world_bcast[r].data(), 4096, world - 1);
                        tbccl::barrier(*w);
                    }
                    catch (const std::exception &e)
                    {
                        errors[r] = e.what();
                    }
                });
            }
            for (auto &t : threads) t.join();
            for (std::size_t r = 0; r < world; ++r) expect(errors[r].empty(), "World oracle failed on rank " + std::to_string(r) + ": " + errors[r]);
        }
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            auto mine = values<std::int32_t>(rank, count, 9);
            auto sum = mine;
            wait_ok(comm.all_reduce(view(sum.data(), count * 4), view(sum.data(), count * 4), count, DataType::Int32, ReduceOp::Sum), "all_reduce");
            expect(sum == world_sum[rank], "all_reduce matches the World oracle on rank " + std::to_string(rank));
            std::vector<std::int32_t> gathered(count * world);
            std::vector<BufferView> outs;
            for (std::size_t r = 0; r < world; ++r) outs.push_back(view(gathered.data() + r * count, count * 4));
            wait_ok(comm.all_gather(view(mine.data(), count * 4), outs), "all_gather");
            expect(gathered == world_gather[rank], "all_gather matches the World oracle on rank " + std::to_string(rank));
            auto b = rank == world - 1 ? bytes_of(3, 4096) : std::vector<std::uint8_t>(4096, 0);
            wait_ok(comm.broadcast(view(b.data(), 4096), world - 1), "broadcast");
            expect(b == world_bcast[rank], "broadcast matches the World oracle on rank " + std::to_string(rank));
        });
        std::cout << "[PASS] Communicator == World reference collectives (all_reduce, all_gather, broadcast) world_size=" << world << "\n";
    }

} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(240));
        std::cerr << "[FAIL] watchdog: a collective test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        for (std::size_t world = 1; world <= 4; ++world)
        {
            test_barrier(world);
            test_broadcast(world);
            test_all_gather(world);
            test_all_reduce(world);
            test_queued_sequence(world);
        }
        test_low_precision_rejected_beyond_two();
        test_unsupported_memory_kind_fails_everywhere();
        test_int8_on_metal_shared_rank_rejected();
        test_mismatches(3);
        test_mismatches(4);
        test_against_world_oracle(3);
        test_against_world_oracle(4);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All N-rank collective tests passed.\n";
    return 0;
}
