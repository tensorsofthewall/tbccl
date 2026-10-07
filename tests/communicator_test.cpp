// Correctness tests for the public tbccl::Communicator API, using
// Host-memory BufferViews only (MemoryKind::MetalShared reuses the exact
// same code path -- see docs/framework_integration_architecture.md Section 5
// -- so these tests also exercise that path by construction). Real
// TB4/CUDA/Metal coverage lives in separate test files.

#include <tbccl/communicator.hpp>
#include <tbccl/tcp.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

void expect(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error("assertion failed: " + message);
}

constexpr std::uint16_t kBasePort = 29100;

// Runs `rank0_fn`/`rank1_fn` concurrently against a freshly-created pair
// of communicators over real TCP loopback, joining both before returning.
void run_pair(
    std::uint16_t port,
    const std::function<void(tbccl::Communicator &)> &rank0_fn,
    const std::function<void(tbccl::Communicator &)> &rank1_fn)
{
    tbccl::CommunicatorOptions opts0;
    opts0.rank = 0;
    opts0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};

    tbccl::CommunicatorOptions opts1 = opts0;
    opts1.rank = 1;

    std::exception_ptr err0, err1;
    std::thread t1([&] {
        try
        {
            auto comm1 = tbccl::Communicator::create(opts1);
            rank1_fn(*comm1);
        }
        catch (...)
        {
            err1 = std::current_exception();
        }
    });

    try
    {
        auto comm0 = tbccl::Communicator::create(opts0);
        rank0_fn(*comm0);
    }
    catch (...)
    {
        err0 = std::current_exception();
    }
    t1.join();

    if (err0) std::rethrow_exception(err0);
    if (err1) std::rethrow_exception(err1);
}

std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint8_t seed)
{
    std::vector<std::uint8_t> v(bytes);
    for (std::size_t i = 0; i < bytes; ++i) v[i] = static_cast<std::uint8_t>((i * 131u + seed) & 0xffu);
    return v;
}

// ---------------------------------------------------------------------
// P2P correctness
// ---------------------------------------------------------------------

// DataType has no 1-byte element type, so byte-exact coverage uses
// element counts (including non-power-of-two counts) over Int32 rather
// than raw byte sizes -- BufferView.bytes itself stays exact regardless.
void test_p2p_roundtrip()
{
    const std::vector<std::size_t> element_counts = {1, 1024, 1024 * 256, 3333};
    std::uint16_t port = kBasePort;
    for (std::size_t count : element_counts)
    {
        const std::size_t bytes = count * sizeof(std::int32_t);
        auto src = pattern(bytes, 0x5A);
        std::vector<std::uint8_t> dst(bytes, 0xCC);

        run_pair(
            port,
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, src.data(), bytes, 0};
                auto work = comm.send(view, count, tbccl::DataType::Int32, 1);
                work.wait();
                expect(!work.has_error(), "sender must not error: " + work.error());
            },
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, dst.data(), bytes, 0};
                auto work = comm.recv(view, count, tbccl::DataType::Int32, 0);
                work.wait();
                expect(!work.has_error(), "receiver must not error: " + work.error());
            });

        expect(std::memcmp(src.data(), dst.data(), bytes) == 0, "P2P roundtrip must reproduce exact bytes, count=" + std::to_string(count));
        port += 2;
    }
    std::cout << "[PASS] test_p2p_roundtrip\n";
}

void test_p2p_repeated_reuse()
{
    const std::size_t count = 4096;
    const std::size_t bytes = count * sizeof(std::int32_t);
    run_pair(
        kBasePort + 100,
        [&](tbccl::Communicator &comm) {
            for (int round = 0; round < 5; ++round)
            {
                auto src = pattern(bytes, static_cast<std::uint8_t>(round));
                tbccl::BufferView view{tbccl::MemoryKind::Host, src.data(), bytes, 0};
                auto work = comm.send(view, count, tbccl::DataType::Int32, 1);
                work.wait();
                expect(!work.has_error(), "round " + std::to_string(round) + " sender must not error");
            }
        },
        [&](tbccl::Communicator &comm) {
            for (int round = 0; round < 5; ++round)
            {
                std::vector<std::uint8_t> dst(bytes, 0);
                tbccl::BufferView view{tbccl::MemoryKind::Host, dst.data(), bytes, 0};
                auto work = comm.recv(view, count, tbccl::DataType::Int32, 0);
                work.wait();
                expect(!work.has_error(), "round " + std::to_string(round) + " receiver must not error");
                auto expected = pattern(bytes, static_cast<std::uint8_t>(round));
                expect(std::memcmp(expected.data(), dst.data(), bytes) == 0, "round " + std::to_string(round) + " data mismatch");
            }
        });
    std::cout << "[PASS] test_p2p_repeated_reuse\n";
}

void test_p2p_multiple_outstanding()
{
    // 3 queued sends, waited in submission order (TensorCommWorker
    // processes requests one at a time in submission order -- see
    // async_transfer.hpp); this test proves 3 outstanding Work objects
    // from one Communicator all complete correctly without
    // cross-contamination.
    const std::size_t count = 1024;
    const std::size_t bytes = count * sizeof(std::int32_t);
    run_pair(
        kBasePort + 110,
        [&](tbccl::Communicator &comm) {
            std::vector<std::vector<std::uint8_t>> buffers;
            std::vector<tbccl::Work> works;
            for (int i = 0; i < 3; ++i)
            {
                buffers.push_back(pattern(bytes, static_cast<std::uint8_t>(i + 1)));
                tbccl::BufferView view{tbccl::MemoryKind::Host, buffers.back().data(), bytes, 0};
                works.push_back(comm.send(view, count, tbccl::DataType::Int32, 1));
            }
            for (auto &w : works)
            {
                w.wait();
                expect(!w.has_error(), "multi-outstanding sender must not error");
            }
        },
        [&](tbccl::Communicator &comm) {
            std::vector<std::vector<std::uint8_t>> buffers(3, std::vector<std::uint8_t>(bytes, 0));
            std::vector<tbccl::Work> works;
            for (int i = 0; i < 3; ++i)
            {
                tbccl::BufferView view{tbccl::MemoryKind::Host, buffers[static_cast<std::size_t>(i)].data(), bytes, 0};
                works.push_back(comm.recv(view, count, tbccl::DataType::Int32, 0));
            }
            for (auto &w : works)
            {
                w.wait();
                expect(!w.has_error(), "multi-outstanding receiver must not error");
            }
            for (int i = 0; i < 3; ++i)
            {
                auto expected = pattern(bytes, static_cast<std::uint8_t>(i + 1));
                expect(std::memcmp(expected.data(), buffers[static_cast<std::size_t>(i)].data(), bytes) == 0,
                       "multi-outstanding buffer " + std::to_string(i) + " must not cross-contaminate");
            }
        });
    std::cout << "[PASS] test_p2p_multiple_outstanding\n";
}

// ---------------------------------------------------------------------
// Host N=2 AllReduce
// ---------------------------------------------------------------------

void test_all_reduce_float32()
{
    const std::vector<std::size_t> counts = {1, 7, 1024, 1024 * 256};
    std::uint16_t port = kBasePort + 200;
    for (std::size_t count : counts)
    {
        std::vector<float> rank0_data(count), rank1_data(count), expected(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            rank0_data[i] = static_cast<float>(i) * 1.5f + 1.0f;
            rank1_data[i] = static_cast<float>(i) * 0.5f - 2.0f;
            expected[i] = rank0_data[i] + rank1_data[i];
        }

        run_pair(
            port,
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, rank0_data.data(), count * sizeof(float), 0};
                auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                work.wait();
                expect(!work.has_error(), "rank0 all_reduce must not error: " + work.error());
            },
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, rank1_data.data(), count * sizeof(float), 0};
                auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                work.wait();
                expect(!work.has_error(), "rank1 all_reduce must not error: " + work.error());
            });

        for (std::size_t i = 0; i < count; ++i)
        {
            expect(rank0_data[i] == expected[i], "rank0 result mismatch at count=" + std::to_string(count));
            expect(rank1_data[i] == expected[i], "rank1 result mismatch at count=" + std::to_string(count));
        }
        port += 2;
    }
    std::cout << "[PASS] test_all_reduce_float32\n";
}

void test_all_reduce_repeated_rounds()
{
    const std::size_t count = 4096;
    run_pair(
        kBasePort + 300,
        [&](tbccl::Communicator &comm) {
            for (int round = 0; round < 4; ++round)
            {
                std::vector<float> data(count, static_cast<float>(round + 1));
                tbccl::BufferView view{tbccl::MemoryKind::Host, data.data(), count * sizeof(float), 0};
                auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                work.wait();
                expect(!work.has_error(), "round " + std::to_string(round) + " rank0 must not error");
                for (float v : data) expect(v == static_cast<float>((round + 1) * 2), "round " + std::to_string(round) + " rank0 value mismatch");
            }
        },
        [&](tbccl::Communicator &comm) {
            for (int round = 0; round < 4; ++round)
            {
                std::vector<float> data(count, static_cast<float>(round + 1));
                tbccl::BufferView view{tbccl::MemoryKind::Host, data.data(), count * sizeof(float), 0};
                auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                work.wait();
                expect(!work.has_error(), "round " + std::to_string(round) + " rank1 must not error");
            }
        });
    std::cout << "[PASS] test_all_reduce_repeated_rounds\n";
}

// ---------------------------------------------------------------------
// Error tests
// ---------------------------------------------------------------------

void test_errors_invalid_buffer()
{
    run_pair(
        kBasePort + 400,
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Host, nullptr, 0, 0};
            bool threw = false;
            try
            {
                comm.send(view, 10, tbccl::DataType::Float32, 1);
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, "send() with null buffer and nonzero count must throw");

            float x = 1.0f;
            tbccl::BufferView too_small{tbccl::MemoryKind::Host, &x, sizeof(float), 0};
            threw = false;
            try
            {
                comm.send(too_small, 2, tbccl::DataType::Float32, 1); // 2*4=8 > 4 bytes
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, "send() with count*datatype_size > bytes must throw");

            threw = false;
            try
            {
                comm.send(too_small, 1, tbccl::DataType::Float32, 7); // invalid peer
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, "send() to an invalid peer index must throw");
        },
        [&](tbccl::Communicator &) {});
    std::cout << "[PASS] test_errors_invalid_buffer\n";
}

void test_errors_unsupported_world_size()
{
    // world_size 1 is a real world (and opens no socket); a world beyond the full-mesh limit is refused up front.
    tbccl::CommunicatorOptions opts;
    opts.rank = 0;
    opts.peers = {{"127.0.0.1", kBasePort + 500}};
    {
        auto comm = tbccl::Communicator::create(opts);
        expect(comm->world_size() == 1 && comm->rank() == 0, "world_size 1 communicator");
        auto probe = tbccl::tcp_listen("127.0.0.1", kBasePort + 500, {}); // would throw if the communicator had bound it
        auto probe_data = tbccl::tcp_listen("127.0.0.1", kBasePort + 1500, {});
    }
    tbccl::CommunicatorOptions too_big;
    for (std::size_t r = 0; r < tbccl::kMaxFullMeshWorldSize + 1; ++r)
        too_big.rank_directory.entries.push_back({r, {"127.0.0.1", static_cast<std::uint16_t>(kBasePort + 520 + 2 * r)}, {"127.0.0.1", static_cast<std::uint16_t>(kBasePort + 521 + 2 * r)}});
    bool threw = false;
    try
    {
        tbccl::Communicator::create(too_big);
    }
    catch (const std::exception &e)
    {
        threw = std::string(e.what()).find("unsupported") != std::string::npos;
    }
    expect(threw, "Communicator::create() beyond the full-mesh limit must throw unsupported");
    std::cout << "[PASS] test_errors_unsupported_world_size\n";
}

void test_errors_unsupported_reduce_op()
{
    run_pair(
        kBasePort + 600,
        [&](tbccl::Communicator &comm) {
            float x = 1.0f;
            tbccl::BufferView view{tbccl::MemoryKind::Host, &x, sizeof(float), 0};
            bool threw = false;
            try
            {
                comm.all_reduce(view, view, 1, tbccl::DataType::Float32, tbccl::ReduceOp::Max);
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, "all_reduce with ReduceOp::Max must throw (Sum-only this phase)");
        },
        [&](tbccl::Communicator &) {});
    std::cout << "[PASS] test_errors_unsupported_reduce_op\n";
}

// ---------------------------------------------------------------------
// Communicator lifecycle
// ---------------------------------------------------------------------

void test_communicator_zero_operations()
{
    run_pair(kBasePort + 700, [&](tbccl::Communicator &) {}, [&](tbccl::Communicator &) {});
    std::cout << "[PASS] test_communicator_zero_operations\n";
}

void test_two_independent_communicators()
{
    // Two fully independent communicator pairs, distinct ports, running
    // concurrently -- proves no global singleton state (Part 151/152).
    std::thread t([&] {
        run_pair(
            kBasePort + 800,
            [&](tbccl::Communicator &comm) {
                float x = 10.0f;
                tbccl::BufferView view{tbccl::MemoryKind::Host, &x, sizeof(float), 0};
                auto work = comm.all_reduce(view, view, 1, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                work.wait();
                expect(!work.has_error(), "comm A rank0 must not error");
                expect(x == 15.0f, "comm A result mismatch");
            },
            [&](tbccl::Communicator &comm) {
                float x = 5.0f;
                tbccl::BufferView view{tbccl::MemoryKind::Host, &x, sizeof(float), 0};
                auto work = comm.all_reduce(view, view, 1, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                work.wait();
                expect(!work.has_error(), "comm A rank1 must not error");
            });
    });

    run_pair(
        kBasePort + 810,
        [&](tbccl::Communicator &comm) {
            float x = 100.0f;
            tbccl::BufferView view{tbccl::MemoryKind::Host, &x, sizeof(float), 0};
            auto work = comm.all_reduce(view, view, 1, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            work.wait();
            expect(!work.has_error(), "comm B rank0 must not error");
            expect(x == 150.0f, "comm B result mismatch");
        },
        [&](tbccl::Communicator &comm) {
            float x = 50.0f;
            tbccl::BufferView view{tbccl::MemoryKind::Host, &x, sizeof(float), 0};
            auto work = comm.all_reduce(view, view, 1, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            work.wait();
            expect(!work.has_error(), "comm B rank1 must not error");
        });

    t.join();
    std::cout << "[PASS] test_two_independent_communicators\n";
}

void test_repeated_create_destroy()
{
    for (int i = 0; i < 3; ++i)
    {
        run_pair(static_cast<std::uint16_t>(kBasePort + 900 + i * 10), [&](tbccl::Communicator &) {}, [&](tbccl::Communicator &) {});
    }
    std::cout << "[PASS] test_repeated_create_destroy\n";
}

} // namespace

int main()
{
    try
    {
        test_p2p_roundtrip();
        test_p2p_repeated_reuse();
        test_p2p_multiple_outstanding();
        test_all_reduce_float32();
        test_all_reduce_repeated_rounds();
        test_errors_invalid_buffer();
        test_errors_unsupported_world_size();
        test_errors_unsupported_reduce_op();
        test_communicator_zero_operations();
        test_two_independent_communicators();
        test_repeated_create_destroy();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All communicator tests passed.\n";
    return 0;
}
