// Tests for the async tensor-transfer portable async tensor-transfer
// substrate (async_transfer.hpp): ChunkPlan correctness, StagingPool
// race-safety, and TensorCommWorker
// lifecycle/ordering/error-propagation, all over real local TCP
// loopback connections via TcpTransport -- no GPU involved (Part Q:
// host path first, validates the substrate itself).

#include <tbccl/async_transfer.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    constexpr std::uint16_t kBasePort = 28720;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    // A deterministic byte pattern that varies by both transfer ID and
    // chunk index, so stale-slot-reuse / chunk-ordering bugs produce a
    // detectable mismatch rather than silently passing.
    std::uint8_t pattern_byte(std::uint64_t transfer_id, std::size_t i)
    {
        const std::uint64_t v =
            static_cast<std::uint64_t>(i) * 2654435761u + transfer_id * 97u;
        return static_cast<std::uint8_t>(v & 0xffu);
    }

    // Minimal host-memory AsyncMemoryBackend for testing: source/
    // destination are plain std::vector<uint8_t> the test owns.
    // stage_source_chunk/commit_destination_chunk are synchronous
    // memcpys -- the explicit allowance ("host backend can complete
    // readiness immediately").
    class VectorAsyncBackend : public tbccl::AsyncMemoryBackend
    {
    public:
        explicit VectorAsyncBackend(std::vector<std::uint8_t> &buffer)
            : buffer_(buffer)
        {
        }

        void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
        {
            std::memcpy(staging, buffer_.data() + chunk.offset, chunk.size);
        }

        void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
        {
            std::memcpy(buffer_.data() + chunk.offset, staging, chunk.size);
        }

        // Plain host memory is directly transport-accessible --
        // exercises TensorCommWorker's direct path (chunk_hint == 0) in
        // these tests, not just the staged path via
        // stage_source_chunk()/commit_destination_chunk() above.
        bool supports_direct_transport_access() const noexcept override { return true; }
        const void *direct_source_data() const noexcept override { return buffer_.data(); }
        void *direct_destination_data() noexcept override { return buffer_.data(); }

    private:
        std::vector<std::uint8_t> &buffer_;
    };

    // A backend whose stage/commit calls always throw, for error-
    // propagation tests.
    class FailingAsyncBackend : public tbccl::AsyncMemoryBackend
    {
    public:
        void stage_source_chunk(const tbccl::Chunk &, void *) override
        {
            throw std::runtime_error("synthetic source staging failure");
        }
        void commit_destination_chunk(const tbccl::Chunk &, const void *) override
        {
            throw std::runtime_error("synthetic destination commit failure");
        }
    };

    // -------------------------------------------------------------
    // ChunkPlan
    // -------------------------------------------------------------

    void test_chunk_plan_cases()
    {
        using tbccl::plan_chunks;

        expect(plan_chunks(0, 100).empty(), "zero bytes should plan to zero chunks");

        {
            auto chunks = plan_chunks(1, 100);
            expect(chunks.size() == 1 && chunks[0].size == 1,
                   "1-byte payload should be a single chunk");
        }
        {
            // chunk - 1
            auto chunks = plan_chunks(99, 100);
            expect(chunks.size() == 1 && chunks[0].size == 99,
                   "payload smaller than chunk_hint should be one chunk");
        }
        {
            // exact chunk
            auto chunks = plan_chunks(100, 100);
            expect(chunks.size() == 1 && chunks[0].size == 100,
                   "payload exactly one chunk should be one chunk");
        }
        {
            // chunk + 1
            auto chunks = plan_chunks(101, 100);
            expect(chunks.size() == 2 && chunks[0].size == 100 && chunks[1].size == 1,
                   "payload one byte over a chunk should split 100+1");
        }
        {
            // exact 2 chunks
            auto chunks = plan_chunks(200, 100);
            expect(chunks.size() == 2 && chunks[0].size == 100 && chunks[1].size == 100,
                   "exact 2-chunk payload should split evenly");
        }
        {
            // non-divisible, large multi-chunk
            auto chunks = plan_chunks(250, 100);
            expect(chunks.size() == 3 &&
                       chunks[0].size == 100 && chunks[1].size == 100 && chunks[2].size == 50,
                   "non-divisible payload should end with a smaller final chunk");
            expect(chunks[0].offset == 0 && chunks[1].offset == 100 && chunks[2].offset == 200,
                   "chunk offsets should be contiguous with no gap/overlap");
        }
        {
            // chunk_hint == 0 means "whole payload"
            auto chunks = plan_chunks(4096, 0);
            expect(chunks.size() == 1 && chunks[0].size == 4096,
                   "chunk_hint 0 should mean one whole-payload chunk");
        }
        {
            // alignment never drops/duplicates bytes
            auto chunks = plan_chunks(97, 10, 8);
            std::size_t total = 0;
            std::size_t expected_offset = 0;
            for (auto &c : chunks)
            {
                expect(c.offset == expected_offset, "aligned chunks must remain contiguous");
                total += c.size;
                expected_offset += c.size;
            }
            expect(total == 97, "aligned chunk plan must cover every byte exactly once");
        }

        std::cout << "[PASS] test_chunk_plan_cases\n";
    }

    // -------------------------------------------------------------
    // StagingPool
    // -------------------------------------------------------------

    void test_staging_pool_basic_lifecycle()
    {
        tbccl::StagingPool pool(64, 2);

        expect(pool.depth() == 2, "pool depth should match constructor argument");
        expect(pool.slot_bytes() == 64, "pool slot_bytes should match constructor argument");

        const std::size_t a = pool.acquire();
        const std::size_t b = pool.acquire();
        expect(a != b, "two acquires on a depth-2 pool should return distinct slots");

        pool.release(a);
        const std::size_t c = pool.acquire();
        expect(c == a, "release should make a slot immediately reusable");

        pool.release(b);
        pool.release(c);

        std::cout << "[PASS] test_staging_pool_basic_lifecycle\n";
    }

    // acquire() on a fully-depleted pool blocks until release(),
    // exercised across two threads (Part M item 54: race-safe).
    void test_staging_pool_acquire_blocks_until_release()
    {
        tbccl::StagingPool pool(16, 1);

        const std::size_t first = pool.acquire();

        std::atomic<bool> acquired_second{false};
        std::thread waiter(
            [&]()
            {
                pool.acquire();
                acquired_second.store(true);
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        expect(!acquired_second.load(),
               "acquire() on a depleted pool must block, not return early");

        pool.release(first);
        waiter.join();

        expect(acquired_second.load(),
               "acquire() must unblock once the only slot is released");

        std::cout << "[PASS] test_staging_pool_acquire_blocks_until_release\n";
    }

    // -------------------------------------------------------------
    // TensorCommWorker: host<->host over real TCP loopback
    // -------------------------------------------------------------

    struct Endpoint
    {
        std::unique_ptr<tbccl::Transport> transport;
        tbccl::TensorCommWorker worker;

        explicit Endpoint(std::unique_ptr<tbccl::Connection> connection,
                           std::size_t pipeline_depth = 2)
            : transport(std::make_unique<tbccl::TcpTransport>(std::move(connection))),
              worker(pipeline_depth, /*queue_depth=*/4)
        {
        }
    };

    // Runs one send/recv pair for `bytes`, with independently
    // deterministic-but-verifiable source content, and returns
    // whether the destination content matches exactly.
    bool run_one_transfer(
        std::uint16_t port,
        std::uint64_t transfer_id,
        std::size_t bytes,
        std::size_t chunk_hint,
        std::size_t pipeline_depth)
    {
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> source(bytes);
        for (std::size_t i = 0; i < bytes; ++i)
        {
            source[i] = pattern_byte(transfer_id, i);
        }
        std::vector<std::uint8_t> destination(bytes, 0);

        VectorAsyncBackend source_backend(source);
        VectorAsyncBackend dest_backend(destination);

        bool receiver_ok = true;
        std::thread receiver(
            [&]()
            {
                Endpoint server(listener->accept(), pipeline_depth);

                tbccl::TransferRequest request;
                request.transfer_id = transfer_id;
                request.direction = tbccl::TransferDirection::Recv;
                request.backend = &dest_backend;
                request.transport = server.transport.get();
                request.total_bytes = bytes;
                request.chunk_hint = chunk_hint;

                auto work = server.worker.enqueue(request);
                work.wait();
                receiver_ok = !work.has_error();
            });

        Endpoint client(tbccl::tcp_connect("127.0.0.1", port, {}), pipeline_depth);

        tbccl::TransferRequest request;
        request.transfer_id = transfer_id;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = client.transport.get();
        request.total_bytes = bytes;
        request.chunk_hint = chunk_hint;

        auto work = client.worker.enqueue(request);
        work.wait();

        receiver.join();

        if (work.has_error() || !receiver_ok)
        {
            return false;
        }

        return destination == source;
    }

    void test_host_roundtrip_sizes_and_depths()
    {
        struct Case
        {
            std::size_t bytes;
            std::size_t chunk_hint;
            std::size_t depth;
        };

        const std::vector<Case> cases = {
            {1, 64, 1},           // 1 byte
            {63, 64, 2},          // chunk - 1
            {64, 64, 2},          // exact chunk
            {65, 64, 2},          // chunk + 1
            {128, 64, 2},         // exact 2 chunks
            {150, 64, 4},         // non-divisible, depth > chunk count (3 chunks, depth 4)
            {1 << 20, 65536, 2},  // large multi-chunk
            {1 << 20, 65536, 4},  // large multi-chunk, depth 4
            {4096, 0, 1},         // whole-payload single chunk (staged, depth > 1 irrelevant since chunk_hint==0)
            // The async fast-path work item 134: direct path
            // (chunk_hint == 0, HostAsyncBackend/VectorAsyncBackend
            // both report supports_direct_transport_access()==true) at
            // the sizes the plan explicitly asks for. depth is
            // irrelevant here -- the direct path bypasses
            // StagingPool/pipeline_depth entirely -- included anyway to
            // prove that's true (a mismatched depth would only matter
            // if this secretly fell through to the staged path).
            {1, 0, 1},            // 1 byte, direct path
            {4096, 0, 4},         // 4 KiB, direct path
            {256 * 1024, 0, 1},   // 256 KiB, direct path
            {1 << 20, 0, 2},      // 1 MiB, direct path
        };

        std::uint16_t port = kBasePort;
        std::uint64_t transfer_id = 1;

        for (const auto &c : cases)
        {
            const bool ok = run_one_transfer(port++, transfer_id++, c.bytes, c.chunk_hint, c.depth);
            expect(ok, "host round-trip mismatch for bytes=" + std::to_string(c.bytes) +
                           " chunk_hint=" + std::to_string(c.chunk_hint) +
                           " depth=" + std::to_string(c.depth));
        }

        std::cout << "[PASS] test_host_roundtrip_sizes_and_depths\n";
    }

    // Multiple outstanding TransferWork objects, waited in a different
    // order than enqueued: completion of one must not accidentally
    // imply completion of another, and ordered submission over the
    // single persistent connection must still deliver each transfer's
    // exact bytes to the exact matching receive call.
    void test_multiple_outstanding_transfers_ordered_wire_any_order_wait()
    {
        const std::uint16_t port = kBasePort + 100;
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        constexpr std::size_t kBytes = 4096;
        std::vector<std::vector<std::uint8_t>> sources(3), destinations(3);
        std::vector<std::unique_ptr<VectorAsyncBackend>> source_backends, dest_backends;

        for (int i = 0; i < 3; ++i)
        {
            sources[i].resize(kBytes);
            for (std::size_t b = 0; b < kBytes; ++b)
            {
                sources[i][b] = pattern_byte(static_cast<std::uint64_t>(i), b);
            }
            destinations[i].assign(kBytes, 0);
            source_backends.push_back(std::make_unique<VectorAsyncBackend>(sources[i]));
            dest_backends.push_back(std::make_unique<VectorAsyncBackend>(destinations[i]));
        }

        std::vector<bool> receiver_ok(3, false);

        std::thread receiver(
            [&]()
            {
                Endpoint server(listener->accept(), 2);

                // Receiver enqueues its three recvs in the SAME order
                // the sender will send them -- TCP is a single ordered
                // byte stream, so both sides must agree on framing
                // order even though the *caller* can wait on the
                // resulting TransferWork handles in any order it likes
                // (that is what this test actually varies).
                std::vector<tbccl::TransferWork> works;
                for (int i = 0; i < 3; ++i)
                {
                    tbccl::TransferRequest request;
                    request.transfer_id = static_cast<std::uint64_t>(i);
                    request.direction = tbccl::TransferDirection::Recv;
                    request.backend = dest_backends[static_cast<std::size_t>(i)].get();
                    request.transport = server.transport.get();
                    request.total_bytes = kBytes;
                    request.chunk_hint = 1024;
                    works.push_back(server.worker.enqueue(request));
                }

                // Wait in reverse order (C, then A, then B is the
                // classic "any order" case from) -- this only
                // affects when THIS thread observes completion, not
                // the underlying processing order, which is still
                // FIFO on the worker.
                works[2].wait();
                works[0].wait();
                works[1].wait();

                for (int i = 0; i < 3; ++i)
                {
                    receiver_ok[static_cast<std::size_t>(i)] = !works[static_cast<std::size_t>(i)].has_error();
                }
            });

        Endpoint client(tbccl::tcp_connect("127.0.0.1", port, {}), 2);

        std::vector<tbccl::TransferWork> works;
        for (int i = 0; i < 3; ++i)
        {
            tbccl::TransferRequest request;
            request.transfer_id = static_cast<std::uint64_t>(i);
            request.direction = tbccl::TransferDirection::Send;
            request.backend = source_backends[static_cast<std::size_t>(i)].get();
            request.transport = client.transport.get();
            request.total_bytes = kBytes;
            request.chunk_hint = 1024;
            works.push_back(client.worker.enqueue(request));
        }

        // Client also waits out of submission order.
        works[1].wait();
        works[2].wait();
        works[0].wait();

        receiver.join();

        for (int i = 0; i < 3; ++i)
        {
            expect(!works[static_cast<std::size_t>(i)].has_error(),
                   "sender transfer " + std::to_string(i) + " should not error");
            expect(receiver_ok[static_cast<std::size_t>(i)],
                   "receiver transfer " + std::to_string(i) + " should not error");
            expect(destinations[static_cast<std::size_t>(i)] == sources[static_cast<std::size_t>(i)],
                   "transfer " + std::to_string(i) +
                       " destination must exactly match its OWN source, not a neighbor's");
        }

        std::cout << "[PASS] test_multiple_outstanding_transfers_ordered_wire_any_order_wait\n";
    }

    // -------------------------------------------------------------
    // Error propagation
    // -------------------------------------------------------------

    void test_source_backend_error_propagates_to_work()
    {
        const std::uint16_t port = kBasePort + 200;
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        FailingAsyncBackend failing_backend;

        std::thread receiver(
            [&]()
            {
                // Receiver just accepts and lets its recv() fail
                // naturally once the sender never sends (or closes) --
                // this test only asserts on the SENDER side's work.
                auto connection = listener->accept();
                std::uint8_t buffer[1];
                try
                {
                    connection->recv(buffer, sizeof(buffer));
                }
                catch (const std::exception &)
                {
                    // expected: sender fails before sending anything
                }
            });

        bool has_error = false;
        bool error_nonempty = false;
        {
            // Scoped so the client connection closes (letting the
            // receiver's blocking recv() observe EOF and unwind) before
            // this thread joins the receiver below -- the sender fails
            // before sending anything, so nothing else would ever wake
            // that recv().
            Endpoint client(tbccl::tcp_connect("127.0.0.1", port, {}), 2);

            tbccl::TransferRequest request;
            request.transfer_id = 42;
            request.direction = tbccl::TransferDirection::Send;
            request.backend = &failing_backend;
            request.transport = client.transport.get();
            request.total_bytes = 1024;
            request.chunk_hint = 256;

            auto work = client.worker.enqueue(request);
            work.wait();

            has_error = work.has_error();
            error_nonempty = !work.error().empty();
        }

        expect(has_error, "a failing source backend must produce a TransferWork error");
        expect(error_nonempty, "TransferWork error() must carry a message");

        receiver.join();

        std::cout << "[PASS] test_source_backend_error_propagates_to_work\n";
    }

    // -------------------------------------------------------------
    // Backpressure
    // -------------------------------------------------------------

    // enqueue() blocks once queue_depth is reached, rather than
    // growing unbounded.
    void test_enqueue_backpressure_blocks_when_queue_full()
    {
        const std::uint16_t port = kBasePort + 300;
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> destination(64, 0);
        VectorAsyncBackend dest_backend(destination);

        // A receiver that accepts but never drains requests, so
        // whatever the sender enqueues stays queued.
        std::thread receiver(
            [&]()
            {
                auto connection = listener->accept();
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                // Drain exactly one small recv so the sender's queue
                // eventually unblocks and the test can finish cleanly.
                std::uint8_t buffer[64];
                try
                {
                    connection->recv(buffer, sizeof(buffer));
                }
                catch (const std::exception &)
                {
                }
            });

        std::vector<std::uint8_t> source(64, 7);
        VectorAsyncBackend source_backend(source);

        Endpoint client(tbccl::tcp_connect("127.0.0.1", port, {}),
                         /*pipeline_depth=*/1);
        // queue_depth defaults to 4 in Endpoint's constructor.

        tbccl::TransferRequest request;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = client.transport.get();
        request.total_bytes = 64;
        request.chunk_hint = 64;

        // Enqueue more than queue_depth (4) quickly; this must not
        // throw, allocate unbounded memory, or spin -- it simply
        // blocks until the worker drains enough of the backlog.
        std::vector<tbccl::TransferWork> works;
        for (int i = 0; i < 4; ++i)
        {
            works.push_back(client.worker.enqueue(request));
        }

        for (auto &w : works)
        {
            w.wait();
        }

        receiver.join();

        std::cout << "[PASS] test_enqueue_backpressure_blocks_when_queue_full\n";
    }

    // -------------------------------------------------------------
    // Worker shutdown
    // -------------------------------------------------------------

    // Destroying a TensorCommWorker with no outstanding work must not
    // hang.
    void test_worker_shutdown_does_not_deadlock()
    {
        for (int i = 0; i < 5; ++i)
        {
            tbccl::TensorCommWorker worker(2, 4);
            (void)worker;
        }

        std::cout << "[PASS] test_worker_shutdown_does_not_deadlock\n";
    }

    void test_worker_stats_reflect_submitted_completed_failed()
    {
        const std::uint16_t port = kBasePort + 400;
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> destination(64, 0);
        VectorAsyncBackend dest_backend(destination);

        std::thread receiver(
            [&]()
            {
                auto connection = listener->accept();
                std::uint8_t buffer[64];
                connection->recv(buffer, sizeof(buffer));
            });

        std::vector<std::uint8_t> source(64, 3);
        VectorAsyncBackend source_backend(source);

        Endpoint client(tbccl::tcp_connect("127.0.0.1", port, {}), 2);

        tbccl::TransferRequest request;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = client.transport.get();
        request.total_bytes = 64;
        request.chunk_hint = 64;

        auto work = client.worker.enqueue(request);
        work.wait();
        receiver.join();

        const auto stats = client.worker.stats();
        expect(stats.submitted == 1, "stats.submitted should reflect the one enqueue()");
        expect(stats.completed == 1, "stats.completed should reflect the one success");
        expect(stats.failed == 0, "stats.failed should be zero for a clean transfer");

        std::cout << "[PASS] test_worker_stats_reflect_submitted_completed_failed\n";
    }

    // The async fast-path work item 137-139: direct-path buffer
    // lifetime contract. The SAME source/destination buffers are
    // reused across 3 rounds with distinct content each round, only
    // being overwritten AFTER the previous round's
    // TransferWork::wait() has returned -- proves the direct path
    // (which hands the caller's own buffer straight to
    // Transport::send/recv, no TBCCL-owned copy) never reads/writes
    // stale content and never needs a hidden defensive copy to stay
    // correct.
    void test_direct_path_buffer_reused_only_after_wait()
    {
        const std::uint16_t port = kBasePort + 500;
        constexpr std::size_t kBytes = 65536;
        constexpr int kRounds = 3;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> destination(kBytes, 0);
        VectorAsyncBackend dest_backend(destination);
        std::vector<bool> receiver_ok(kRounds, false);

        std::thread receiver(
            [&]()
            {
                Endpoint server(listener->accept(), 2);
                for (int round = 0; round < kRounds; ++round)
                {
                    tbccl::TransferRequest request;
                    request.transfer_id = static_cast<std::uint64_t>(round);
                    request.direction = tbccl::TransferDirection::Recv;
                    request.backend = &dest_backend;
                    request.transport = server.transport.get();
                    request.total_bytes = kBytes;
                    request.chunk_hint = 0; // direct path

                    auto work = server.worker.enqueue(request);
                    work.wait();
                    if (work.has_error())
                    {
                        receiver_ok[static_cast<std::size_t>(round)] = false;
                        continue;
                    }
                    const std::uint8_t expected = pattern_byte(static_cast<std::uint64_t>(round), 0);
                    receiver_ok[static_cast<std::size_t>(round)] =
                        std::all_of(destination.begin(), destination.end(),
                                    [&](std::uint8_t b) { return b == expected; });
                }
            });

        std::vector<std::uint8_t> source(kBytes, 0);
        VectorAsyncBackend source_backend(source);
        Endpoint client(tbccl::tcp_connect("127.0.0.1", port, {}), 2);

        for (int round = 0; round < kRounds; ++round)
        {
            // Overwritten here, AFTER the previous round's wait()
            // returned (or, for round 0, before any transfer) -- never
            // while a transfer referencing this buffer is outstanding.
            std::fill(source.begin(), source.end(), pattern_byte(static_cast<std::uint64_t>(round), 0));

            tbccl::TransferRequest request;
            request.transfer_id = static_cast<std::uint64_t>(round);
            request.direction = tbccl::TransferDirection::Send;
            request.backend = &source_backend;
            request.transport = client.transport.get();
            request.total_bytes = kBytes;
            request.chunk_hint = 0; // direct path

            auto work = client.worker.enqueue(request);
            work.wait();
            expect(!work.has_error(), "round " + std::to_string(round) + " sender must not error");
        }

        receiver.join();

        for (int round = 0; round < kRounds; ++round)
        {
            expect(receiver_ok[static_cast<std::size_t>(round)],
                   "round " + std::to_string(round) +
                       " destination must match that round's own pattern, not a stale/reused one");
        }

        std::cout << "[PASS] test_direct_path_buffer_reused_only_after_wait\n";
    }

} // namespace

int main()
{
    try
    {
        test_chunk_plan_cases();
        test_staging_pool_basic_lifecycle();
        test_staging_pool_acquire_blocks_until_release();
        test_host_roundtrip_sizes_and_depths();
        test_multiple_outstanding_transfers_ordered_wire_any_order_wait();
        test_source_backend_error_propagates_to_work();
        test_enqueue_backpressure_blocks_when_queue_full();
        test_worker_shutdown_does_not_deadlock();
        test_direct_path_buffer_reused_only_after_wait();
        test_worker_stats_reflect_submitted_completed_failed();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
