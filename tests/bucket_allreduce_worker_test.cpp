// Phase 39 Part BH/BI/BJ: tests for BucketAllReduceWorker
// (benchmarks/bucket_allreduce_worker.hpp), the benchmark-support
// persistent collective progress worker. Host-only, real TCP loopback,
// same pattern as tests/hetero_allreduce_test.cpp.

#include "bucket_allreduce_worker.hpp"
#include "tensor/host_dual_buffer_backend.hpp"
#include "tensor/host_reduce_backend.hpp"

#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

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
    constexpr std::uint16_t kBasePort = 29100;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    struct Endpoint
    {
        std::unique_ptr<tbccl::Transport> transport;
        tbccl::TensorCommWorker worker;

        explicit Endpoint(std::unique_ptr<tbccl::Connection> connection)
            : transport(std::make_unique<tbccl::TcpTransport>(std::move(connection))),
              worker(/*pipeline_depth=*/2, /*queue_depth=*/4)
        {
        }
    };

    float value_for(std::size_t i, std::uint32_t seed, std::uint32_t modulus)
    {
        return static_cast<float>((i + seed) % modulus);
    }

    // Runs `bucket_count` sequential bucket AllReduces (root=0) through a
    // BucketAllReduceWorker on each rank, across real TCP loopback,
    // verifying FIFO order (bucket seeds are chosen so each bucket's
    // expected sum is distinct and position-identifiable) and exact
    // correctness.
    void test_fifo_order_and_correctness()
    {
        constexpr std::size_t kBucketCount = 4;
        constexpr std::size_t kCount = 1024;
        constexpr std::size_t kBytes = kCount * sizeof(float);
        const std::uint16_t port = kBasePort + 0;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::vector<float>> expected(kBucketCount, std::vector<float>(kCount));
        for (std::size_t b = 0; b < kBucketCount; ++b)
        {
            for (std::size_t i = 0; i < kCount; ++i)
            {
                expected[b][i] = value_for(i, 11u + static_cast<std::uint32_t>(b), 127) +
                                  value_for(i, 97u + static_cast<std::uint32_t>(b), 113);
            }
        }

        bool rank1_ok = true;
        std::string rank1_error;
        std::thread rank1_thread(
            [&]()
            {
                try
                {
                    Endpoint ep(tbccl::tcp_connect("127.0.0.1", port, {}));
                    tbccl_bench::BucketAllReduceWorker worker(/*queue_depth=*/kBucketCount + 1);

                    std::vector<std::unique_ptr<tbccl_bench::tensor::HostDualBufferBackend>> backends;
                    std::vector<tbccl_bench::BucketAllReduceWork> works;
                    works.reserve(kBucketCount);
                    for (std::size_t b = 0; b < kBucketCount; ++b)
                    {
                        backends.push_back(
                            std::make_unique<tbccl_bench::tensor::HostDualBufferBackend>(kBytes));
                        auto *src = static_cast<float *>(backends.back()->source_data());
                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            src[i] = value_for(i, 97u + static_cast<std::uint32_t>(b), 113);
                        }

                        tbccl_bench::BucketAllReduceJob job;
                        job.transport = ep.transport.get();
                        job.worker = &ep.worker;
                        job.recv_backend = backends.back().get();
                        job.send_backend = backends.back().get();
                        job.reduce_backend = nullptr;
                        job.rank = 1;
                        job.root = 0;
                        job.total_bytes = kBytes;
                        job.chunk_hint = 0;
                        job.count = kCount;
                        job.bucket_index = b;

                        works.push_back(worker.enqueue(job));
                    }

                    for (std::size_t b = 0; b < kBucketCount; ++b)
                    {
                        works[b].wait();
                        if (works[b].has_error())
                        {
                            rank1_ok = false;
                            rank1_error = works[b].error();
                            break;
                        }
                        const auto *result =
                            static_cast<const float *>(backends[b]->destination_data());
                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            if (result[i] != expected[b][i])
                            {
                                rank1_ok = false;
                                rank1_error = "bucket " + std::to_string(b) + " mismatch";
                                break;
                            }
                        }
                    }
                }
                catch (const std::exception &error)
                {
                    rank1_ok = false;
                    rank1_error = error.what();
                }
            });

        bool rank0_ok = true;
        {
            auto connection = listener->accept();
            Endpoint ep(std::move(connection));
            tbccl_bench::BucketAllReduceWorker worker(/*queue_depth=*/kBucketCount + 1);

            std::vector<std::unique_ptr<tbccl_bench::tensor::HostDualBufferBackend>> backends;
            std::vector<std::unique_ptr<tbccl_bench::tensor::HostReduceBackend>> reduces;
            std::vector<tbccl_bench::BucketAllReduceWork> works;
            works.reserve(kBucketCount);
            for (std::size_t b = 0; b < kBucketCount; ++b)
            {
                backends.push_back(
                    std::make_unique<tbccl_bench::tensor::HostDualBufferBackend>(kBytes));
                auto *src = static_cast<float *>(backends.back()->source_data());
                for (std::size_t i = 0; i < kCount; ++i)
                {
                    src[i] = value_for(i, 11u + static_cast<std::uint32_t>(b), 127);
                }
                reduces.push_back(std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                    backends.back()->source_data(), backends.back()->destination_data()));

                tbccl_bench::BucketAllReduceJob job;
                job.transport = ep.transport.get();
                job.worker = &ep.worker;
                job.recv_backend = backends.back().get();
                job.send_backend = backends.back().get();
                job.reduce_backend = reduces.back().get();
                job.rank = 0;
                job.root = 0;
                job.total_bytes = kBytes;
                job.chunk_hint = 0;
                job.count = kCount;
                job.bucket_index = b;

                // Submitting bucket b+1 here, BEFORE waiting on bucket b,
                // is exactly the "producer runs ahead" pattern Part R
                // exists to support -- this is the FIFO-order assertion
                // under test, not an afterthought.
                works.push_back(worker.enqueue(job));
            }

            for (std::size_t b = 0; b < kBucketCount; ++b)
            {
                works[b].wait();
                if (works[b].has_error())
                {
                    rank0_ok = false;
                    break;
                }
                const auto *result = static_cast<const float *>(backends[b]->source_data());
                for (std::size_t i = 0; i < kCount; ++i)
                {
                    if (result[i] != expected[b][i])
                    {
                        rank0_ok = false;
                        break;
                    }
                }
            }
        }

        rank1_thread.join();
        expect(rank0_ok, "rank 0 (root) buckets must all be correct and in FIFO order");
        expect(rank1_ok, "rank 1: " + rank1_error);

        std::cout << "[PASS] test_fifo_order_and_correctness\n";
    }

    // A job whose backend always throws on commit; verifies the worker
    // reports the error on that job's Work, drains/fails subsequent
    // queued jobs rather than executing them, and shutdown still
    // completes cleanly (Part BJ).
    class ThrowingBackend final : public tbccl::AsyncMemoryBackend
    {
    public:
        explicit ThrowingBackend(std::size_t capacity) : buffer_(capacity) {}

        void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
        {
            std::memcpy(staging, buffer_.data() + chunk.offset, chunk.size);
        }

        void commit_destination_chunk(const tbccl::Chunk &, const void *) override
        {
            throw std::runtime_error("synthetic failure");
        }

    private:
        std::vector<std::uint8_t> buffer_;
    };

    void test_error_propagation_and_abort()
    {
        constexpr std::size_t kCount = 64;
        constexpr std::size_t kBytes = kCount * sizeof(float);
        const std::uint16_t port = kBasePort + 10;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::thread rank1_thread(
            [&]()
            {
                try
                {
                    Endpoint ep(tbccl::tcp_connect("127.0.0.1", port, {}));
                    tbccl_bench::tensor::HostDualBufferBackend backend(kBytes);
                    // Non-root: just participate normally; root's recv
                    // failure is what this test is actually about.
                    tbccl_bench::BucketAllReduceJob job;
                    job.transport = ep.transport.get();
                    job.worker = &ep.worker;
                    job.recv_backend = &backend;
                    job.send_backend = &backend;
                    job.reduce_backend = nullptr;
                    job.rank = 1;
                    job.root = 0;
                    job.total_bytes = kBytes;
                    job.chunk_hint = 0;
                    job.count = kCount;

                    tbccl_bench::BucketAllReduceWorker worker;
                    auto work = worker.enqueue(job);
                    work.wait();
                }
                catch (...)
                {
                    // Peer may see a connection error if root aborts
                    // first -- not the thing under test on this side.
                }
            });

        {
            auto connection = listener->accept();
            Endpoint ep(std::move(connection));
            tbccl_bench::BucketAllReduceWorker worker(/*queue_depth=*/4);

            ThrowingBackend failing_backend(kBytes);
            tbccl_bench::tensor::HostDualBufferBackend ok_backend(kBytes);
            tbccl_bench::tensor::HostReduceBackend reduce(
                ok_backend.source_data(), ok_backend.destination_data());

            tbccl_bench::BucketAllReduceJob failing_job;
            failing_job.transport = ep.transport.get();
            failing_job.worker = &ep.worker;
            failing_job.recv_backend = &failing_backend;
            failing_job.send_backend = &failing_backend;
            failing_job.reduce_backend = &reduce;
            failing_job.rank = 0;
            failing_job.root = 0;
            failing_job.total_bytes = kBytes;
            failing_job.chunk_hint = 0;
            failing_job.count = kCount;

            auto failing_work = worker.enqueue(failing_job);
            failing_work.wait();
            expect(failing_work.has_error(), "the injected failure must be reported on its Work");

            // A second job submitted after the failure must also report
            // an error (worker aborted), not hang or silently succeed.
            tbccl_bench::BucketAllReduceJob second_job = failing_job;
            second_job.recv_backend = &ok_backend;
            second_job.send_backend = &ok_backend;
            auto second_work = worker.enqueue(second_job);
            second_work.wait();
            expect(second_work.has_error(),
                   "a job submitted after an abort must also fail, not hang or silently run");
        }

        rank1_thread.join();
        std::cout << "[PASS] test_error_propagation_and_abort\n";
    }

    void test_clean_shutdown_no_jobs()
    {
        {
            tbccl_bench::BucketAllReduceWorker worker;
            (void)worker;
        }
        std::cout << "[PASS] test_clean_shutdown_no_jobs\n";
    }

    void test_repeated_small_jobs_host_loopback()
    {
        // Rank 0 and rank 1 both loop back to the SAME process via two
        // real TCP endpoints is not meaningful for an N=2 protocol (each
        // side needs its own peer) -- reuse the FIFO test's shape but
        // with more rounds and alternating roots, covering Part BK's
        // "repeated application iterations, no stale data / no Work
        // leakage" at the worker level specifically (not just the
        // underlying collective, already covered by
        // hetero_allreduce_test's own repeated-call test).
        constexpr std::size_t kRounds = 20;
        constexpr std::size_t kCount = 256;
        constexpr std::size_t kBytes = kCount * sizeof(float);
        const std::uint16_t port = kBasePort + 20;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        bool rank1_ok = true;
        std::thread rank1_thread(
            [&]()
            {
                try
                {
                    Endpoint ep(tbccl::tcp_connect("127.0.0.1", port, {}));
                    tbccl_bench::BucketAllReduceWorker worker;
                    for (std::size_t round = 0; round < kRounds; ++round)
                    {
                        const auto root = round % 2;
                        tbccl_bench::tensor::HostDualBufferBackend backend(kBytes);
                        auto *src = static_cast<float *>(backend.source_data());
                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            src[i] = value_for(i, 97u + static_cast<std::uint32_t>(round), 113);
                        }
                        std::unique_ptr<tbccl_bench::tensor::HostReduceBackend> reduce;
                        tbccl::LocalReduceBackend *reduce_ptr = nullptr;
                        if (root == 1)
                        {
                            reduce = std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                                backend.source_data(), backend.destination_data());
                            reduce_ptr = reduce.get();
                        }

                        tbccl_bench::BucketAllReduceJob job;
                        job.transport = ep.transport.get();
                        job.worker = &ep.worker;
                        job.recv_backend = &backend;
                        job.send_backend = &backend;
                        job.reduce_backend = reduce_ptr;
                        job.rank = 1;
                        job.root = root;
                        job.total_bytes = kBytes;
                        job.chunk_hint = 0;
                        job.count = kCount;

                        auto work = worker.enqueue(job);
                        work.wait();
                        if (work.has_error()) { rank1_ok = false; break; }

                        const void *ptr = (root == 1) ? backend.source_data() : backend.destination_data();
                        const auto *result = static_cast<const float *>(ptr);
                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            const float expected = value_for(i, 11u + static_cast<std::uint32_t>(round), 127) +
                                                    value_for(i, 97u + static_cast<std::uint32_t>(round), 113);
                            if (result[i] != expected) { rank1_ok = false; break; }
                        }
                        if (!rank1_ok) break;
                    }
                }
                catch (const std::exception &)
                {
                    rank1_ok = false;
                }
            });

        bool rank0_ok = true;
        {
            auto connection = listener->accept();
            Endpoint ep(std::move(connection));
            tbccl_bench::BucketAllReduceWorker worker;
            for (std::size_t round = 0; round < kRounds; ++round)
            {
                const auto root = round % 2;
                tbccl_bench::tensor::HostDualBufferBackend backend(kBytes);
                auto *src = static_cast<float *>(backend.source_data());
                for (std::size_t i = 0; i < kCount; ++i)
                {
                    src[i] = value_for(i, 11u + static_cast<std::uint32_t>(round), 127);
                }
                std::unique_ptr<tbccl_bench::tensor::HostReduceBackend> reduce;
                tbccl::LocalReduceBackend *reduce_ptr = nullptr;
                if (root == 0)
                {
                    reduce = std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                        backend.source_data(), backend.destination_data());
                    reduce_ptr = reduce.get();
                }

                tbccl_bench::BucketAllReduceJob job;
                job.transport = ep.transport.get();
                job.worker = &ep.worker;
                job.recv_backend = &backend;
                job.send_backend = &backend;
                job.reduce_backend = reduce_ptr;
                job.rank = 0;
                job.root = root;
                job.total_bytes = kBytes;
                job.chunk_hint = 0;
                job.count = kCount;

                auto work = worker.enqueue(job);
                work.wait();
                if (work.has_error()) { rank0_ok = false; break; }

                const void *ptr = (root == 0) ? backend.source_data() : backend.destination_data();
                const auto *result = static_cast<const float *>(ptr);
                for (std::size_t i = 0; i < kCount; ++i)
                {
                    const float expected = value_for(i, 11u + static_cast<std::uint32_t>(round), 127) +
                                            value_for(i, 97u + static_cast<std::uint32_t>(round), 113);
                    if (result[i] != expected) { rank0_ok = false; break; }
                }
                if (!rank0_ok) break;
            }
        }

        rank1_thread.join();
        expect(rank0_ok, "rank 0: repeated rounds must all be correct, no stale data");
        expect(rank1_ok, "rank 1: repeated rounds must all be correct, no stale data");

        std::cout << "[PASS] test_repeated_small_jobs_host_loopback\n";
    }

} // namespace

int main()
{
    try
    {
        test_clean_shutdown_no_jobs();
        test_fifo_order_and_correctness();
        test_error_propagation_and_abort();
        test_repeated_small_jobs_host_loopback();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";
    return 0;
}
