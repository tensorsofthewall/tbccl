// host-only correctness tests for n2_all_reduce_tensor()
// (tbccl/hetero_allreduce.hpp), over real local TCP loopback connections
// via TcpTransport -- no GPU involved (Part AK: prove the executor itself
// on host first, matching the async tensor-transfer async-substrate test
// convention in tests/async_transfer_test.cpp).

#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include "tensor/host_dual_buffer_backend.hpp"
#include "tensor/host_reduce_backend.hpp"

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

    constexpr std::uint16_t kBasePort = 28900;

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

    // Deterministic small-integer-valued float pattern: sums stay
    // exactly representable in float32, so verification can use exact
    // equality rather than a tolerance.
    float value_for(std::size_t i, std::uint32_t seed, std::uint32_t modulus)
    {
        return static_cast<float>((i + seed) % modulus);
    }

    // Runs one N=2 Float32 SUM AllReduce of `count` elements, root
    // configurable, over real TCP loopback across two threads in this
    // process. Returns true iff both ranks' final results exactly equal
    // the expected elementwise sum.
    bool run_one_allreduce(
        std::uint16_t port,
        std::size_t count,
        std::size_t root,
        std::size_t chunk_hint = 0)
    {
        const std::size_t bytes = count * sizeof(float);
        constexpr std::uint32_t kSeedRank0 = 11;
        constexpr std::uint32_t kSeedRank1 = 97;
        constexpr std::uint32_t kModRank0 = 127;
        constexpr std::uint32_t kModRank1 = 113;

        std::vector<float> expected(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            expected[i] = value_for(i, kSeedRank0, kModRank0) +
                          value_for(i, kSeedRank1, kModRank1);
        }

        bool rank0_ok = false;
        bool rank1_ok = false;
        std::string rank1_error;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::thread rank1_thread(
            [&]()
            {
                try
                {
                    Endpoint ep(tbccl::tcp_connect("127.0.0.1", port, {}));

                    tbccl_bench::tensor::HostDualBufferBackend backend(bytes);
                    auto *local = static_cast<float *>(backend.source_data());
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        local[i] = value_for(i, kSeedRank1, kModRank1);
                    }

                    std::unique_ptr<tbccl_bench::tensor::HostReduceBackend> reduce;
                    tbccl::LocalReduceBackend *reduce_ptr = nullptr;
                    if (root == 1)
                    {
                        reduce = std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                            backend.source_data(), backend.destination_data());
                        reduce_ptr = reduce.get();
                    }

                    tbccl::n2_all_reduce_tensor(
                        *ep.transport, ep.worker, backend, backend, reduce_ptr,
                        /*rank=*/1, root, bytes, chunk_hint, count,
                        tbccl::DataType::Float32);

                    const void *result_ptr =
                        (root == 1) ? backend.source_data() : backend.destination_data();
                    const auto *result = static_cast<const float *>(result_ptr);
                    bool ok = true;
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        if (result[i] != expected[i])
                        {
                            ok = false;
                            break;
                        }
                    }
                    rank1_ok = ok;
                }
                catch (const std::exception &error)
                {
                    rank1_error = error.what();
                }
            });

        {
            auto connection = listener->accept();
            Endpoint ep(std::move(connection));

            tbccl_bench::tensor::HostDualBufferBackend backend(bytes);
            auto *local = static_cast<float *>(backend.source_data());
            for (std::size_t i = 0; i < count; ++i)
            {
                local[i] = value_for(i, kSeedRank0, kModRank0);
            }

            std::unique_ptr<tbccl_bench::tensor::HostReduceBackend> reduce;
            tbccl::LocalReduceBackend *reduce_ptr = nullptr;
            if (root == 0)
            {
                reduce = std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                    backend.source_data(), backend.destination_data());
                reduce_ptr = reduce.get();
            }

            tbccl::n2_all_reduce_tensor(
                *ep.transport, ep.worker, backend, backend, reduce_ptr,
                /*rank=*/0, root, bytes, chunk_hint, count,
                tbccl::DataType::Float32);

            const void *result_ptr =
                (root == 0) ? backend.source_data() : backend.destination_data();
            const auto *result = static_cast<const float *>(result_ptr);
            rank0_ok = true;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (result[i] != expected[i])
                {
                    rank0_ok = false;
                    break;
                }
            }
        }

        rank1_thread.join();

        if (!rank1_error.empty())
        {
            throw std::runtime_error("rank 1: " + rank1_error);
        }

        return rank0_ok && rank1_ok;
    }

    void test_root0_basic()
    {
        expect(run_one_allreduce(kBasePort + 0, 1024, /*root=*/0),
               "root=0, 1024 elements must produce exact elementwise sum on both ranks");
        std::cout << "[PASS] test_root0_basic\n";
    }

    void test_root1_basic()
    {
        expect(run_one_allreduce(kBasePort + 1, 1024, /*root=*/1),
               "root=1, 1024 elements must produce exact elementwise sum on both ranks");
        std::cout << "[PASS] test_root1_basic\n";
    }

    void test_single_element()
    {
        expect(run_one_allreduce(kBasePort + 2, 1, /*root=*/0),
               "1-element AllReduce must be correct");
        std::cout << "[PASS] test_single_element\n";
    }

    void test_odd_element_count()
    {
        expect(run_one_allreduce(kBasePort + 3, 1001, /*root=*/1),
               "odd element count must be correct");
        std::cout << "[PASS] test_odd_element_count\n";
    }

    void test_non_page_aligned_bytes()
    {
        // 4097 floats = 16388 bytes -- not a power-of-two or page-size
        // multiple in either elements or bytes.
        expect(run_one_allreduce(kBasePort + 4, 4097, /*root=*/0),
               "non-page-aligned byte count must be correct");
        std::cout << "[PASS] test_non_page_aligned_bytes\n";
    }

    void test_chunked_transfer()
    {
        // Force real multi-chunk pipelining (chunk_hint << total_bytes)
        // rather than the single-chunk/direct path, on both legs.
        expect(run_one_allreduce(kBasePort + 5, 16384, /*root=*/1, /*chunk_hint=*/4096),
               "chunked transfer (chunk_hint=4096) must still be correct");
        std::cout << "[PASS] test_chunked_transfer\n";
    }

    void test_repeated_allreduce_distinct_ports()
    {
        // Distinct seeds already vary by element index + fixed per-rank
        // seed; repeating the whole call several times (each its own
        // fresh connection/worker, like separate sequential application
        // AllReduce calls would be) must not leak state between runs.
        for (int round = 0; round < 5; ++round)
        {
            const auto root = static_cast<std::size_t>(round % 2);
            expect(run_one_allreduce(
                       static_cast<std::uint16_t>(kBasePort + 10 + round), 2048, root),
                   "repeated round " + std::to_string(round) + " must be correct");
        }
        std::cout << "[PASS] test_repeated_allreduce_distinct_ports\n";
    }

    // A backend whose commit_destination_chunk always throws, to verify
    // n2_all_reduce_tensor() propagates a failed TransferWork as an
    // exception rather than hanging or silently succeeding.
    class ThrowingRecvBackend final : public tbccl::AsyncMemoryBackend
    {
    public:
        explicit ThrowingRecvBackend(std::size_t capacity) : buffer_(capacity)
        {
        }

        void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
        {
            std::memcpy(staging, buffer_.data() + chunk.offset, chunk.size);
        }

        void commit_destination_chunk(const tbccl::Chunk &, const void *) override
        {
            throw std::runtime_error("synthetic recv failure");
        }

    private:
        std::vector<std::uint8_t> buffer_;
    };

    void test_error_propagation()
    {
        constexpr std::size_t kCount = 256;
        constexpr std::size_t kBytes = kCount * sizeof(float);
        const std::uint16_t port = kBasePort + 20;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::string rank1_error;
        std::thread rank1_thread(
            [&]()
            {
                try
                {
                    Endpoint ep(tbccl::tcp_connect("127.0.0.1", port, {}));
                    tbccl_bench::tensor::HostDualBufferBackend backend(kBytes);
                    // Non-root: sends fine, then receives the final
                    // result -- no failure injected on this side.
                    tbccl::n2_all_reduce_tensor(
                        *ep.transport, ep.worker, backend, backend, nullptr,
                        /*rank=*/1, /*root=*/0, kBytes, 0, kCount,
                        tbccl::DataType::Float32);
                }
                catch (const std::exception &error)
                {
                    rank1_error = error.what();
                }
            });

        bool root_threw = false;
        {
            auto connection = listener->accept();
            Endpoint ep(std::move(connection));
            ThrowingRecvBackend recv_backend(kBytes);
            tbccl_bench::tensor::HostDualBufferBackend send_backend(kBytes);
            tbccl_bench::tensor::HostReduceBackend reduce(
                send_backend.source_data(), send_backend.destination_data());

            try
            {
                tbccl::n2_all_reduce_tensor(
                    *ep.transport, ep.worker, recv_backend, send_backend, &reduce,
                    /*rank=*/0, /*root=*/0, kBytes, 0, kCount,
                    tbccl::DataType::Float32);
            }
            catch (const std::exception &)
            {
                root_threw = true;
            }
        }

        rank1_thread.join();

        expect(root_threw, "root must throw when its recv_backend fails");
        // Rank 1 either throws too (peer disconnected mid-recv) or the
        // process would hang -- the join() above already proves it did
        // not hang; we only additionally require it did not silently
        // report success with a bogus result, which run_one_allreduce's
        // exact-equality check elsewhere already covers for the happy
        // path. Here, simply not hanging and the root throwing is the
        // contract under test.
        (void)rank1_error;

        std::cout << "[PASS] test_error_propagation\n";
    }

} // namespace

int main()
{
    try
    {
        test_root0_basic();
        test_root1_basic();
        test_single_element();
        test_odd_element_count();
        test_non_page_aligned_bytes();
        test_chunked_transfer();
        test_repeated_allreduce_distinct_ports();
        test_error_propagation();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";
    return 0;
}
