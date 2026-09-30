// Phase 35: correctness tests for CudaChunkedAsyncBackend
// (benchmarks/tensor/cuda_chunked_async_backend.{hpp,cu}), the new
// true per-chunk CUDA D2H/H2D async staging backend. Only
// built/run when TBCCL_ENABLE_CUDA is on and only meaningful with a
// CUDA device actually present at runtime.
//
// Covers Part AI's development-order steps 1-3 in one file:
//   1. CUDA local per-chunk staging (round-trips through this
//      backend's own stage_source_chunk/commit_destination_chunk
//      directly, no network).
//   2. CUDA -> host loopback correctness (real TCP, via
//      TensorCommWorker, exercising the actual staged pipeline path).
//   3. Host -> CUDA loopback correctness (roles reversed).

#include "tensor/cuda_chunked_async_backend.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <algorithm>
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
    using tbccl_bench::tensor::CudaChunkedAsyncBackend;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    std::uint8_t pattern_byte(std::size_t i, std::uint32_t seed)
    {
        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value =
            index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
        return static_cast<std::uint8_t>(value & 0xffu);
    }

    // Plain host-memory AsyncMemoryBackend, for the "other side" of a
    // CUDA<->host loopback test (mirrors async_transfer_test.cpp's
    // VectorAsyncBackend).
    class VectorAsyncBackend : public tbccl::AsyncMemoryBackend
    {
    public:
        explicit VectorAsyncBackend(std::vector<std::uint8_t> &buffer) : buffer_(buffer) {}
        void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
        {
            std::memcpy(staging, buffer_.data() + chunk.offset, chunk.size);
        }
        void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
        {
            std::memcpy(buffer_.data() + chunk.offset, staging, chunk.size);
        }

    private:
        std::vector<std::uint8_t> &buffer_;
    };

    constexpr std::uint32_t kSeed = 0xA5A5A5A5u;

    // ---------------------------------------------------------------
    // 1. Local per-chunk staging, no network: drive
    //    stage_source_chunk()/commit_destination_chunk() directly with
    //    a hand-built chunk plan, verifying each chunk lands correctly
    //    and that the pinned scratch buffer is correctly reused across
    //    chunks and across repeated transfers.
    // ---------------------------------------------------------------
    void test_local_chunked_round_trip()
    {
        constexpr std::size_t kBytes = 1 << 20; // 1 MiB
        constexpr std::size_t kChunkBytes = 256 * 1024;

        CudaChunkedAsyncBackend source;
        source.allocate(kBytes, kChunkBytes);
        source.initialize_source(kSeed);
        expect(source.verify_source(kSeed), "source must verify immediately after initialize_source");

        const auto chunks = tbccl::plan_chunks(kBytes, kChunkBytes, 1);
        expect(chunks.size() == 4, "1 MiB / 256 KiB should plan to 4 chunks");

        std::vector<std::uint8_t> wire(kBytes, 0);
        for (const auto &chunk : chunks)
        {
            source.stage_source_chunk(chunk, wire.data() + chunk.offset);
        }
        for (std::size_t i = 0; i < kBytes; ++i)
        {
            expect(wire[i] == pattern_byte(i, kSeed), "staged byte must match pattern at offset " + std::to_string(i));
        }

        CudaChunkedAsyncBackend destination;
        destination.allocate(kBytes, kChunkBytes);
        for (const auto &chunk : chunks)
        {
            destination.commit_destination_chunk(chunk, wire.data() + chunk.offset);
        }
        expect(destination.verify_destination(kSeed), "destination must verify after committing all chunks");

        // Repeat once more, reusing the same instances (Part W: no
        // reallocation on an identical allocate() call).
        const auto pinned_before = source.diagnostic_pinned_alloc_count();
        const auto stream_before = source.diagnostic_stream_create_count();
        source.allocate(kBytes, kChunkBytes);
        source.initialize_source(kSeed + 1);
        for (const auto &chunk : chunks)
        {
            source.stage_source_chunk(chunk, wire.data() + chunk.offset);
        }
        for (std::size_t i = 0; i < kBytes; ++i)
        {
            expect(wire[i] == pattern_byte(i, kSeed + 1), "second-round staged byte must match new pattern");
        }
        expect(source.diagnostic_pinned_alloc_count() == pinned_before,
               "repeated allocate() with identical shape must not reallocate pinned memory");
        expect(source.diagnostic_stream_create_count() == stream_before,
               "repeated allocate() must not recreate the CUDA stream");

        std::cout << "[PASS] test_local_chunked_round_trip\n";
    }

    // Odd-size chunk correctness (Part AC): chunk-1, chunk, chunk+1,
    // 2 chunks + 1 byte, non-aligned final chunk.
    void test_odd_size_chunking()
    {
        constexpr std::size_t kChunkBytes = 65536;
        const std::vector<std::size_t> sizes = {
            kChunkBytes - 1, kChunkBytes, kChunkBytes + 1,
            2 * kChunkBytes + 1, 3 * kChunkBytes - 17,
        };

        for (std::size_t bytes : sizes)
        {
            CudaChunkedAsyncBackend source;
            source.allocate(bytes, kChunkBytes);
            source.initialize_source(kSeed);

            CudaChunkedAsyncBackend destination;
            destination.allocate(bytes, kChunkBytes);

            const auto chunks = tbccl::plan_chunks(bytes, kChunkBytes, 1);
            std::vector<std::uint8_t> wire(bytes, 0);
            for (const auto &chunk : chunks)
            {
                source.stage_source_chunk(chunk, wire.data() + chunk.offset);
                destination.commit_destination_chunk(chunk, wire.data() + chunk.offset);
            }
            expect(destination.verify_destination(kSeed),
                   "odd-size transfer must verify exactly for bytes=" + std::to_string(bytes));
        }

        std::cout << "[PASS] test_odd_size_chunking\n";
    }

    // ---------------------------------------------------------------
    // 2/3. Real TCP loopback through TensorCommWorker: CUDA source ->
    // host destination, then host source -> CUDA destination.
    // ---------------------------------------------------------------
    void test_cuda_to_host_loopback(std::uint16_t port, std::size_t bytes, std::size_t chunk_bytes)
    {
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> destination(bytes, 0);
        VectorAsyncBackend dest_backend(destination);
        bool receiver_ok = false;

        std::thread receiver([&]() {
            auto connection = listener->accept();
            tbccl::TcpTransport transport(std::move(connection));
            tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

            tbccl::TransferRequest request;
            request.transfer_id = 0;
            request.direction = tbccl::TransferDirection::Recv;
            request.backend = &dest_backend;
            request.transport = &transport;
            request.total_bytes = bytes;
            request.chunk_hint = chunk_bytes;

            auto work = worker.enqueue(request);
            work.wait();
            receiver_ok = !work.has_error();
        });

        CudaChunkedAsyncBackend source_backend;
        source_backend.allocate(bytes, chunk_bytes == 0 ? bytes : chunk_bytes);
        source_backend.initialize_source(kSeed);

        auto connection = tbccl::tcp_connect("127.0.0.1", port, {});
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

        tbccl::TransferRequest request;
        request.transfer_id = 0;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = &transport;
        request.total_bytes = bytes;
        request.chunk_hint = chunk_bytes;

        auto work = worker.enqueue(request);
        work.wait();
        expect(!work.has_error(), "CUDA->host sender must not error");

        receiver.join();
        expect(receiver_ok, "CUDA->host receiver must not error");
        for (std::size_t i = 0; i < bytes; ++i)
        {
            expect(destination[i] == pattern_byte(i, kSeed),
                   "CUDA->host destination byte mismatch at offset " + std::to_string(i));
        }

        std::cout << "[PASS] test_cuda_to_host_loopback(bytes=" << bytes
                   << ", chunk=" << chunk_bytes << ")\n";
    }

    void test_host_to_cuda_loopback(std::uint16_t port, std::size_t bytes, std::size_t chunk_bytes)
    {
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        CudaChunkedAsyncBackend dest_backend;
        dest_backend.allocate(bytes, chunk_bytes == 0 ? bytes : chunk_bytes);
        bool receiver_ok = false;

        std::thread receiver([&]() {
            auto connection = listener->accept();
            tbccl::TcpTransport transport(std::move(connection));
            tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

            tbccl::TransferRequest request;
            request.transfer_id = 0;
            request.direction = tbccl::TransferDirection::Recv;
            request.backend = &dest_backend;
            request.transport = &transport;
            request.total_bytes = bytes;
            request.chunk_hint = chunk_bytes;

            auto work = worker.enqueue(request);
            work.wait();
            receiver_ok = !work.has_error();
        });

        std::vector<std::uint8_t> source(bytes);
        for (std::size_t i = 0; i < bytes; ++i) source[i] = pattern_byte(i, kSeed);
        VectorAsyncBackend source_backend(source);

        auto connection = tbccl::tcp_connect("127.0.0.1", port, {});
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

        tbccl::TransferRequest request;
        request.transfer_id = 0;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = &transport;
        request.total_bytes = bytes;
        request.chunk_hint = chunk_bytes;

        auto work = worker.enqueue(request);
        work.wait();
        expect(!work.has_error(), "host->CUDA sender must not error");

        receiver.join();
        expect(receiver_ok, "host->CUDA receiver must not error");
        expect(dest_backend.verify_destination(kSeed), "host->CUDA destination must verify");

        std::cout << "[PASS] test_host_to_cuda_loopback(bytes=" << bytes
                   << ", chunk=" << chunk_bytes << ")\n";
    }

    // Repeated transfers over the same persistent worker/backend (Part
    // AD's "multiple outstanding" spirit, sequential here since
    // TensorCommWorker processes one request at a time in submission
    // order -- Part AU). Each round uses a distinct seed, so stale
    // pinned-scratch-buffer reuse or cross-round slot staleness would
    // produce a detectable mismatch.
    void test_repeated_transfers_same_backend(std::uint16_t port)
    {
        constexpr std::size_t kBytes = 512 * 1024;
        constexpr std::size_t kChunkBytes = 64 * 1024;
        constexpr int kRounds = 3;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> destination(kBytes, 0);
        VectorAsyncBackend dest_backend(destination);
        std::vector<bool> receiver_ok(kRounds, false);

        std::thread receiver([&]() {
            auto connection = listener->accept();
            tbccl::TcpTransport transport(std::move(connection));
            tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

            for (int round = 0; round < kRounds; ++round)
            {
                tbccl::TransferRequest request;
                request.transfer_id = static_cast<std::uint64_t>(round);
                request.direction = tbccl::TransferDirection::Recv;
                request.backend = &dest_backend;
                request.transport = &transport;
                request.total_bytes = kBytes;
                request.chunk_hint = kChunkBytes;

                auto work = worker.enqueue(request);
                work.wait();
                const std::uint32_t seed = kSeed + static_cast<std::uint32_t>(round);
                bool bytes_match = true;
                for (std::size_t i = 0; i < destination.size(); ++i)
                {
                    if (destination[i] != pattern_byte(i, seed)) { bytes_match = false; break; }
                }
                receiver_ok[static_cast<std::size_t>(round)] = !work.has_error() && bytes_match;
            }
        });

        CudaChunkedAsyncBackend source_backend;
        source_backend.allocate(kBytes, kChunkBytes);
        auto connection = tbccl::tcp_connect("127.0.0.1", port, {});
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

        for (int round = 0; round < kRounds; ++round)
        {
            const std::uint32_t seed = kSeed + static_cast<std::uint32_t>(round);
            source_backend.allocate(kBytes, kChunkBytes);
            source_backend.initialize_source(seed);

            tbccl::TransferRequest request;
            request.transfer_id = static_cast<std::uint64_t>(round);
            request.direction = tbccl::TransferDirection::Send;
            request.backend = &source_backend;
            request.transport = &transport;
            request.total_bytes = kBytes;
            request.chunk_hint = kChunkBytes;

            auto work = worker.enqueue(request);
            work.wait();
            expect(!work.has_error(), "round " + std::to_string(round) + " sender must not error");
        }

        receiver.join();
        for (int round = 0; round < kRounds; ++round)
        {
            expect(receiver_ok[static_cast<std::size_t>(round)],
                   "round " + std::to_string(round) + " destination must match that round's own seed");
        }
        std::cout << "[PASS] test_repeated_transfers_same_backend\n";
    }

} // namespace

int main()
{
    try
    {
        test_local_chunked_round_trip();
        test_odd_size_chunking();
        test_cuda_to_host_loopback(29500, 1 << 20, 256 * 1024);
        test_cuda_to_host_loopback(29501, 1 << 20, 0); // chunk_hint 0: single whole-buffer chunk, depth1 control
        test_host_to_cuda_loopback(29502, 1 << 20, 256 * 1024);
        test_repeated_transfers_same_backend(29503);
        std::cout << "All tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
