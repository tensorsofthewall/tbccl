// Correctness tests for TensorBackendAsyncAdapter
// (tensor_backend_async_adapter.hpp): moves a real TensorBackend's
// data through TensorCommWorker over a real TCP loopback connection,
// proving the adapter's staged-once/committed-N-times bookkeeping is
// correct for both the always-available Host backend and (when this
// build has TBCCL_ENABLE_CUDA) a real CUDA backend.

#include "tensor/tensor_backend.hpp"
#include "tensor/tensor_backend_async_adapter.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    using tbccl_bench::tensor::BackendKind;
    using tbccl_bench::tensor::make_backend;
    using tbccl_bench::tensor::TensorBackend;
    using tbccl_bench::tensor::TensorBackendAsyncAdapter;

    constexpr std::uint16_t kBasePort = 28820;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    // Runs one full send/recv round trip for `backend_kind`, verifying
    // the destination backend's buffer exactly matches the source
    // backend's generated pattern after going through
    // TensorBackendAsyncAdapter + TensorCommWorker + TcpTransport.
    void run_backend_roundtrip(
        BackendKind backend_kind,
        std::uint16_t port,
        std::size_t bytes,
        std::size_t chunk_hint,
        std::size_t pipeline_depth)
    {
        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::unique_ptr<TensorBackend> source_backend = make_backend(backend_kind);
        source_backend->allocate(bytes);
        source_backend->initialize_source(/*seed=*/0xC0FFEEu);
        source_backend->prepare_source();

        TensorBackendAsyncAdapter source_adapter(*source_backend);

        bool receiver_ok = true;
        std::thread receiver(
            [&]()
            {
                auto connection = listener->accept();
                tbccl::TcpTransport transport(std::move(connection));
                tbccl::TensorCommWorker worker(pipeline_depth, 2);

                std::unique_ptr<TensorBackend> dest_backend = make_backend(backend_kind);
                dest_backend->allocate(bytes);
                TensorBackendAsyncAdapter dest_adapter(*dest_backend);

                const auto chunks = tbccl::plan_chunks(bytes, chunk_hint, 1);
                dest_adapter.begin_transfer(chunks.empty() ? 1 : chunks.size());

                tbccl::TransferRequest request;
                request.direction = tbccl::TransferDirection::Recv;
                request.backend = &dest_adapter;
                request.transport = &transport;
                request.total_bytes = bytes;
                request.chunk_hint = chunk_hint;

                auto work = worker.enqueue(request);
                work.wait();

                receiver_ok = !work.has_error() && dest_backend->verify_destination(0xC0FFEEu);
            });

        tbccl::TcpTransport client_transport(tbccl::tcp_connect("127.0.0.1", port, {}));
        tbccl::TensorCommWorker worker(pipeline_depth, 2);

        const auto chunks = tbccl::plan_chunks(bytes, chunk_hint, 1);
        source_adapter.begin_transfer(chunks.empty() ? 1 : chunks.size());

        tbccl::TransferRequest request;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_adapter;
        request.transport = &client_transport;
        request.total_bytes = bytes;
        request.chunk_hint = chunk_hint;

        auto work = worker.enqueue(request);
        work.wait();

        receiver.join();

        expect(!work.has_error(), "sender TransferWork must not error");
        expect(receiver_ok, "receiver TransferWork must not error and destination must verify");

        std::cout << "[PASS] run_backend_roundtrip(" << tbccl_bench::tensor::backend_kind_name(backend_kind)
                   << ", bytes=" << bytes << ", chunk_hint=" << chunk_hint
                   << ", depth=" << pipeline_depth << ")\n";
    }

    void test_host_backend_single_chunk()
    {
        run_backend_roundtrip(BackendKind::Host, kBasePort, 65536, 0, 1);
    }

    void test_host_backend_multi_chunk_depth2()
    {
        run_backend_roundtrip(BackendKind::Host, kBasePort + 1, 1 << 20, 65536, 2);
    }

    void test_host_backend_multi_chunk_depth4()
    {
        run_backend_roundtrip(BackendKind::Host, kBasePort + 2, 1 << 20, 65536, 4);
    }

    // Repeated transfers on the SAME adapter instance (begin_transfer()
    // resetting bookkeeping correctly each round) -- catches a stale
    // "already staged"/"already committed enough chunks" bug that a
    // single-transfer test cannot.
    void test_adapter_reused_across_multiple_transfers()
    {
        const std::uint16_t port = kBasePort + 3;
        constexpr std::size_t kBytes = 65536;
        constexpr std::size_t kChunk = 16384;
        constexpr int kRounds = 3;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::unique_ptr<TensorBackend> source_backend = make_backend(BackendKind::Host);
        source_backend->allocate(kBytes);
        TensorBackendAsyncAdapter source_adapter(*source_backend);

        std::vector<bool> receiver_ok(kRounds, false);
        std::thread receiver(
            [&]()
            {
                auto connection = listener->accept();
                tbccl::TcpTransport transport(std::move(connection));
                tbccl::TensorCommWorker worker(2, 2);

                std::unique_ptr<TensorBackend> dest_backend = make_backend(BackendKind::Host);
                dest_backend->allocate(kBytes);
                TensorBackendAsyncAdapter dest_adapter(*dest_backend);

                const auto chunks = tbccl::plan_chunks(kBytes, kChunk, 1);

                for (int round = 0; round < kRounds; ++round)
                {
                    dest_adapter.begin_transfer(chunks.size());

                    tbccl::TransferRequest request;
                    request.direction = tbccl::TransferDirection::Recv;
                    request.backend = &dest_adapter;
                    request.transport = &transport;
                    request.total_bytes = kBytes;
                    request.chunk_hint = kChunk;

                    auto work = worker.enqueue(request);
                    work.wait();

                    receiver_ok[static_cast<std::size_t>(round)] =
                        !work.has_error() && dest_backend->verify_destination(
                                                  static_cast<std::uint32_t>(round));
                }
            });

        tbccl::TcpTransport client_transport(tbccl::tcp_connect("127.0.0.1", port, {}));
        tbccl::TensorCommWorker worker(2, 2);

        const auto chunks = tbccl::plan_chunks(kBytes, kChunk, 1);

        for (int round = 0; round < kRounds; ++round)
        {
            source_backend->initialize_source(static_cast<std::uint32_t>(round));
            source_backend->prepare_source();
            source_adapter.begin_transfer(chunks.size());

            tbccl::TransferRequest request;
            request.direction = tbccl::TransferDirection::Send;
            request.backend = &source_adapter;
            request.transport = &client_transport;
            request.total_bytes = kBytes;
            request.chunk_hint = kChunk;

            auto work = worker.enqueue(request);
            work.wait();

            expect(!work.has_error(), "round " + std::to_string(round) + " sender must not error");
        }

        receiver.join();

        for (int round = 0; round < kRounds; ++round)
        {
            expect(receiver_ok[static_cast<std::size_t>(round)],
                   "round " + std::to_string(round) + " destination must verify against ITS OWN seed");
        }

        std::cout << "[PASS] test_adapter_reused_across_multiple_transfers\n";
    }

#if defined(TBCCL_ENABLE_CUDA)
    void test_cuda_backend_roundtrip()
    {
        if (!tbccl_bench::tensor::backend_kind_available(BackendKind::CudaPinned))
        {
            std::cout << "[SKIP] test_cuda_backend_roundtrip (no CUDA device at runtime)\n";
            return;
        }
        run_backend_roundtrip(BackendKind::CudaPinned, kBasePort + 4, 4 << 20, 262144, 2);
    }
#endif

#if defined(TBCCL_ENABLE_METAL)
    // No Metal-specific adapter code exists -- TensorBackendAsyncAdapter
    // is written entirely against the abstract TensorBackend interface,
    // so it works for metal-shared/metal-private-staged exactly as it
    // does for host/cuda-*, with zero additional code (see this file's
    // top-of-file scope note in tensor_backend_async_adapter.hpp).
    void test_metal_shared_backend_roundtrip()
    {
        if (!tbccl_bench::tensor::backend_kind_available(BackendKind::MetalShared))
        {
            std::cout << "[SKIP] test_metal_shared_backend_roundtrip (no Metal device at runtime)\n";
            return;
        }
        run_backend_roundtrip(BackendKind::MetalShared, kBasePort + 5, 4 << 20, 262144, 2);
    }

    void test_metal_private_staged_backend_roundtrip()
    {
        if (!tbccl_bench::tensor::backend_kind_available(BackendKind::MetalPrivateStaged))
        {
            std::cout << "[SKIP] test_metal_private_staged_backend_roundtrip (no Metal device at runtime)\n";
            return;
        }
        run_backend_roundtrip(BackendKind::MetalPrivateStaged, kBasePort + 6, 4 << 20, 262144, 2);
    }
#endif

} // namespace

int main()
{
    try
    {
        test_host_backend_single_chunk();
        test_host_backend_multi_chunk_depth2();
        test_host_backend_multi_chunk_depth4();
        test_adapter_reused_across_multiple_transfers();
#if defined(TBCCL_ENABLE_CUDA)
        test_cuda_backend_roundtrip();
#endif
#if defined(TBCCL_ENABLE_METAL)
        test_metal_shared_backend_roundtrip();
        test_metal_private_staged_backend_roundtrip();
#endif
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
