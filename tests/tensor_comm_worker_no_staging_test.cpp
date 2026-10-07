// Tests for the TBCCL_ASYNC_NO_STAGING_THREAD
// diagnostic control (src/core/tensor_comm_worker.cpp), which skips
// creating TensorCommWorker's staging thread entirely -- used to test
// the "idle staging thread affects network-thread scheduling"
// hypothesis (found NOT to matter: the no-staging-thread direct-path
// transfer measured the same as the two-thread configuration; the
// actual async fast-path "unexplained" gap turned out to be a benchmark
// methodology artifact.md).
//
// This is its own test BINARY, not folded into async_transfer_test.cpp,
// because no_staging_thread_enabled() in tensor_comm_worker.cpp caches
// getenv() on first call (the first TensorCommWorker ever constructed
// in the process) -- async_transfer_test.cpp already constructs many
// TensorCommWorkers with the env var unset, so setenv() after that
// point would have no effect. setenv() must happen here, first, before
// any TensorCommWorker exists in this process.

#include <tbccl/async_transfer.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
    constexpr std::uint16_t kPort = 28920;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    // Plain host memory: directly transport-accessible, exercising
    // TensorCommWorker's direct path -- the only path this diagnostic
    // mode supports (no staging thread to service the staged path).
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
        bool supports_direct_transport_access() const noexcept override { return true; }
        const void *direct_source_data() const noexcept override { return buffer_.data(); }
        void *direct_destination_data() noexcept override { return buffer_.data(); }

    private:
        std::vector<std::uint8_t> &buffer_;
    };

    void test_direct_transfer_correct_with_no_staging_thread()
    {
        constexpr std::size_t kBytes = 65536;
        auto listener = tbccl::tcp_listen("127.0.0.1", kPort, {});

        std::vector<std::uint8_t> destination(kBytes, 0);
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
            request.total_bytes = kBytes;
            request.chunk_hint = 0; // direct path

            auto work = worker.enqueue(request);
            work.wait();
            receiver_ok = !work.has_error();
        });

        std::vector<std::uint8_t> source(kBytes, 0xC3);
        VectorAsyncBackend source_backend(source);
        auto connection = tbccl::tcp_connect("127.0.0.1", kPort, {});
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

        tbccl::TransferRequest request;
        request.transfer_id = 0;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = &transport;
        request.total_bytes = kBytes;
        request.chunk_hint = 0; // direct path

        auto work = worker.enqueue(request);
        work.wait();
        expect(!work.has_error(), "sender must not error in no-staging-thread mode");

        receiver.join();
        expect(receiver_ok, "receiver must not error in no-staging-thread mode");
        expect(destination == source,
               "destination content must match exactly with no staging thread");

        std::cout << "[PASS] test_direct_transfer_correct_with_no_staging_thread\n";
    }

    void test_staged_path_request_fails_cleanly_instead_of_deadlocking()
    {
        // A request that needs the staged path (chunk_hint != 0, or a
        // backend that doesn't support the direct path) must fail with
        // an error through TransferWork, NOT deadlock forever waiting
        // for a staging thread that was never created -- the safety
        // net in tensor_comm_worker.cpp's process_request().
        constexpr std::size_t kBytes = 65536;
        auto listener = tbccl::tcp_listen("127.0.0.1", kPort + 1, {});

        std::vector<std::uint8_t> destination(kBytes, 0);
        VectorAsyncBackend dest_backend(destination);

        std::thread receiver([&]() {
            auto connection = listener->accept();
            (void)connection; // receiver side not exercised by this test
        });

        std::vector<std::uint8_t> source(kBytes, 0x5A);
        VectorAsyncBackend source_backend(source);
        auto connection = tbccl::tcp_connect("127.0.0.1", kPort + 1, {});
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

        tbccl::TransferRequest request;
        request.transfer_id = 0;
        request.direction = tbccl::TransferDirection::Send;
        request.backend = &source_backend;
        request.transport = &transport;
        request.total_bytes = kBytes;
        request.chunk_hint = 4096; // forces the staged path

        auto work = worker.enqueue(request);
        work.wait(); // must return promptly with an error, not hang
        expect(work.has_error(), "staged-path request must fail cleanly, not hang, "
                                  "when the staging thread was disabled");

        receiver.join();
        std::cout << "[PASS] test_staged_path_request_fails_cleanly_instead_of_deadlocking\n";
    }

} // namespace

int main()
{
    // Must happen before any TensorCommWorker is constructed in this
    // process -- see file header comment.
#if defined(_WIN32)
    _putenv_s("TBCCL_ASYNC_NO_STAGING_THREAD", "1");
#else
    setenv("TBCCL_ASYNC_NO_STAGING_THREAD", "1", 1);
#endif

    try
    {
        test_direct_transfer_correct_with_no_staging_thread();
        test_staged_path_request_fails_cleanly_instead_of_deadlocking();
        std::cout << "All tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
