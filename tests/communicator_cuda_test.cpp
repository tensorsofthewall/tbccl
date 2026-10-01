// Phase 41 Part J/K/L/M/AS/AT/AU/AV: correctness tests for the external
// CUDA memory provider (benchmarks/tensor/cuda_external_async_backend.{hpp,cu},
// registered via benchmarks/tensor/cuda_memory_provider.{hpp,cu}) behind
// the public tbccl::Communicator API. Only built/run when
// TBCCL_ENABLE_CUDA is on and only meaningful with a real CUDA device.

#include "tensor/cuda_external_async_backend.hpp"
#include "tensor/cuda_memory_provider.hpp"
#include "tensor/cuda_reduce_backend.hpp" // cuda_copy_host_to_device/device_to_host helpers

#include <tbccl/communicator.hpp>

#include <cuda_runtime.h>

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

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(status));
    }
}

constexpr std::uint16_t kBasePort = 29300;

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

// ---------------------------------------------------------------------
// Part J: external CUDA pointer ownership + correctness.
// ---------------------------------------------------------------------

void test_external_cuda_pointer_ownership_and_roundtrip()
{
    const std::size_t count = 65536;
    const std::size_t bytes = count * sizeof(std::int32_t);

    // Allocated OUTSIDE any TBCCL provider -- proves TBCCL can wrap, use,
    // and never free this pointer.
    void *device_ptr = nullptr;
    check_cuda(cudaMalloc(&device_ptr, bytes), "cudaMalloc (external test buffer)");

    std::vector<std::int32_t> host_src(count), host_dst(count, 0);
    for (std::size_t i = 0; i < count; ++i) host_src[i] = static_cast<std::int32_t>(i * 7 - 3);
    tbccl_bench::tensor::cuda_copy_host_to_device(host_src.data(), device_ptr, bytes);

    {
        std::vector<std::int32_t> recv_buf(count, 0);
        run_pair(
            kBasePort,
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Cuda, device_ptr, bytes, 0};
                auto work = comm.send(view, count, tbccl::DataType::Int32, 1);
                work.wait();
                expect(!work.has_error(), "CUDA external sender must not error: " + work.error());
            },
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, recv_buf.data(), bytes, 0};
                auto work = comm.recv(view, count, tbccl::DataType::Int32, 0);
                work.wait();
                expect(!work.has_error(), "host receiver must not error: " + work.error());
            });
        expect(std::memcmp(host_src.data(), recv_buf.data(), bytes) == 0, "CUDA external -> host must reproduce exact bytes");
    }

    // Symmetric direction: host -> external CUDA destination, then
    // independent GPU readback (not host-readback) to verify.
    {
        std::vector<std::int32_t> host_src2(count);
        for (std::size_t i = 0; i < count; ++i) host_src2[i] = static_cast<std::int32_t>(i * 3 + 11);

        run_pair(
            kBasePort + 2,
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Cuda, device_ptr, bytes, 0};
                auto work = comm.recv(view, count, tbccl::DataType::Int32, 1);
                work.wait();
                expect(!work.has_error(), "CUDA external receiver must not error: " + work.error());
            },
            [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, host_src2.data(), bytes, 0};
                auto work = comm.send(view, count, tbccl::DataType::Int32, 0);
                work.wait();
                expect(!work.has_error(), "host sender must not error: " + work.error());
            });

        tbccl_bench::tensor::cuda_copy_device_to_host(device_ptr, host_dst.data(), bytes);
        expect(std::memcmp(host_src2.data(), host_dst.data(), bytes) == 0, "host -> CUDA external must reproduce exact bytes (GPU readback)");

        // Part AV: GPU consumer kernel directly over the received device
        // buffer, proving genuine GPU visibility (not merely host-readback
        // correctness).
        void *consumer_out = nullptr;
        check_cuda(cudaMalloc(&consumer_out, bytes), "cudaMalloc (consumer output)");
        tbccl_bench::tensor::cuda_external_launch_consumer_double_i32(device_ptr, consumer_out, count, nullptr);
        std::vector<std::int32_t> consumer_host(count);
        tbccl_bench::tensor::cuda_copy_device_to_host(consumer_out, consumer_host.data(), bytes);
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::int32_t expected = static_cast<std::int32_t>(static_cast<std::uint32_t>(host_src2[i]) * 2u);
            expect(consumer_host[i] == expected, "GPU consumer kernel result mismatch at i=" + std::to_string(i));
        }
        check_cuda(cudaFree(consumer_out), "cudaFree (consumer output)");
    }

    // Communicators from both run_pair() calls above are already
    // destroyed (out of scope) -- device_ptr must still be exactly
    // once-freeable by the ORIGINAL caller now.
    check_cuda(cudaFree(device_ptr), "cudaFree (external test buffer, caller-owned)");
    std::cout << "[PASS] test_external_cuda_pointer_ownership_and_roundtrip\n";
}

// ---------------------------------------------------------------------
// Part AT/AU: delayed-producer CUDA stream dependency correctness.
// ---------------------------------------------------------------------

void test_delayed_producer_stream_dependency()
{
    const std::size_t count = 1 << 16; // large enough that a missing
                                        // dependency would very likely
                                        // read stale/zero data, not
                                        // coincidentally-correct data.
    const std::size_t bytes = count * sizeof(std::int32_t);
    const std::int32_t written_value = 0x1234;

    void *device_ptr = nullptr;
    check_cuda(cudaMalloc(&device_ptr, bytes), "cudaMalloc (delayed producer test buffer)");
    check_cuda(cudaMemset(device_ptr, 0, bytes), "cudaMemset (zero before delayed write)");

    // Launch a kernel that busy-spins substantially before writing
    // `written_value`, on its own stream, WITHOUT synchronizing --
    // submission to Communicator::send() happens immediately after,
    // while the kernel is (almost certainly) still spinning. If TBCCL's
    // stream-wait mechanism is broken (e.g. ignores the supplied
    // stream), the D2H would very likely race ahead and read the
    // zeroed buffer instead of written_value.
    void *producer_stream = tbccl_bench::tensor::cuda_external_test_launch_delayed_write_i32(
        device_ptr, count, written_value, /*spin_iterations=*/30'000'000ull);

    std::vector<std::int32_t> recv_buf(count, -1);
    run_pair(
        kBasePort + 10,
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Cuda, device_ptr, bytes, 0};
            tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, producer_stream};
            auto work = comm.send(view, count, tbccl::DataType::Int32, 1, ctx);
            work.wait();
            expect(!work.has_error(), "delayed-producer sender must not error: " + work.error());
        },
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Host, recv_buf.data(), bytes, 0};
            auto work = comm.recv(view, count, tbccl::DataType::Int32, 0);
            work.wait();
            expect(!work.has_error(), "delayed-producer receiver must not error: " + work.error());
        });

    for (std::size_t i = 0; i < count; ++i)
    {
        expect(recv_buf[static_cast<std::size_t>(i)] == written_value,
               "stream dependency not enforced: expected delayed write to be visible, got stale/zero data at i=" + std::to_string(i));
    }

    tbccl_bench::tensor::cuda_external_test_destroy_stream(producer_stream);
    check_cuda(cudaFree(device_ptr), "cudaFree (delayed producer test buffer)");
    std::cout << "[PASS] test_delayed_producer_stream_dependency\n";
}

// ---------------------------------------------------------------------
// Part AS/AR: CUDA<->Host AllReduce through the public API.
// ---------------------------------------------------------------------

void test_cuda_host_all_reduce()
{
    const std::size_t count = 8192;
    const std::size_t bytes = count * sizeof(float);

    void *device_ptr = nullptr;
    check_cuda(cudaMalloc(&device_ptr, bytes), "cudaMalloc (allreduce test buffer)");
    std::vector<float> cuda_local(count);
    for (std::size_t i = 0; i < count; ++i) cuda_local[static_cast<std::size_t>(i)] = static_cast<float>(i) * 2.0f + 1.0f;
    tbccl_bench::tensor::cuda_copy_host_to_device(cuda_local.data(), device_ptr, bytes);

    std::vector<float> host_local(count);
    for (std::size_t i = 0; i < count; ++i) host_local[static_cast<std::size_t>(i)] = static_cast<float>(i) * -0.5f + 4.0f;

    run_pair(
        kBasePort + 20,
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Cuda, device_ptr, bytes, 0};
            auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            work.wait();
            expect(!work.has_error(), "CUDA rank0 (root) all_reduce must not error: " + work.error());
        },
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Host, host_local.data(), bytes, 0};
            auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            work.wait();
            expect(!work.has_error(), "host rank1 all_reduce must not error: " + work.error());
        });

    std::vector<float> cuda_result(count);
    tbccl_bench::tensor::cuda_copy_device_to_host(device_ptr, cuda_result.data(), bytes);

    for (std::size_t i = 0; i < count; ++i)
    {
        const float expected = (static_cast<float>(i) * 2.0f + 1.0f) + (static_cast<float>(i) * -0.5f + 4.0f);
        expect(cuda_result[i] == expected, "CUDA-root result mismatch at i=" + std::to_string(i));
        expect(host_local[i] == expected, "host non-root result mismatch at i=" + std::to_string(i));
    }

    check_cuda(cudaFree(device_ptr), "cudaFree (allreduce test buffer)");
    std::cout << "[PASS] test_cuda_host_all_reduce\n";
}

} // namespace

int main()
{
    tbccl_bench::tensor::register_cuda_memory_provider();
    try
    {
        test_external_cuda_pointer_ownership_and_roundtrip();
        test_delayed_producer_stream_dependency();
        test_cuda_host_all_reduce();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All communicator CUDA tests passed.\n";
    return 0;
}
