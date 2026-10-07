// Proves a genuinely externally-owned, CPU-visible
// MTLBuffer (MTLResourceStorageModeShared) can participate in the
// public Communicator API tagged as MemoryKind::MetalShared, with TBCCL
// never allocating or freeing it -- the architecture doc's "Metal
// simplification" (docs/framework_integration_architecture.md Section
// 5) validated against real hardware, not just the Host-only loopback
// suite in communicator_test.cpp.

#import <Metal/Metal.h>

#include <tbccl/communicator.hpp>

#include <cstdint>
#include <cstring>
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

constexpr std::uint16_t kBasePort = 29200;

struct MetalSharedBuffer
{
    id<MTLDevice> device;
    id<MTLBuffer> buffer;

    explicit MetalSharedBuffer(std::size_t bytes)
    {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no Metal device available");
        buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("MTLBuffer allocation failed");
    }

    void *contents() const { return buffer.contents; }
};

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

// A genuine MTLBuffer source (rank0) sends to a plain host-malloc'd
// destination (rank1) -- proves TBCCL correctly reads from real
// Metal-shared memory via the direct-access path (Host/MetalShared both
// register supports_direct_transport_access()==true).
void test_metal_source_to_host_destination()
{
    const std::size_t count = 65536;
    const std::size_t bytes = count * sizeof(std::int32_t);

    MetalSharedBuffer metal_buf(bytes);
    auto *src = static_cast<std::uint8_t *>(metal_buf.contents());
    for (std::size_t i = 0; i < bytes; ++i) src[i] = static_cast<std::uint8_t>((i * 131u + 0x5A) & 0xffu);

    std::vector<std::uint8_t> dst(bytes, 0xCC);

    run_pair(
        kBasePort,
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::MetalShared, metal_buf.contents(), bytes, 0};
            auto work = comm.send(view, count, tbccl::DataType::Int32, 1);
            work.wait();
            expect(!work.has_error(), "Metal-shared sender must not error: " + work.error());
        },
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Host, dst.data(), bytes, 0};
            auto work = comm.recv(view, count, tbccl::DataType::Int32, 0);
            work.wait();
            expect(!work.has_error(), "host receiver must not error: " + work.error());
        });

    expect(std::memcmp(metal_buf.contents(), dst.data(), bytes) == 0, "Metal-shared source bytes must match exactly");
    // TBCCL never freed metal_buf.buffer -- it is still a live, valid
    // ARC-managed Objective-C object here; destructor runs normally at
    // scope exit, proving TBCCL took no ownership of it.
    std::cout << "[PASS] test_metal_source_to_host_destination\n";
}

// A genuine MTLBuffer destination (rank0) receives from a host source
// (rank1) -- the symmetric direction.
void test_host_source_to_metal_destination()
{
    const std::size_t count = 65536;
    const std::size_t bytes = count * sizeof(std::int32_t);

    std::vector<std::uint8_t> src(bytes);
    for (std::size_t i = 0; i < bytes; ++i) src[i] = static_cast<std::uint8_t>((i * 97u + 0x33) & 0xffu);

    MetalSharedBuffer metal_buf(bytes);
    std::memset(metal_buf.contents(), 0, bytes);

    run_pair(
        kBasePort + 10,
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::MetalShared, metal_buf.contents(), bytes, 0};
            auto work = comm.recv(view, count, tbccl::DataType::Int32, 1);
            work.wait();
            expect(!work.has_error(), "Metal-shared receiver must not error: " + work.error());
        },
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Host, src.data(), bytes, 0};
            auto work = comm.send(view, count, tbccl::DataType::Int32, 0);
            work.wait();
            expect(!work.has_error(), "host sender must not error: " + work.error());
        });

    expect(std::memcmp(src.data(), metal_buf.contents(), bytes) == 0, "Metal-shared destination bytes must match exactly");
    std::cout << "[PASS] test_host_source_to_metal_destination\n";
}

// Real AllReduce with one rank's buffer genuinely backed by MTLBuffer
// (in-place, send_buf==recv_buf -- the common usage).
void test_metal_all_reduce()
{
    const std::size_t count = 4096;
    const std::size_t bytes = count * sizeof(float);

    MetalSharedBuffer metal_buf(bytes);
    auto *metal_floats = static_cast<float *>(metal_buf.contents());
    for (std::size_t i = 0; i < count; ++i) metal_floats[i] = static_cast<float>(i) * 2.0f + 3.0f;

    std::vector<float> host_data(count);
    for (std::size_t i = 0; i < count; ++i) host_data[i] = static_cast<float>(i) * -1.0f + 10.0f;

    run_pair(
        kBasePort + 20,
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::Host, host_data.data(), bytes, 0};
            auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            work.wait();
            expect(!work.has_error(), "rank0 (host) all_reduce must not error: " + work.error());
        },
        [&](tbccl::Communicator &comm) {
            tbccl::BufferView view{tbccl::MemoryKind::MetalShared, metal_buf.contents(), bytes, 0};
            auto work = comm.all_reduce(view, view, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            work.wait();
            expect(!work.has_error(), "rank1 (metal-shared) all_reduce must not error: " + work.error());
        });

    for (std::size_t i = 0; i < count; ++i)
    {
        const float expected = (static_cast<float>(i) * -1.0f + 10.0f) + (static_cast<float>(i) * 2.0f + 3.0f);
        expect(host_data[i] == expected, "host result mismatch at i=" + std::to_string(i));
        expect(metal_floats[i] == expected, "metal result mismatch at i=" + std::to_string(i));
    }
    std::cout << "[PASS] test_metal_all_reduce\n";
}

// 5x repeated reuse of the same MTLBuffer with distinct content each
// round.
void test_metal_repeated_reuse()
{
    const std::size_t count = 1024;
    const std::size_t bytes = count * sizeof(std::int32_t);
    MetalSharedBuffer metal_buf(bytes);

    run_pair(
        kBasePort + 30,
        [&](tbccl::Communicator &comm) {
            for (int round = 0; round < 5; ++round)
            {
                std::vector<std::uint8_t> src(bytes, static_cast<std::uint8_t>(round + 1));
                tbccl::BufferView view{tbccl::MemoryKind::Host, src.data(), bytes, 0};
                auto work = comm.send(view, count, tbccl::DataType::Int32, 1);
                work.wait();
                expect(!work.has_error(), "round " + std::to_string(round) + " host sender must not error");
            }
        },
        [&](tbccl::Communicator &comm) {
            for (int round = 0; round < 5; ++round)
            {
                std::memset(metal_buf.contents(), 0, bytes);
                tbccl::BufferView view{tbccl::MemoryKind::MetalShared, metal_buf.contents(), bytes, 0};
                auto work = comm.recv(view, count, tbccl::DataType::Int32, 0);
                work.wait();
                expect(!work.has_error(), "round " + std::to_string(round) + " metal receiver must not error");
                auto *bytes_ptr = static_cast<std::uint8_t *>(metal_buf.contents());
                for (std::size_t i = 0; i < bytes; ++i)
                {
                    expect(bytes_ptr[i] == static_cast<std::uint8_t>(round + 1),
                           "round " + std::to_string(round) + " stale data detected");
                }
            }
        });
    std::cout << "[PASS] test_metal_repeated_reuse\n";
}

} // namespace

int main()
{
    try
    {
        test_metal_source_to_host_destination();
        test_host_source_to_metal_destination();
        test_metal_all_reduce();
        test_metal_repeated_reuse();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All communicator Metal tests passed.\n";
    return 0;
}
