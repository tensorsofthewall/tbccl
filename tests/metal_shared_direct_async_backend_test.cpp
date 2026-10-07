// Correctness tests for MetalSharedDirectAsyncBackend.
//
// Two kinds of coverage:
//  - Portable tests (build on any platform): constructor rejects a
//    non-MetalShared backend; a generic asymmetric chunk-plan proof using
//    HostAsyncBackend (sender explicitly chunked, receiver full-buffer
//    direct) that validates the wire-compatibility claim without
//    requiring real Metal hardware.
//  - TBCCL_ENABLE_METAL-only tests: real MetalShared roundtrips over TCP
//    loopback, direct pointer identity, odd sizes, repeated reuse.

#include "tensor/host_async_backend.hpp"
#include "tensor/metal_shared_direct_async_backend.hpp"
#include "tensor/tensor_backend.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

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

using tbccl_bench::tensor::BackendKind;
using tbccl_bench::tensor::HostAsyncBackend;
using tbccl_bench::tensor::make_backend;
using tbccl_bench::tensor::MetalSharedDirectAsyncBackend;
using tbccl_bench::tensor::pattern_byte;
using tbccl_bench::tensor::TensorBackend;

constexpr std::uint16_t kBasePort = 28900;

void expect(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error("assertion failed: " + message);
    }
}

// ---------------------------------------------------------------------
// Portable: constructor rejects non-MetalShared backend.
// ---------------------------------------------------------------------
void test_wrong_kind_throws()
{
    auto backend = make_backend(BackendKind::Host);
    backend->allocate(1024);
    bool threw = false;
    try
    {
        MetalSharedDirectAsyncBackend wrapper(*backend);
        (void)wrapper;
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    expect(threw, "constructing with a non-MetalShared backend must throw");
    std::cout << "[PASS] test_wrong_kind_throws\n";
}

// ---------------------------------------------------------------------
// Portable: asymmetric chunk-plan proof, using HostAsyncBackend on both
// ends since both support direct access -- sender explicitly requests
// chunk_hint=65536 (genuinely chunked/pipelined on its side), receiver
// requests chunk_hint=0 (single whole-buffer direct recv). Proves a
// sender's local chunking decision is invisible to -- and does not need to
// match -- the receiver's, over a real TCP connection.
// ---------------------------------------------------------------------
void test_asymmetric_chunk_plan()
{
    const std::uint16_t port = kBasePort;
    const std::size_t bytes = 3 * 1024 * 1024 + 777; // deliberately not chunk-aligned
    const std::uint32_t seed = 0xA5A5A5A5u;

    auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

    std::vector<std::byte> source(bytes);
    for (std::size_t i = 0; i < bytes; ++i)
    {
        source[i] = static_cast<std::byte>(pattern_byte(i, seed));
    }
    std::vector<std::byte> destination(bytes, std::byte{0});

    HostAsyncBackend sender_backend(source.data(), bytes);
    HostAsyncBackend receiver_backend(destination.data(), bytes);

    std::thread server_thread([&]()
    {
        auto connection = listener->accept();
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker;

        tbccl::TransferRequest request;
        request.direction = tbccl::TransferDirection::Recv;
        request.backend = &receiver_backend;
        request.transport = &transport;
        request.total_bytes = bytes;
        request.chunk_hint = 0; // full-buffer direct recv

        auto work = worker.enqueue(request);
        work.wait();
        expect(!work.has_error(), "receiver must not error: " + work.error());
    });

    auto connection = tbccl::tcp_connect("127.0.0.1", port, {});
    tbccl::TcpTransport client_transport(std::move(connection));
    tbccl::TensorCommWorker client_worker;

    tbccl::TransferRequest send_request;
    send_request.direction = tbccl::TransferDirection::Send;
    send_request.backend = &sender_backend;
    send_request.transport = &client_transport;
    send_request.total_bytes = bytes;
    send_request.chunk_hint = 65536; // sender genuinely chunked/pipelined

    auto send_work = client_worker.enqueue(send_request);
    send_work.wait();
    expect(!send_work.has_error(), "sender must not error: " + send_work.error());

    server_thread.join();

    expect(std::memcmp(source.data(), destination.data(), bytes) == 0,
           "asymmetric chunk-plan transfer must reproduce exact bytes");

    std::cout << "[PASS] test_asymmetric_chunk_plan\n";
}

#if defined(TBCCL_ENABLE_METAL)

// Shared helper: full send/recv roundtrip for MetalShared through
// MetalSharedDirectAsyncBackend on both ends, verifying exact bytes.
void run_metal_direct_roundtrip(std::uint16_t port, std::size_t bytes, std::uint32_t seed)
{
    auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

    auto source_backend = make_backend(BackendKind::MetalShared);
    source_backend->allocate(bytes);
    source_backend->initialize_source(seed);
    source_backend->prepare_source();
    source_backend->stage_device_to_host(); // no-op for MetalShared, called for contract parity

    auto destination_backend = make_backend(BackendKind::MetalShared);
    destination_backend->allocate(bytes);

    MetalSharedDirectAsyncBackend source_direct(*source_backend);
    MetalSharedDirectAsyncBackend destination_direct(*destination_backend);

    // Direct pointer identity: direct_source_data()/
    // direct_destination_data() must return exactly what the underlying
    // TensorBackend already exposes as its staging/contents pointer.
    expect(source_direct.direct_source_data() == source_backend->source_staging_data(),
           "direct_source_data must equal backend source_staging_data");
    expect(destination_direct.direct_destination_data() == destination_backend->destination_staging_data(),
           "direct_destination_data must equal backend destination_staging_data");

    std::thread server_thread([&]()
    {
        auto connection = listener->accept();
        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker worker;

        tbccl::TransferRequest request;
        request.direction = tbccl::TransferDirection::Recv;
        request.backend = &destination_direct;
        request.transport = &transport;
        request.total_bytes = bytes;
        request.chunk_hint = 0;

        auto work = worker.enqueue(request);
        work.wait();
        expect(!work.has_error(), "receiver must not error: " + work.error());
    });

    auto connection = tbccl::tcp_connect("127.0.0.1", port, {});
    tbccl::TcpTransport client_transport(std::move(connection));
    tbccl::TensorCommWorker client_worker;

    tbccl::TransferRequest send_request;
    send_request.direction = tbccl::TransferDirection::Send;
    send_request.backend = &source_direct;
    send_request.transport = &client_transport;
    send_request.total_bytes = bytes;
    send_request.chunk_hint = 0;

    auto send_work = client_worker.enqueue(send_request);
    send_work.wait();
    expect(!send_work.has_error(), "sender must not error: " + send_work.error());

    server_thread.join();

    destination_backend->stage_host_to_device(); // no-op for MetalShared, called for contract parity
    expect(destination_backend->verify_destination(seed),
           "destination must verify byte-for-byte against source seed");
}

void test_metal_direct_roundtrip()
{
    if (!tbccl_bench::tensor::backend_kind_available(BackendKind::MetalShared))
    {
        std::cout << "[SKIP] test_metal_direct_roundtrip (no Metal device at runtime)\n";
        return;
    }
    run_metal_direct_roundtrip(kBasePort + 1, 4 << 20, 0xC0FFEEu);
    std::cout << "[PASS] test_metal_direct_roundtrip\n";
}

void test_metal_direct_odd_sizes()
{
    if (!tbccl_bench::tensor::backend_kind_available(BackendKind::MetalShared))
    {
        std::cout << "[SKIP] test_metal_direct_odd_sizes (no Metal device at runtime)\n";
        return;
    }
    const std::vector<std::size_t> sizes = {
        1,
        4096,
        (256 * 1024) - 1,
        256 * 1024,
        (256 * 1024) + 1,
        1 << 20,
        1234567, // non-aligned
    };
    std::uint16_t port = kBasePort + 2;
    for (std::size_t bytes : sizes)
    {
        run_metal_direct_roundtrip(port++, bytes, 0xD00Du + static_cast<std::uint32_t>(bytes));
    }
    std::cout << "[PASS] test_metal_direct_odd_sizes\n";
}

void test_metal_direct_repeated_reuse()
{
    if (!tbccl_bench::tensor::backend_kind_available(BackendKind::MetalShared))
    {
        std::cout << "[SKIP] test_metal_direct_repeated_reuse (no Metal device at runtime)\n";
        return;
    }
    const std::size_t bytes = 1 << 18;
    std::uint16_t port = kBasePort + 20;
    for (int round = 0; round < 5; ++round)
    {
        run_metal_direct_roundtrip(port++, bytes, 0x1000u + static_cast<std::uint32_t>(round));
    }
    std::cout << "[PASS] test_metal_direct_repeated_reuse\n";
}

#endif // TBCCL_ENABLE_METAL

} // namespace

int main()
{
    try
    {
        test_wrong_kind_throws();
        test_asymmetric_chunk_plan();
#if defined(TBCCL_ENABLE_METAL)
        test_metal_direct_roundtrip();
        test_metal_direct_odd_sizes();
        test_metal_direct_repeated_reuse();
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All metal_shared_direct_async_backend tests passed.\n";
    return 0;
}
