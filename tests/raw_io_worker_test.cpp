// Phase 34 Part AP: tests for RawIoWorker (benchmarks/tensor/raw_io_worker.hpp),
// the minimal diagnostic-only background-thread control used to test
// whether generic thread execution context (as opposed to anything
// TensorCommWorker-specific) explains the Phase 33 async throughput
// regression. Exercises RawIoWorker directly against a real loopback
// TcpTransport -- no TensorCommWorker/StagingPool involved.

#include "../benchmarks/tensor/raw_io_worker.hpp"

#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <chrono>
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
    constexpr std::uint16_t kBasePort = 28820;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    // Establishes one loopback TcpTransport pair, running `body` with
    // (client_transport, server_transport) once both sides are
    // connected. Mirrors async_transfer_test.cpp's loopback pattern.
    void with_loopback_pair(
        std::uint16_t port,
        const std::function<void(tbccl::TcpTransport &, tbccl::TcpTransport &)> &body)
    {
        std::unique_ptr<tbccl::Connection> server_conn;
        std::thread server_thread([&]() {
            auto listener = tbccl::tcp_listen("127.0.0.1", port, {});
            server_conn = listener->accept();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto client_conn = tbccl::tcp_connect("127.0.0.1", port, {});
        server_thread.join();

        tbccl::TcpTransport client_transport(std::move(client_conn));
        tbccl::TcpTransport server_transport(std::move(server_conn));
        body(client_transport, server_transport);
    }

    void test_start_stop_no_ops()
    {
        with_loopback_pair(kBasePort, [](tbccl::TcpTransport &client, tbccl::TcpTransport &) {
            tbccl_bench::RawIoWorker worker(&client);
            (void)worker;
            // Destructor must join cleanly with zero ops ever submitted.
        });
        std::cout << "[PASS] test_start_stop_no_ops\n";
    }

    void test_single_send_recv_round_trip()
    {
        with_loopback_pair(
            kBasePort + 1,
            [](tbccl::TcpTransport &client, tbccl::TcpTransport &server) {
                tbccl_bench::RawIoWorker client_worker(&client);
                std::vector<std::uint8_t> src(4096, 0x7A);
                std::vector<std::uint8_t> dst(4096, 0);

                std::thread server_thread(
                    [&]() { server.recv(dst.data(), dst.size()); });
                client_worker.run_op(tbccl_bench::RawIoWorker::Op::Send, src.data(), src.size());
                server_thread.join();

                expect(dst == src, "single round-trip payload must match exactly");
            });
        std::cout << "[PASS] test_single_send_recv_round_trip\n";
    }

    void test_repeated_operations_same_worker()
    {
        with_loopback_pair(
            kBasePort + 2,
            [](tbccl::TcpTransport &client, tbccl::TcpTransport &server) {
                tbccl_bench::RawIoWorker client_worker(&client);
                constexpr int kRounds = 10;
                std::thread server_thread([&]() {
                    for (int round = 0; round < kRounds; ++round)
                    {
                        std::vector<std::uint8_t> dst(256, 0);
                        server.recv(dst.data(), dst.size());
                        for (auto b : dst)
                        {
                            expect(b == static_cast<std::uint8_t>(round),
                                   "repeated-op round content must match its own round index");
                        }
                    }
                });
                for (int round = 0; round < kRounds; ++round)
                {
                    std::vector<std::uint8_t> src(256, static_cast<std::uint8_t>(round));
                    client_worker.run_op(
                        tbccl_bench::RawIoWorker::Op::Send, src.data(), src.size());
                }
                server_thread.join();
            });
        std::cout << "[PASS] test_repeated_operations_same_worker\n";
    }

    void test_clean_shutdown_does_not_hang()
    {
        // A worker that has done real work still joins cleanly and
        // promptly on destruction -- this test itself hanging (rather
        // than a thrown assertion) is the failure mode it targets.
        with_loopback_pair(
            kBasePort + 3,
            [](tbccl::TcpTransport &client, tbccl::TcpTransport &server) {
                std::vector<std::uint8_t> src(64, 0x11);
                std::vector<std::uint8_t> dst(64, 0);
                std::thread server_thread([&]() { server.recv(dst.data(), dst.size()); });
                {
                    tbccl_bench::RawIoWorker client_worker(&client);
                    client_worker.run_op(
                        tbccl_bench::RawIoWorker::Op::Send, src.data(), src.size());
                }
                server_thread.join();
            });
        std::cout << "[PASS] test_clean_shutdown_does_not_hang\n";
    }

    void test_error_propagates_to_caller()
    {
        // Close the peer before the client sends; the resulting
        // exception must surface from run_op() on the CALLING thread,
        // not be silently swallowed inside the worker.
        std::unique_ptr<tbccl::Connection> server_conn;
        std::thread server_thread([&]() {
            auto listener = tbccl::tcp_listen("127.0.0.1", kBasePort + 5, {});
            server_conn = listener->accept();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto client_conn = tbccl::tcp_connect("127.0.0.1", kBasePort + 5, {});
        server_thread.join();
        server_conn.reset(); // close the peer before any data is sent

        tbccl::TcpTransport client_transport(std::move(client_conn));
        tbccl_bench::RawIoWorker client_worker(&client_transport);

        std::vector<std::uint8_t> src(1 << 20, 0x22); // large enough to force a write error
        bool threw = false;
        try
        {
            client_worker.run_op(
                tbccl_bench::RawIoWorker::Op::Send, src.data(), src.size());
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        expect(threw, "run_op must propagate a Transport error to the caller");
        std::cout << "[PASS] test_error_propagates_to_caller\n";
    }

} // namespace

int main()
{
    try
    {
        test_start_stop_no_ops();
        test_single_send_recv_round_trip();
        test_repeated_operations_same_worker();
        test_clean_shutdown_does_not_hang();
        test_error_propagates_to_caller();
        std::cout << "All tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
