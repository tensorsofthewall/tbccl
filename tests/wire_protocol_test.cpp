// Phase 50: the Communicator handshake and control frames over real loopback sockets (ports are allocated by the kernel).
// Every rejection must surface as "protocol_mismatch: ..." on the accepting AND the dialing side, and a stranger that never
// completes a hello must not stall the acceptor past its io timeout.

#include "wire_protocol.hpp"

#include <tbccl/tcp.hpp>

#include "test_utils.hpp"

#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <vector>

using tbccl_test::expect;
using namespace tbccl;
using namespace tbccl::detail;

namespace
{

    struct Outcome
    {
        std::string dial_error;
        std::string accept_error;
        Hello accepted;
    };

    Hello hello(const CommunicatorId &id, std::size_t rank, std::size_t world, ConnectionRole role = ConnectionRole::Control)
    {
        Hello h;
        h.communicator_id = id;
        h.rank = static_cast<std::uint32_t>(rank);
        h.world_size = static_cast<std::uint32_t>(world);
        h.role = role;
        return h;
    }

    Outcome run(const Hello &dialer, std::size_t dialed_rank, const AcceptExpectation &expect_in)
    {
        auto listener = tcp_listen("127.0.0.1", 0);
        const auto port = listener->local_port();
        expect(port != 0, "listener reports its ephemeral port");
        Outcome out;
        auto acceptor = std::async(std::launch::async, [&] {
            try
            {
                auto c = listener->accept_for(std::chrono::seconds(5));
                expect(c != nullptr, "a connection arrived");
                c->set_io_timeout(std::chrono::seconds(2));
                out.accepted = accept_handshake(*c, expect_in);
            }
            catch (const std::exception &e)
            {
                out.accept_error = e.what();
            }
        });
        try
        {
            auto c = tcp_connect("127.0.0.1", port);
            c->set_io_timeout(std::chrono::seconds(2));
            dial_handshake(*c, dialer, dialed_rank);
        }
        catch (const std::exception &e)
        {
            out.dial_error = e.what();
        }
        acceptor.get();
        return out;
    }

    bool has(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

    void expect_rejected(const Outcome &o, const std::string &needle, const std::string &label)
    {
        expect(has(o.accept_error, "protocol_mismatch:") && has(o.accept_error, needle), label + ": acceptor error was '" + o.accept_error + "'");
        expect(has(o.dial_error, "protocol_mismatch:") && has(o.dial_error, needle), label + ": dialer error was '" + o.dial_error + "'");
    }

} // namespace

int main()
{
    const auto id = CommunicatorId::generate();
    const auto other = CommunicatorId::generate();
    AcceptExpectation base;
    base.communicator_id = id;
    base.local_rank = 0;
    base.world_size = 3;
    base.role = ConnectionRole::Control;

    {
        auto o = run(hello(id, 1, 3), 0, base);
        expect(o.dial_error.empty() && o.accept_error.empty(), "valid handshake: " + o.dial_error + o.accept_error);
        expect(o.accepted.rank == 1 && o.accepted.world_size == 3 && o.accepted.role == ConnectionRole::Control, "accepted hello carries the dialer identity");
    }
    {
        auto o = run(hello(id, 2, 3, ConnectionRole::Data), 0, [&] { auto e = base; e.role = ConnectionRole::Data; return e; }());
        expect(o.dial_error.empty() && o.accept_error.empty() && o.accepted.role == ConnectionRole::Data, "valid data handshake");
    }
    expect_rejected(run(hello(other, 1, 3), 0, base), "communicator id", "wrong communicator id");
    expect_rejected(run(hello(id, 1, 4), 0, base), "world_size", "world size mismatch");
    expect_rejected(run(hello(id, 9, 3), 0, base), "outside", "rank >= world_size");
    expect_rejected(run(hello(id, 1, 3, ConnectionRole::Data), 0, base), "connection arrived", "role mismatch");
    {
        auto h = hello(id, 1, 3);
        h.wire_version = 1;
        expect_rejected(run(h, 0, base), "wire protocol", "wire protocol version mismatch");
    }
    {
        auto e = base;
        e.local_rank = 2;
        expect_rejected(run(hello(id, 1, 3), 2, e), "lower rank dials", "a lower rank must not connect to a higher one's listener the wrong way round");
    }
    {
        std::vector<bool> connected = {false, true, false};
        auto e = base;
        e.already_connected = &connected;
        expect_rejected(run(hello(id, 1, 3), 0, e), "duplicate rank 1", "duplicate rank");
    }
    {
        // dialer connected to the right listener but believes it dialed rank 2
        auto o = run(hello(id, 1, 3), 2, base);
        expect(has(o.dial_error, "protocol_mismatch:") && has(o.dial_error, "identifies as rank 0"), "dialer detects the wrong peer rank: " + o.dial_error);
    }

    // A stranger: garbage, an old-style short hello, silence. The acceptor never blocks past its io timeout.
    for (int kind = 0; kind < 3; ++kind)
    {
        auto listener = tcp_listen("127.0.0.1", 0);
        const auto port = listener->local_port();
        std::string error;
        const auto start = std::chrono::steady_clock::now();
        auto acceptor = std::async(std::launch::async, [&] {
            try
            {
                auto c = listener->accept_for(std::chrono::seconds(5));
                c->set_io_timeout(std::chrono::milliseconds(300));
                accept_handshake(*c, base);
            }
            catch (const std::exception &e)
            {
                error = e.what();
            }
        });
        auto c = tcp_connect("127.0.0.1", port);
        if (kind == 0)
        {
            std::vector<unsigned char> junk(tbccl::detail::kHelloWireSize, 0x7f);
            c->send(junk.data(), junk.size());
        }
        else if (kind == 1)
        {
            const unsigned char old_hello[24] = {0x54, 0x42, 0x43, 0x4c};
            c->send(old_hello, sizeof(old_hello));
        }
        acceptor.get();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        expect(has(error, "protocol_mismatch: not a TBCCL peer"), "stranger kind " + std::to_string(kind) + " is rejected: " + error);
        expect(elapsed < std::chrono::seconds(3), "stranger kind " + std::to_string(kind) + " did not stall the acceptor");
        if (kind == 2) expect(has(error, "timeout"), "silence is reported as a timeout: " + error);
    }

    // Control frames
    {
        auto listener = tcp_listen("127.0.0.1", 0);
        auto client = tcp_connect("127.0.0.1", listener->local_port());
        auto server = listener->accept_for(std::chrono::seconds(5));
        ControlFrame f;
        f.type = ControlFrameType::Abort;
        f.origin_rank = 3;
        f.reason = "rank 3 failed: " + std::string(600, 'x');
        send_control_frame(*client, f);
        auto g = recv_control_frame(*server);
        expect(g.type == ControlFrameType::Abort && g.origin_rank == 3, "control frame header round trip");
        expect(g.reason.size() == kControlReasonBytes - 1 && g.reason.rfind("rank 3 failed: ", 0) == 0, "long reasons are truncated, not rejected");
        f.type = ControlFrameType::Goodbye;
        f.reason.clear();
        send_control_frame(*server, f);
        expect(recv_control_frame(*client).type == ControlFrameType::Goodbye, "goodbye round trip");
    }

    std::cout << "wire_protocol_test passed\n";
    return 0;
}
