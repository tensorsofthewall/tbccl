// The N-rank full mesh. world_size 1..4 initialize through the same Communicator::create() with explicit control/data endpoints, keep a
// per-rank capability table, close every socket exactly once across repeated create/destroy loops, and reject a bad handshake (world-size
// mismatch, wrong communicator id, duplicate rank) instead of hanging. Dynamic ports only.

#include "mesh_test_support.hpp"

#include <tbccl/tcp.hpp>

#include <iostream>

using namespace mesh_test;

namespace
{

    bool has(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

    void test_init_each_world_size()
    {
        for (std::size_t n = 1; n <= 4; ++n)
        {
            run_world(n, [n](std::size_t rank, tbccl::Communicator &comm) {
                expect(comm.rank() == rank && comm.world_size() == n, "rank/world_size accessors");
                expect(comm.capabilities().world_size() == n, "capability table has one entry per rank");
                for (std::size_t r = 0; r < n; ++r)
                {
                    const auto &caps = comm.capabilities().for_rank(r);
                    expect(!caps.memory_backends.empty() && caps.protocol_version == tbccl::kProtocolVersion, "rank capabilities were exchanged");
                }
                expect(comm.capabilities().negotiation().ok, "aggregate negotiation ok");
                expect(comm.capabilities().supports_memory_kind(tbccl::MemoryKind::Host), "host memory is supported");
                expect(!comm.failed() && !comm.aborted(), "a fresh communicator is healthy");
            });
            std::cout << "[PASS] init world_size=" << n << "\n";
        }
    }

    void test_no_fd_leak_across_create_destroy()
    {
        for (std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{4}})
        {
            run_world(n, [](std::size_t, tbccl::Communicator &) {}); // warm-up: lazily opened process-wide fds
            const int before = open_fd_count();
            for (int i = 0; i < 12; ++i) run_world(n, [](std::size_t, tbccl::Communicator &) {});
            const int after = open_fd_count();
            expect(after == before, "world_size " + std::to_string(n) + " create/destroy loop leaked descriptors: " + std::to_string(before) + " -> " + std::to_string(after));
        }
        std::cout << "[PASS] repeated create/destroy closes every descriptor exactly once (world_size 2, 3, 4)\n";
    }

    void test_world_size_one_opens_nothing()
    {
        tbccl::CommunicatorOptions o;
        o.rank = 0;
        o.world_size = 1;
        o.rank_directory.entries = {{0, {"127.0.0.1", 1}, {"127.0.0.1", 2}}};
        const int before = open_fd_count();
        {
            auto comm = tbccl::Communicator::create(o);
            expect(comm->world_size() == 1, "world_size 1");
            expect(open_fd_count() == before, "world_size 1 opens no sockets");
        }
        std::cout << "[PASS] world_size 1 opens no sockets\n";
    }

    void test_world_size_mismatch()
    {
        // Rank 0 and 1 believe the world has 3 ranks, rank 2 believes it has 4.
        auto id = tbccl::CommunicatorId::generate();
        auto results = bootstrap({{0, 3, id}, {1, 3, id}, {2, 4, id}}, std::chrono::milliseconds(2500));
        bool detected = false;
        for (const auto &r : results)
        {
            expect(r.comm == nullptr, "no rank may come up on a world-size mismatch");
            expect(!r.error.empty(), "every rank reports an error");
            detected = detected || has(r.error, "world_size");
        }
        expect(detected, "the mismatch is named: " + results[0].error + " | " + results[1].error + " | " + results[2].error);
        std::cout << "[PASS] world-size mismatch fails initialization on every rank\n";
    }

    void test_wrong_communicator_id()
    {
        auto id = tbccl::CommunicatorId::generate();
        auto other = tbccl::CommunicatorId::generate();
        auto results = bootstrap({{0, 3, id}, {1, 3, id}, {2, 3, other}}, std::chrono::milliseconds(2500));
        bool detected = false;
        for (const auto &r : results)
        {
            expect(r.comm == nullptr, "no rank may come up with a foreign communicator id");
            detected = detected || has(r.error, "communicator id") || has(r.error, "communicator ");
        }
        expect(detected, "the foreign id is named: " + results[0].error + " | " + results[2].error);
        std::cout << "[PASS] wrong communicator id is rejected\n";
    }

    void test_duplicate_rank()
    {
        auto id = tbccl::CommunicatorId::generate();
        // Two processes claim rank 1; nobody claims rank 2.
        auto results = bootstrap({{0, 3, id}, {1, 3, id}, {1, 3, id}}, std::chrono::milliseconds(2500));
        bool duplicate = false;
        for (const auto &r : results)
        {
            expect(r.comm == nullptr, "no rank may come up while a rank is claimed twice and another is missing");
            duplicate = duplicate || has(r.error, "duplicate rank");
        }
        expect(duplicate, "the duplicate claim is named: " + results[0].error + " | " + results[1].error + " | " + results[2].error);
        std::cout << "[PASS] duplicate rank is rejected\n";
    }

    void test_missing_rank_times_out()
    {
        auto id = tbccl::CommunicatorId::generate();
        const auto start = std::chrono::steady_clock::now();
        auto results = bootstrap({{0, 3, id}, {1, 3, id}}, std::chrono::milliseconds(1200)); // rank 2 never starts
        const auto elapsed = std::chrono::steady_clock::now() - start;
        for (const auto &r : results) expect(r.comm == nullptr && has(r.error, "timeout"), "a missing rank is a bounded timeout: " + r.error);
        expect(elapsed < std::chrono::seconds(6), "the bootstrap deadline is honoured");
        std::cout << "[PASS] a missing rank times out within the bootstrap deadline\n";
    }

    void test_legacy_peers_still_work_for_two()
    {
        // The pre-N-rank-runtime `peers` list: control = peers[r], data = peers[0].port + 1000. Ports probed free, then
        // released.
        auto probe = [] {
            for (std::uint16_t p = 21000; p < 21900; p += 7)
            {
                try
                {
                    auto a = tbccl::tcp_listen("127.0.0.1", p, {});
                    auto b = tbccl::tcp_listen("127.0.0.1", static_cast<std::uint16_t>(p + 1000), {});
                    return p;
                }
                catch (const std::exception &)
                {
                }
            }
            throw std::runtime_error("no free legacy port pair");
        };
        const auto base = probe();
        std::unique_ptr<tbccl::Communicator> c0, c1;
        std::string e0, e1;
        std::thread t([&] {
            try
            {
                tbccl::CommunicatorOptions o;
                o.rank = 1;
                o.peers = {{"127.0.0.1", base}, {"127.0.0.1", static_cast<std::uint16_t>(base + 1)}};
                c1 = tbccl::Communicator::create(o);
            }
            catch (const std::exception &e)
            {
                e1 = e.what();
            }
        });
        try
        {
            tbccl::CommunicatorOptions o;
            o.rank = 0;
            o.peers = {{"127.0.0.1", base}, {"127.0.0.1", static_cast<std::uint16_t>(base + 1)}};
            c0 = tbccl::Communicator::create(o);
        }
        catch (const std::exception &e)
        {
            e0 = e.what();
        }
        t.join();
        expect(c0 && c1, "legacy peers world of two: " + e0 + e1);
        std::cout << "[PASS] legacy peers bootstrap (world_size 2) still works\n";
    }

} // namespace

int main()
{
    try
    {
        test_world_size_one_opens_nothing();
        test_init_each_world_size();
        test_no_fd_leak_across_create_destroy();
        test_world_size_mismatch();
        test_wrong_communicator_id();
        test_duplicate_rank();
        test_missing_rank_times_out();
        test_legacy_peers_still_work_for_two();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All communicator mesh tests passed.\n";
    return 0;
}
