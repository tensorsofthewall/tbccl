// The two-rank all_gather posts its two directions together (full duplex). This test pins what that must not break, over real loopback sockets with one
// Communicator per rank: the gathered contents of both ranks (every byte), payloads far beyond the socket buffers (both ranks send before either has received),
// the supported local-copy aliasing, ordering against the other collectives and against point-to-point traffic, resource stability over many collectives, and
// bounded termination under abort, a silent peer and a peer that goes away. The deterministic proof that the two legs really overlap is in n2_exchange_unit_test.
// Run once per TBCCL_STAGED_CHUNK_BYTES setting (the setting only affects non-direct memory, so host runs must be unaffected by it).

#include "communicator_debug.hpp"
#include "mesh_test_support.hpp"

#include <tbccl/communicator.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace mesh_test;
using namespace std::chrono_literals;

namespace
{
std::uint8_t pat(std::size_t rank, std::size_t seq, std::size_t j) { return static_cast<std::uint8_t>(rank * 61 + seq * 29 + j * 7 + (j >> 12) * 3 + 5); }

tbccl::BufferView view(std::vector<std::uint8_t> &v) { return {tbccl::MemoryKind::Host, v.data(), v.size(), -1}; }

void wait_ok(tbccl::Work work, const std::string &what)
{
    expect(work.wait_for(60s), what + ": the Work did not finish within 60 s");
    expect(!work.has_error(), what + ": " + work.error());
}

// Gathers `bytes` per rank and checks every byte of both slots.
void gather_and_check(tbccl::Communicator &comm, std::size_t rank, std::size_t bytes, std::size_t seq)
{
    std::vector<std::uint8_t> in(bytes), g0(bytes, 0xEE), g1(bytes, 0xEE);
    for (std::size_t j = 0; j < bytes; ++j) in[j] = pat(rank, seq, j);
    wait_ok(comm.all_gather(view(in), {view(g0), view(g1)}), "all_gather " + std::to_string(bytes));
    for (std::size_t j = 0; j < bytes; ++j)
    {
        expect(g0[j] == pat(0, seq, j), "slot 0 byte " + std::to_string(j) + " of " + std::to_string(bytes));
        expect(g1[j] == pat(1, seq, j), "slot 1 byte " + std::to_string(j) + " of " + std::to_string(bytes));
    }
}

// The input is its own local output (the supported alias): the local copy is skipped and the peer's slot is received.
void gather_in_place(tbccl::Communicator &comm, std::size_t rank, std::size_t bytes, std::size_t seq)
{
    std::vector<std::uint8_t> g0(bytes, 0xEE), g1(bytes, 0xEE);
    auto &mine = rank == 0 ? g0 : g1;
    for (std::size_t j = 0; j < bytes; ++j) mine[j] = pat(rank, seq, j);
    wait_ok(comm.all_gather(view(mine), {view(g0), view(g1)}), "in-place all_gather " + std::to_string(bytes));
    for (std::size_t j = 0; j < bytes; ++j)
    {
        expect(g0[j] == pat(0, seq, j), "in-place slot 0 byte " + std::to_string(j));
        expect(g1[j] == pat(1, seq, j), "in-place slot 1 byte " + std::to_string(j));
    }
}

long thread_count()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("Threads:", 0) == 0) return std::stol(line.substr(8));
    return -1;
}
} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(900s);
        std::fprintf(stderr, "STALLED: communicator_n2_all_gather_exchange_test exceeded 900 s\n");
        std::fflush(stderr);
        std::_Exit(124);
    }).detach();
    try
    {
        {
            World w(2);
            // payloads from nothing to well past the kernel socket buffers; repeated so an ordering error between consecutive collectives shows up
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                std::size_t seq = 0;
                for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{64}, std::size_t{4096}, std::size_t{100003},
                                          std::size_t{257 * 1024 + 1}, (std::size_t{1} << 20) + 1, std::size_t{4} << 20})
                    for (int i = 0; i < 20; ++i) gather_and_check(comm, rank, bytes, seq++);
                for (int i = 0; i < 3; ++i) gather_and_check(comm, rank, std::size_t{48} << 20, seq++); // both ranks send 48 MiB before either receives
            });
            std::printf("contents ok\n");
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                std::size_t seq = 1000;
                for (std::size_t bytes : {std::size_t{1}, std::size_t{4096}, std::size_t{100003}, std::size_t{4} << 20})
                    for (int i = 0; i < 5; ++i) gather_in_place(comm, rank, bytes, seq++);
            });
            std::printf("in-place alias ok\n");
        }
        {
            // The three sequences the exchange must not reorder, and a point-to-point stream in both directions running through them on the other connection.
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t peer = 1 - rank, p2p_bytes = std::size_t{8} << 20;
                std::vector<std::uint8_t> p2p_out(p2p_bytes), p2p_in(p2p_bytes);
                for (std::size_t j = 0; j < p2p_bytes; ++j) p2p_out[j] = pat(rank, 900, j);
                auto s = comm.send(view(p2p_out), p2p_out.size(), tbccl::DataType::UInt8, peer);
                auto r = comm.recv(view(p2p_in), p2p_in.size(), tbccl::DataType::UInt8, peer);

                // AllGather -> AllGather -> Barrier
                gather_and_check(comm, rank, 200000, 1);
                gather_and_check(comm, rank, 65536, 2);
                wait_ok(comm.barrier(), "barrier");

                // Broadcast -> AllGather -> AllReduce (posted back to back without waiting in between: the executor runs them in order)
                std::vector<std::uint8_t> bc(100003, 0);
                if (rank == 0)
                    for (std::size_t j = 0; j < bc.size(); ++j) bc[j] = pat(0, 3, j);
                std::vector<std::uint8_t> in(300001), g0(300001), g1(300001);
                for (std::size_t j = 0; j < in.size(); ++j) in[j] = pat(rank, 4, j);
                std::vector<float> ar(70001, static_cast<float>(rank + 1));
                tbccl::BufferView ar_view{tbccl::MemoryKind::Host, ar.data(), ar.size() * sizeof(float), -1};
                auto w1 = comm.broadcast(view(bc), 0);
                auto w2 = comm.all_gather(view(in), {view(g0), view(g1)});
                auto w3 = comm.all_reduce(ar_view, ar_view, ar.size(), tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                wait_ok(w1, "broadcast");
                wait_ok(w2, "all_gather after broadcast");
                wait_ok(w3, "all_reduce after all_gather");
                for (std::size_t j = 0; j < bc.size(); ++j) expect(bc[j] == pat(0, 3, j), "broadcast byte " + std::to_string(j));
                for (std::size_t j = 0; j < in.size(); ++j)
                {
                    expect(g0[j] == pat(0, 4, j), "gather slot 0 byte " + std::to_string(j));
                    expect(g1[j] == pat(1, 4, j), "gather slot 1 byte " + std::to_string(j));
                }
                for (float v : ar) expect(v == 3.0f, "all_reduce sum");

                // AllGather -> Broadcast -> AllGather
                gather_and_check(comm, rank, 150000, 5);
                std::vector<std::uint8_t> bc2(54321, 0);
                if (rank == 1)
                    for (std::size_t j = 0; j < bc2.size(); ++j) bc2[j] = pat(1, 6, j);
                wait_ok(comm.broadcast(view(bc2), 1), "broadcast root 1");
                for (std::size_t j = 0; j < bc2.size(); ++j) expect(bc2[j] == pat(1, 6, j), "broadcast root 1 byte " + std::to_string(j));
                gather_and_check(comm, rank, 150000, 7);

                // large exchange while the 8 MiB point-to-point stream may still be moving
                gather_and_check(comm, rank, std::size_t{24} << 20, 8);
                wait_ok(comm.barrier(), "final barrier");
                wait_ok(s, "p2p send");
                wait_ok(r, "p2p recv");
                for (std::size_t j = 0; j < p2p_bytes; ++j) expect(p2p_in[j] == pat(peer, 900, j), "p2p payload crossed domains at byte " + std::to_string(j));
            });
            std::printf("ordering ok\n");
        }
        {
            // Many small gathers: no descriptor/thread/fd growth (the exchange adds no persistent thread or connection).
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (int i = 0; i < 50; ++i) gather_and_check(comm, rank, 1000 + static_cast<std::size_t>(i), 50 + static_cast<std::size_t>(i)); // warm up lanes and pools
            });
            const int fds_before = open_fd_count();
            const long threads_before = thread_count();
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (int i = 0; i < 2000; ++i) gather_and_check(comm, rank, 64 + static_cast<std::size_t>(i % 50), 1000 + static_cast<std::size_t>(i));
            });
            expect(open_fd_count() == fds_before, "file descriptors grew over repeated all_gathers");
            if (threads_before >= 0)
            {
                // The test's own just-joined rank threads can still be listed for an instant while they exit; a real leak does not go away.
                const auto settle = std::chrono::steady_clock::now() + 10s;
                while (thread_count() > threads_before && std::chrono::steady_clock::now() < settle) std::this_thread::sleep_for(10ms);
                expect(thread_count() <= threads_before, "threads grew over repeated all_gathers: " + std::to_string(threads_before) + " -> " + std::to_string(thread_count()));
            }
            std::printf("resource stability ok\n");
        }
        {
            // abort while a large exchange is in flight on both ranks: every Work reaches a terminal state in bounded time and the communicator reports aborted
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t bytes = std::size_t{96} << 20;
                std::vector<std::uint8_t> in(bytes, static_cast<std::uint8_t>(rank + 1)), g0(bytes), g1(bytes);
                std::optional<tbccl::Work> work;
                try
                {
                    work = comm.all_gather(view(in), {view(g0), view(g1)});
                }
                catch (const std::exception &)
                {
                    expect(rank == 1 && comm.aborted(), "only a rank that already learned of the peer's abort may refuse the submission"); // the abort raced ahead of rank 1's call
                }
                if (rank == 0) comm.abort("test abort during all_gather");
                if (work)
                {
                    expect(work->wait_for(30s), "rank " + std::to_string(rank) + ": all_gather Work did not become terminal within 30 s after abort");
                    if (rank == 0) expect(work->has_error(), "an aborted exchange must not succeed");
                }
                if (rank == 0) expect(comm.aborted(), "rank 0 not aborted");
                else
                {
                    // rank 1 learns of the abort (or its exchange had already finished); either way nothing is left hanging
                    for (int i = 0; i < 300 && !comm.aborted(); ++i) std::this_thread::sleep_for(100ms);
                    expect(comm.aborted() || (work && !work->has_error()), "rank 1 neither finished nor learned of the abort");
                }
            });
            std::printf("abort ok\n");
        }
        for (int variant = 0; variant < 2; ++variant)
        {
            // Variant 1 also makes the abort reason impossible to build, so the legs are interrupted directly (transport and worker) instead of through the communicator's abort request;
            // the executor's own failure handling then aborts the communicator as usual.
            // The second admission of one rank's exchange fails (as if out of memory) after its send was accepted. The failing job aborts the communicator from the
            // collective executor thread, so that abort must not wait for the executor (a blocking abort there would hang this test): the job's Work fails, the
            // communicator is aborted, the peer's Work becomes terminal, and later submissions are refused.
            World w(2);
            tbccl::detail::g_n2_exchange_fail_second_enqueue = 1;
            tbccl::detail::g_n2_exchange_fail_abort_message = variant;
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t bytes = std::size_t{8} << 20;
                std::vector<std::uint8_t> in(bytes, static_cast<std::uint8_t>(rank + 1)), g0(bytes), g1(bytes);
                std::optional<tbccl::Work> work;
                try
                {
                    work = comm.all_gather(view(in), {view(g0), view(g1)});
                }
                catch (const std::exception &)
                {
                    expect(comm.aborted(), "a refused submission means the communicator was already aborted by the peer's failure");
                }
                if (work) expect(work->wait_for(30s), "rank " + std::to_string(rank) + ": the Work did not become terminal after a partial admission");
                for (int i = 0; i < 300 && !comm.aborted(); ++i) std::this_thread::sleep_for(100ms);
                expect(comm.aborted(), "rank " + std::to_string(rank) + ": the failed exchange must abort the communicator");
                bool refused = false;
                try
                {
                    comm.all_gather(view(in), {view(g0), view(g1)});
                }
                catch (const std::exception &)
                {
                    refused = true;
                }
                expect(refused, "an aborted communicator must refuse new collectives");
            });
            expect(tbccl::detail::g_n2_exchange_fail_second_enqueue.load() == 0, "the injected admission failure was consumed");
            expect(tbccl::detail::g_n2_exchange_fail_abort_message.load() == 0, "the injected abort-reason failure was consumed");
            std::printf("partial admission ok (variant %d)\n", variant);
        }
        {
            // a peer that never joins the collective, then aborts: the waiting rank's Work fails instead of hanging
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t bytes = std::size_t{32} << 20;
                if (rank == 0)
                {
                    std::vector<std::uint8_t> in(bytes, 1), g0(bytes), g1(bytes);
                    auto work = comm.all_gather(view(in), {view(g0), view(g1)});
                    expect(work.wait_for(30s), "rank 0 Work still pending after the peer aborted");
                    expect(work.has_error(), "rank 0 all_gather with a silent, then aborting peer must fail");
                }
                else
                {
                    std::this_thread::sleep_for(300ms);
                    comm.abort("peer aborts without joining the collective");
                }
            });
            std::printf("silent peer ok\n");
        }
        {
            // a peer that goes away mid-exchange (its communicator is destroyed): the survivor's Work fails in bounded time, then its own destruction returns
            auto slots = bootstrap(healthy_slots(2));
            for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
            std::unique_ptr<tbccl::Communicator> survivor = std::move(slots[0].comm), leaver = std::move(slots[1].comm);
            const std::size_t bytes = std::size_t{64} << 20;
            std::vector<std::uint8_t> in(bytes, 1), g0(bytes), g1(bytes);
            auto work = survivor->all_gather(view(in), {view(g0), view(g1)});
            std::this_thread::sleep_for(20ms);
            auto destroyer = std::async(std::launch::async, [&] { leaver.reset(); });
            expect(work.wait_for(30s), "the survivor's all_gather never became terminal after its peer was destroyed");
            expect(work.has_error(), "the survivor's all_gather cannot succeed without the peer");
            expect(destroyer.wait_for(30s) == std::future_status::ready, "the leaving communicator's destruction hung");
            auto closer = std::async(std::launch::async, [&] { survivor.reset(); });
            expect(closer.wait_for(30s) == std::future_status::ready, "the survivor's destruction hung");
            std::printf("peer exit ok\n");
        }
        {
            // destruction with an all_gather still outstanding on both ranks returns (the destructor aborts and drains)
            auto slots = bootstrap(healthy_slots(2));
            for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
            const std::size_t bytes = std::size_t{64} << 20;
            std::vector<std::uint8_t> in0(bytes, 1), in1(bytes, 2), a0(bytes), a1(bytes), b0(bytes), b1(bytes);
            auto w0 = slots[0].comm->all_gather(view(in0), {view(a0), view(a1)});
            auto w1 = slots[1].comm->all_gather(view(in1), {view(b0), view(b1)});
            auto d0 = std::async(std::launch::async, [&] { slots[0].comm.reset(); });
            auto d1 = std::async(std::launch::async, [&] { slots[1].comm.reset(); });
            expect(d0.wait_for(60s) == std::future_status::ready && d1.wait_for(60s) == std::future_status::ready, "destruction with an outstanding all_gather hung");
            expect(w0.wait_for(5s) && w1.wait_for(5s), "an outstanding all_gather was left non-terminal by destruction");
            std::printf("destruction ok\n");
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("communicator_n2_all_gather_exchange_test passed\n");
    return 0;
}
