// The chunk-pipelined two-rank all_reduce over real loopback sockets, one Communicator per rank. Run with TBCCL_ALLREDUCE_PIPELINE_BYTES set (the setting is latched per
// process, so both ranks of a World share it); payloads below two chunks take the unchanged sequential path in the same run. What this must not break: the reduced contents
// of every supported datatype (every element), irregular counts with a partial final chunk, in-place and out-of-place host buffers, ordering against the other collectives and
// point-to-point traffic, resource stability over many collectives, and bounded termination under abort, a partial admission, a silent peer, a peer that goes away and
// destruction. The deterministic proof that receive, reduction and result send overlap is in n2_pipeline_unit_test; schedules that differ between the two ranks
// (pipelined against sequential) are run as separate processes by the benchmark tooling and by n2_pipeline_mixed_test.

#include "communicator_debug.hpp"
#include "mesh_test_support.hpp"
#include "reduce_test_values.hpp"

#include <tbccl/communicator.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mesh_test;
using namespace std::chrono_literals;
using tbccl::DataType;
using namespace redtest;

namespace
{
std::uint8_t pat(std::size_t rank, std::size_t seq, std::size_t j) { return static_cast<std::uint8_t>(rank * 61 + seq * 29 + j * 7 + (j >> 12) * 3 + 5); }

tbccl::BufferView view(void *p, std::size_t bytes) { return {tbccl::MemoryKind::Host, p, bytes, -1}; }

void wait_ok(tbccl::Work work, const std::string &what)
{
    expect(work.wait_for(60s), what + ": the Work did not finish within 60 s");
    expect(!work.has_error(), what + ": " + work.error());
}

// All-reduces `count` elements of `type` in place (or out of place) and checks every element of the result.
void reduce_and_check(tbccl::Communicator &comm, std::size_t rank, DataType type, std::size_t count, bool in_place)
{
    const std::size_t bytes = count * element_size(type);
    std::vector<std::uint8_t> mine(bytes), expect_bytes(bytes), out(in_place ? 0 : bytes, 0xEE);
    for (std::size_t i = 0; i < count; ++i)
    {
        put(mine, type, i, value(rank, i));
        put(expect_bytes, type, i, value(0, i) + value(1, i));
    }
    const std::vector<std::uint8_t> original = mine;
    std::vector<std::uint8_t> &result = in_place ? mine : out;
    const auto what = std::string(name_of(type)) + " x" + std::to_string(count) + (in_place ? " in place" : " out of place");
    wait_ok(comm.all_reduce(view(mine.data(), bytes), view(result.data(), bytes), count, type, tbccl::ReduceOp::Sum), "all_reduce " + what);
    for (std::size_t i = 0; i < count; ++i) expect(same_element(result, expect_bytes, type, i), what + ": element " + std::to_string(i) + " is wrong");
    if (!in_place) expect(mine == original, what + ": the send buffer must be left unchanged");
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
        std::fprintf(stderr, "STALLED: communicator_n2_all_reduce_pipeline_test exceeded 900 s\n");
        std::fflush(stderr);
        std::_Exit(124);
    }).detach();
    try
    {
        {
            // every datatype; counts from nothing to several chunks, odd ones so the final chunk is partial and not a multiple of anything convenient
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (DataType t : {DataType::Float32, DataType::Float64, DataType::Int32, DataType::Int64, DataType::Int8, DataType::UInt8, DataType::Float16, DataType::BFloat16})
                    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{1000}, std::size_t{16384}, std::size_t{100003}, std::size_t{262145}, std::size_t{1000003}})
                        reduce_and_check(comm, rank, t, count, true);
            });
            std::printf("datatypes ok\n");
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (DataType t : {DataType::Float32, DataType::Int64, DataType::BFloat16})
                    for (std::size_t count : {std::size_t{5}, std::size_t{100003}, std::size_t{700001}})
                        reduce_and_check(comm, rank, t, count, false);
            });
            std::printf("out of place ok\n");
        }
        {
            // a buffer larger than count * element size takes the sequential schedule: the first `count` elements are reduced and the rest of the buffer is what the
            // sequential algorithm always did (the root's tail comes back); unchanged by the pipeline
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t count = 200000, extra = 4096, bytes = count * 4 + extra;
                std::vector<std::uint8_t> buf(bytes), want_head(count * 4);
                for (std::size_t j = 0; j < bytes; ++j) buf[j] = pat(rank, 77, j);
                for (std::size_t i = 0; i < count; ++i)
                {
                    float f[1] = {static_cast<float>(value(rank, i))};
                    std::memcpy(buf.data() + i * 4, f, 4);
                    f[0] = static_cast<float>(value(0, i) + value(1, i));
                    std::memcpy(want_head.data() + i * 4, f, 4);
                }
                wait_ok(comm.all_reduce(view(buf.data(), bytes), view(buf.data(), bytes), count, DataType::Float32, tbccl::ReduceOp::Sum), "all_reduce with a larger buffer");
                expect(std::memcmp(buf.data(), want_head.data(), count * 4) == 0, "the reduced head is wrong");
                for (std::size_t j = count * 4; j < bytes; ++j) expect(buf[j] == pat(0, 77, j), "the tail beyond count elements is the root's, as before");
            });
            std::printf("larger buffer ok\n");
        }
        {
            // payloads far beyond the socket buffers, repeated
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (int i = 0; i < 3; ++i) reduce_and_check(comm, rank, DataType::Float32, (std::size_t{48} << 20) / 4 + 3, true);
                for (int i = 0; i < 40; ++i) reduce_and_check(comm, rank, DataType::Float32, 100000 + static_cast<std::size_t>(i) * 977, true);
            });
            std::printf("large and repeated ok\n");
        }
        {
            // ordering against the other collectives and a point-to-point stream running through them on the other connection
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t peer = 1 - rank, p2p_bytes = std::size_t{8} << 20;
                std::vector<std::uint8_t> p2p_out(p2p_bytes), p2p_in(p2p_bytes);
                for (std::size_t j = 0; j < p2p_bytes; ++j) p2p_out[j] = pat(rank, 900, j);
                auto s = comm.send(view(p2p_out.data(), p2p_bytes), p2p_bytes, DataType::UInt8, peer);
                auto r = comm.recv(view(p2p_in.data(), p2p_bytes), p2p_bytes, DataType::UInt8, peer);

                // posted back to back without waiting in between: the executor runs them in order
                std::vector<std::uint8_t> bc(100003, 0), in(300001), g0(300001), g1(300001);
                if (rank == 0)
                    for (std::size_t j = 0; j < bc.size(); ++j) bc[j] = pat(0, 3, j);
                for (std::size_t j = 0; j < in.size(); ++j) in[j] = pat(rank, 4, j);
                const std::size_t n = 600011;
                std::vector<std::uint8_t> a1(n * 4), a2(n * 4), w1exp(n * 4), w2exp(n * 4);
                for (std::size_t i = 0; i < n; ++i)
                {
                    put(a1, DataType::Float32, i, value(rank, i));
                    put(a2, DataType::Float32, i, value(rank, i + 5));
                    put(w1exp, DataType::Float32, i, value(0, i) + value(1, i));
                    put(w2exp, DataType::Float32, i, value(0, i + 5) + value(1, i + 5));
                }
                auto b = comm.broadcast(view(bc.data(), bc.size()), 0);
                auto x1 = comm.all_reduce(view(a1.data(), a1.size()), view(a1.data(), a1.size()), n, DataType::Float32, tbccl::ReduceOp::Sum);
                auto g = comm.all_gather(view(in.data(), in.size()), {view(g0.data(), g0.size()), view(g1.data(), g1.size())});
                auto x2 = comm.all_reduce(view(a2.data(), a2.size()), view(a2.data(), a2.size()), n, DataType::Float32, tbccl::ReduceOp::Sum);
                auto bar = comm.barrier();
                wait_ok(b, "broadcast");
                wait_ok(x1, "first all_reduce");
                wait_ok(g, "all_gather between all_reduces");
                wait_ok(x2, "second all_reduce");
                wait_ok(bar, "barrier");
                for (std::size_t j = 0; j < bc.size(); ++j) expect(bc[j] == pat(0, 3, j), "broadcast byte " + std::to_string(j));
                expect(a1 == w1exp, "first all_reduce result crossed with another collective");
                expect(a2 == w2exp, "second all_reduce result crossed with another collective");
                for (std::size_t j = 0; j < in.size(); ++j)
                {
                    expect(g0[j] == pat(0, 4, j), "gather slot 0 byte " + std::to_string(j));
                    expect(g1[j] == pat(1, 4, j), "gather slot 1 byte " + std::to_string(j));
                }
                wait_ok(s, "p2p send");
                wait_ok(r, "p2p recv");
                for (std::size_t j = 0; j < p2p_bytes; ++j) expect(p2p_in[j] == pat(peer, 900, j), "p2p payload crossed domains at byte " + std::to_string(j));
            });
            std::printf("ordering ok\n");
        }
        {
            // Many collectives: no thread or descriptor growth (the pipeline adds no persistent thread or connection).
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (int i = 0; i < 30; ++i) reduce_and_check(comm, rank, DataType::Float32, 70000 + static_cast<std::size_t>(i), true); // warm up lanes and pools
            });
            const int fds_before = open_fd_count();
            const long threads_before = thread_count();
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                for (int i = 0; i < 600; ++i) reduce_and_check(comm, rank, DataType::Float32, 66000 + static_cast<std::size_t>(i % 40) * 13, true);
            });
            expect(open_fd_count() == fds_before, "file descriptors grew over repeated all_reduces");
            if (threads_before >= 0)
            {
                const auto settle = std::chrono::steady_clock::now() + 10s;
                while (thread_count() > threads_before && std::chrono::steady_clock::now() < settle) std::this_thread::sleep_for(10ms);
                expect(thread_count() <= threads_before, "threads grew over repeated all_reduces: " + std::to_string(threads_before) + " -> " + std::to_string(thread_count()));
            }
            std::printf("resource stability ok\n");
        }
        for (int aborter = 0; aborter < 2; ++aborter)
        {
            // abort while a large all_reduce is in flight: every Work reaches a terminal state in bounded time and the communicator reports aborted
            World w(2);
            w.run([aborter](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t count = (std::size_t{96} << 20) / 4;
                std::vector<float> data(count, static_cast<float>(rank + 1));
                const auto bv = view(data.data(), count * 4);
                std::optional<tbccl::Work> work;
                try
                {
                    work = comm.all_reduce(bv, bv, count, DataType::Float32, tbccl::ReduceOp::Sum);
                }
                catch (const std::exception &)
                {
                    expect(comm.aborted(), "only an already aborted communicator may refuse the submission");
                }
                if (rank == static_cast<std::size_t>(aborter)) comm.abort("test abort during all_reduce");
                if (work)
                {
                    expect(work->wait_for(30s), "rank " + std::to_string(rank) + ": all_reduce Work did not become terminal within 30 s after abort");
                    if (rank == static_cast<std::size_t>(aborter)) expect(work->has_error(), "an aborted all_reduce must not succeed on the aborting rank");
                }
                for (int i = 0; i < 300 && !comm.aborted(); ++i) std::this_thread::sleep_for(100ms);
                expect(comm.aborted(), "rank " + std::to_string(rank) + " did not learn of the abort");
            });
            std::printf("abort ok (rank %d aborts)\n", aborter);
        }
        // The pipeline's chunk geometry for a 16 MiB payload (0 = the pipeline is off for this run: the admission hook below applies to it only).
        const char *setting_text = std::getenv("TBCCL_ALLREDUCE_PIPELINE_BYTES");
        const std::size_t setting = setting_text ? std::strtoull(setting_text, nullptr, 10) : 0;
        const std::size_t chunk_bytes = setting / 64 * 64;
        const std::size_t chunk_count = chunk_bytes ? ((std::size_t{16} << 20) + chunk_bytes - 1) / chunk_bytes : 0;
        for (int variant = 0; chunk_count != 0 && variant < 3; ++variant)
        {
            // The root's admission of its n-th chunk fails (as if out of memory): variant 0 a receive chunk, 1 a result send, 2 a receive chunk while the abort reason also cannot be
            // built (the legs are then interrupted directly). The failing job aborts the communicator from the collective executor thread, which must not wait for the executor
            // (a blocking abort there would hang this test): its Work fails, the communicator is aborted, the peer's Work becomes terminal, later submissions are refused.
            World w(2);
            tbccl::detail::g_n2_pipeline_fail_admission = variant == 1 ? static_cast<int>(chunk_count) + 3 : 2; // a result send, or the second receive
            tbccl::detail::g_n2_exchange_fail_abort_message = variant == 2 ? 1 : 0;
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                const std::size_t count = (std::size_t{16} << 20) / 4;
                std::vector<float> data(count, static_cast<float>(rank + 1));
                const auto bv = view(data.data(), count * 4);
                std::optional<tbccl::Work> work;
                try
                {
                    work = comm.all_reduce(bv, bv, count, DataType::Float32, tbccl::ReduceOp::Sum);
                }
                catch (const std::exception &)
                {
                    expect(comm.aborted(), "a refused submission means the communicator was already aborted by the peer's failure");
                }
                if (work) expect(work->wait_for(30s), "rank " + std::to_string(rank) + ": the Work did not become terminal after a partial admission");
                if (rank == 0 && work) expect(work->has_error(), "the root's all_reduce must fail when it cannot admit a chunk");
                for (int i = 0; i < 300 && !comm.aborted(); ++i) std::this_thread::sleep_for(100ms);
                expect(comm.aborted(), "rank " + std::to_string(rank) + ": the failed all_reduce must abort the communicator");
                bool refused = false;
                try
                {
                    comm.all_reduce(bv, bv, count, DataType::Float32, tbccl::ReduceOp::Sum);
                }
                catch (const std::exception &)
                {
                    refused = true;
                }
                expect(refused, "an aborted communicator must refuse new collectives");
            });
            expect(tbccl::detail::g_n2_pipeline_fail_admission.load() == 0, "the injected admission failure was consumed");
            expect(tbccl::detail::g_n2_exchange_fail_abort_message.load() == 0, "the injected abort-reason failure was consumed");
            std::printf("partial admission ok (variant %d)\n", variant);
        }
        {
            // a peer that never joins the collective, then aborts: the waiting rank's Work fails instead of hanging
            for (std::size_t waiting : {std::size_t{0}, std::size_t{1}})
            {
                World w(2);
                w.run([waiting](std::size_t rank, tbccl::Communicator &comm) {
                    const std::size_t count = (std::size_t{32} << 20) / 4;
                    if (rank == waiting)
                    {
                        std::vector<float> data(count, 1.0f);
                        const auto bv = view(data.data(), count * 4);
                        auto work = comm.all_reduce(bv, bv, count, DataType::Float32, tbccl::ReduceOp::Sum);
                        expect(work.wait_for(30s), "the waiting rank's Work is still pending after the peer aborted");
                        expect(work.has_error(), "all_reduce with a silent, then aborting peer must fail");
                    }
                    else
                    {
                        std::this_thread::sleep_for(300ms);
                        comm.abort("peer aborts without joining the collective");
                    }
                });
            }
            std::printf("silent peer ok\n");
        }
        for (int leaver_rank = 0; leaver_rank < 2; ++leaver_rank)
        {
            // a peer that goes away mid-collective (its communicator is destroyed): the survivor's Work fails in bounded time, then its own destruction returns
            auto slots = bootstrap(healthy_slots(2));
            for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
            std::unique_ptr<tbccl::Communicator> survivor = std::move(slots[1 - leaver_rank].comm), leaver = std::move(slots[leaver_rank].comm);
            const std::size_t count = (std::size_t{64} << 20) / 4;
            std::vector<float> data(count, 1.0f);
            const auto bv = view(data.data(), count * 4);
            auto work = survivor->all_reduce(bv, bv, count, DataType::Float32, tbccl::ReduceOp::Sum);
            std::this_thread::sleep_for(20ms);
            auto destroyer = std::async(std::launch::async, [&] { leaver.reset(); });
            expect(work.wait_for(30s), "the survivor's all_reduce never became terminal after its peer was destroyed");
            expect(work.has_error(), "the survivor's all_reduce cannot succeed without the peer");
            expect(destroyer.wait_for(30s) == std::future_status::ready, "the leaving communicator's destruction hung");
            auto closer = std::async(std::launch::async, [&] { survivor.reset(); });
            expect(closer.wait_for(30s) == std::future_status::ready, "the survivor's destruction hung");
        }
        std::printf("peer exit ok\n");
        {
            // destruction with an all_reduce still outstanding on both ranks returns (the destructor aborts and drains)
            auto slots = bootstrap(healthy_slots(2));
            for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
            const std::size_t count = (std::size_t{64} << 20) / 4;
            std::vector<float> d0(count, 1.0f), d1(count, 2.0f);
            auto w0 = slots[0].comm->all_reduce(view(d0.data(), count * 4), view(d0.data(), count * 4), count, DataType::Float32, tbccl::ReduceOp::Sum);
            auto w1 = slots[1].comm->all_reduce(view(d1.data(), count * 4), view(d1.data(), count * 4), count, DataType::Float32, tbccl::ReduceOp::Sum);
            auto x0 = std::async(std::launch::async, [&] { slots[0].comm.reset(); });
            auto x1 = std::async(std::launch::async, [&] { slots[1].comm.reset(); });
            expect(x0.wait_for(60s) == std::future_status::ready && x1.wait_for(60s) == std::future_status::ready, "destruction with an outstanding all_reduce hung");
            expect(w0.wait_for(5s) && w1.wait_for(5s), "an outstanding all_reduce was left non-terminal by destruction");
            std::printf("destruction ok\n");
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("communicator_n2_all_reduce_pipeline_test passed\n");
    return 0;
}
