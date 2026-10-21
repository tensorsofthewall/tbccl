// Two-rank all_gather with device memory, both directions in flight: contents (CUDA/CUDA, CUDA/Host, Host/CUDA, partial final chunks), abort with a full
// exchange in flight, a peer that goes away, and destruction with an outstanding exchange. After every aborted or failed exchange the buffers are overwritten with a
// sentinel once the Work is terminal and checked again after a settle period: a transfer still writing into (or a staging copy still reading from) the caller's
// device memory would be seen as a changed sentinel, so "terminal" must mean no further access. Every wait is bounded.
// Run once per TBCCL_STAGED_CHUNK_BYTES setting (unset, an aligned and an odd value). Needs a CUDA device.

#include "mesh_test_support.hpp"

#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace mesh_test;
using namespace std::chrono_literals;

namespace
{
void cu(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::uint8_t pat(std::size_t rank, std::size_t seq, std::size_t j) { return static_cast<std::uint8_t>(rank * 61 + seq * 29 + j * 7 + (j >> 12) * 3 + 5); }

struct Dev
{
    void *p = nullptr;
    std::size_t n;
    explicit Dev(std::size_t bytes) : n(bytes) { cu(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    void upload(const std::vector<std::uint8_t> &h) { cu(cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice), "H2D"); }
    void fill(std::uint8_t v) { cu(cudaMemset(p, v, n), "memset"); cu(cudaDeviceSynchronize(), "sync"); }
    std::vector<std::uint8_t> download() const
    {
        std::vector<std::uint8_t> h(n);
        cu(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost), "D2H");
        return h;
    }
};

// One rank's buffers: input, two outputs, each Host or Cuda.
struct Buffers
{
    bool cuda;
    std::size_t bytes;
    std::vector<std::uint8_t> h_in, h_o0, h_o1;
    std::unique_ptr<Dev> d_in, d_o0, d_o1;
    Buffers(bool cuda_, std::size_t n) : cuda(cuda_), bytes(n)
    {
        if (cuda)
        {
            d_in = std::make_unique<Dev>(n);
            d_o0 = std::make_unique<Dev>(n);
            d_o1 = std::make_unique<Dev>(n);
        }
        else
        {
            h_in.assign(n, 0);
            h_o0.assign(n, 0xEE);
            h_o1.assign(n, 0xEE);
        }
    }
    tbccl::BufferView in() { return cuda ? tbccl::BufferView{tbccl::MemoryKind::Cuda, d_in->p, bytes, 0} : tbccl::BufferView{tbccl::MemoryKind::Host, h_in.data(), bytes, -1}; }
    tbccl::BufferView out(int i)
    {
        if (cuda) return {tbccl::MemoryKind::Cuda, (i == 0 ? d_o0 : d_o1)->p, bytes, 0};
        return {tbccl::MemoryKind::Host, (i == 0 ? h_o0 : h_o1).data(), bytes, -1};
    }
    void set_input(const std::vector<std::uint8_t> &v)
    {
        if (cuda) d_in->upload(v);
        else h_in = v;
    }
    void sentinel(std::uint8_t v)
    {
        if (cuda)
        {
            d_in->fill(v);
            d_o0->fill(v);
            d_o1->fill(v);
        }
        else
        {
            std::fill(h_in.begin(), h_in.end(), v);
            std::fill(h_o0.begin(), h_o0.end(), v);
            std::fill(h_o1.begin(), h_o1.end(), v);
        }
    }
    bool sentinel_intact(std::uint8_t v)
    {
        const auto ok = [&](const std::vector<std::uint8_t> &x) { return std::all_of(x.begin(), x.end(), [&](std::uint8_t b) { return b == v; }); };
        if (cuda) return ok(d_in->download()) && ok(d_o0->download()) && ok(d_o1->download());
        return ok(h_in) && ok(h_o0) && ok(h_o1);
    }
    std::vector<std::uint8_t> output(int i) { return cuda ? (i == 0 ? d_o0 : d_o1)->download() : (i == 0 ? h_o0 : h_o1); }
};

// After a terminal Work no TBCCL thread may still access the caller's memory: overwrite with a sentinel, let any late copy land, and check it is untouched.
void expect_no_late_access(Buffers &b, const std::string &what)
{
    b.sentinel(0x5A);
    std::this_thread::sleep_for(300ms);
    expect(b.sentinel_intact(0x5A), what + ": a transfer touched the caller's buffers after the Work was terminal");
}

void gather_case(bool cuda0, bool cuda1, std::size_t bytes, std::size_t seq)
{
    World w(2);
    w.run([&](std::size_t rank, tbccl::Communicator &comm) {
        Buffers b(rank == 0 ? cuda0 : cuda1, bytes);
        std::vector<std::uint8_t> mine(bytes);
        for (std::size_t j = 0; j < bytes; ++j) mine[j] = pat(rank, seq, j);
        b.set_input(mine);
        for (int round = 0; round < 3; ++round)
        {
            auto work = comm.all_gather(b.in(), {b.out(0), b.out(1)});
            expect(work.wait_for(60s), "all_gather did not finish");
            expect(!work.has_error(), work.error());
            const auto o0 = b.output(0), o1 = b.output(1);
            for (std::size_t j = 0; j < bytes; ++j)
            {
                expect(o0[j] == pat(0, seq, j), "slot 0 byte " + std::to_string(j) + " of " + std::to_string(bytes));
                expect(o1[j] == pat(1, seq, j), "slot 1 byte " + std::to_string(j) + " of " + std::to_string(bytes));
            }
        }
    });
}

void abort_case(bool cuda0, bool cuda1, bool use_rank1_abort)
{
    World w(2);
    w.run([&](std::size_t rank, tbccl::Communicator &comm) {
        const std::size_t bytes = std::size_t{48} << 20;
        Buffers b(rank == 0 ? cuda0 : cuda1, bytes);
        b.sentinel(static_cast<std::uint8_t>(rank + 1));
        std::optional<tbccl::Work> work;
        try
        {
            work = comm.all_gather(b.in(), {b.out(0), b.out(1)});
        }
        catch (const std::exception &)
        {
            expect(comm.aborted(), "a refused submission requires an aborted communicator");
        }
        const bool aborter = rank == (use_rank1_abort ? 1u : 0u);
        if (aborter) comm.abort("test abort during a CUDA all_gather");
        if (work)
        {
            expect(work->wait_for(60s), "rank " + std::to_string(rank) + ": all_gather Work not terminal within 60 s after abort");
            if (aborter) expect(work->has_error(), "an aborted exchange must not succeed on the aborting rank");
        }
        for (int i = 0; i < 300 && !comm.aborted(); ++i) std::this_thread::sleep_for(100ms);
        expect(comm.aborted(), "rank " + std::to_string(rank) + " did not learn of the abort");
        expect_no_late_access(b, "abort");
    });
}

// The peer's communicator is destroyed while the exchange is in flight.
void peer_exit_case(bool cuda_survivor, bool cuda_leaver)
{
    auto slots = bootstrap(healthy_slots(2));
    for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
    std::unique_ptr<tbccl::Communicator> survivor = std::move(slots[0].comm), leaver = std::move(slots[1].comm);
    const std::size_t bytes = std::size_t{64} << 20;
    Buffers sb(cuda_survivor, bytes), lb(cuda_leaver, bytes);
    sb.sentinel(1);
    lb.sentinel(2);
    auto work = survivor->all_gather(sb.in(), {sb.out(0), sb.out(1)});
    auto lwork = leaver->all_gather(lb.in(), {lb.out(0), lb.out(1)});
    auto destroyer = std::async(std::launch::async, [&] { leaver.reset(); });
    expect(work.wait_for(60s), "the survivor's all_gather never became terminal after its peer was destroyed");
    expect(destroyer.wait_for(60s) == std::future_status::ready, "the leaving communicator's destruction hung");
    expect(lwork.wait_for(5s), "the leaver's Work was left non-terminal by destruction");
    // the survivor either finished its exchange before the peer went away or failed; either way it is terminal and nothing touches its memory afterwards
    expect_no_late_access(sb, "peer exit (survivor)");
    expect_no_late_access(lb, "peer exit (leaver, after destruction)");
    auto closer = std::async(std::launch::async, [&] { survivor.reset(); });
    expect(closer.wait_for(60s) == std::future_status::ready, "the survivor's destruction hung");
}

void destruction_case(bool cuda0, bool cuda1)
{
    auto slots = bootstrap(healthy_slots(2));
    for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
    const std::size_t bytes = std::size_t{64} << 20;
    Buffers b0(cuda0, bytes), b1(cuda1, bytes);
    b0.sentinel(1);
    b1.sentinel(2);
    auto w0 = slots[0].comm->all_gather(b0.in(), {b0.out(0), b0.out(1)});
    auto w1 = slots[1].comm->all_gather(b1.in(), {b1.out(0), b1.out(1)});
    auto d0 = std::async(std::launch::async, [&] { slots[0].comm.reset(); });
    auto d1 = std::async(std::launch::async, [&] { slots[1].comm.reset(); });
    expect(d0.wait_for(90s) == std::future_status::ready && d1.wait_for(90s) == std::future_status::ready, "destruction with an outstanding CUDA all_gather hung");
    expect(w0.wait_for(5s) && w1.wait_for(5s), "an outstanding all_gather was left non-terminal by destruction");
    expect_no_late_access(b0, "destruction");
    expect_no_late_access(b1, "destruction");
}
} // namespace

int main()
{
    tbccl::register_cuda_support();
    std::thread([] {
        std::this_thread::sleep_for(900s);
        std::fprintf(stderr, "STALLED: communicator_n2_all_gather_cuda_test exceeded 900 s\n");
        std::fflush(stderr);
        std::_Exit(124);
    }).detach();
    try
    {
        std::size_t seq = 0;
        for (std::size_t bytes : {std::size_t{4096}, std::size_t{100003}, (std::size_t{1} << 20) + 1, (std::size_t{4} << 20) + 13})
        {
            gather_case(true, true, bytes, seq++);
            gather_case(true, false, bytes, seq++);
            gather_case(false, true, bytes, seq++);
        }
        std::printf("cuda contents ok\n");
        abort_case(true, true, false);
        abort_case(true, true, true);
        abort_case(true, false, false);
        abort_case(false, true, true);
        std::printf("cuda abort ok\n");
        peer_exit_case(true, true);
        peer_exit_case(true, false);
        peer_exit_case(false, true);
        std::printf("cuda peer exit ok\n");
        destruction_case(true, true);
        destruction_case(true, false);
        std::printf("cuda destruction ok\n");
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("communicator_n2_all_gather_cuda_test passed\n");
    return 0;
}
