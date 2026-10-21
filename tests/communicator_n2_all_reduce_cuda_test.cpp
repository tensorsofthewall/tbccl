// Two-rank all_reduce with device memory and the chunk-pipelined schedule on: reduced contents for every supported datatype (CUDA/CUDA, CUDA/Host, Host/CUDA; counts with
// a partial final chunk), then abort with a full exchange in flight, a peer that goes away, and destruction with an outstanding all_reduce. After every aborted or failed
// all_reduce the buffers are overwritten with a sentinel once the Work is terminal and checked again after a settle period: a transfer still writing into (or a staging copy
// or reduction kernel still touching) the caller's device memory would be seen as a changed sentinel, so "terminal" must mean no further access. Every wait is bounded.
// Run once per TBCCL_ALLREDUCE_PIPELINE_BYTES setting, combined with TBCCL_STAGED_CHUNK_BYTES (unset, an aligned and an odd value). Needs a CUDA device.

#include "mesh_test_support.hpp"
#include "reduce_test_values.hpp"

#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace mesh_test;
using namespace redtest;
using namespace std::chrono_literals;

namespace
{
void cu(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev
{
    void *p = nullptr;
    std::size_t n;
    explicit Dev(std::size_t bytes) : n(bytes) { cu(cudaMalloc(&p, std::max<std::size_t>(bytes, 1)), "cudaMalloc"); }
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

// One rank's in-place buffer, Host or Cuda.
struct Buffer
{
    bool cuda;
    std::size_t bytes;
    std::vector<std::uint8_t> host;
    std::unique_ptr<Dev> dev;
    Buffer(bool cuda_, std::size_t n) : cuda(cuda_), bytes(n), host(n)
    {
        if (cuda) dev = std::make_unique<Dev>(n);
    }
    tbccl::BufferView view() { return cuda ? tbccl::BufferView{tbccl::MemoryKind::Cuda, dev->p, bytes, 0} : tbccl::BufferView{tbccl::MemoryKind::Host, host.data(), bytes, -1}; }
    void set(const std::vector<std::uint8_t> &v)
    {
        host = v;
        if (cuda) dev->upload(v);
    }
    std::vector<std::uint8_t> get() { return cuda ? dev->download() : host; }
    void sentinel(std::uint8_t v)
    {
        if (cuda) dev->fill(v);
        else std::fill(host.begin(), host.end(), v);
    }
    bool sentinel_intact(std::uint8_t v)
    {
        const auto all = get();
        return std::all_of(all.begin(), all.end(), [&](std::uint8_t b) { return b == v; });
    }
};

void expect_no_late_access(Buffer &b, const std::string &what)
{
    b.sentinel(0x5A);
    std::this_thread::sleep_for(300ms);
    expect(b.sentinel_intact(0x5A), what + ": a transfer or kernel touched the caller's buffer after the Work was terminal");
}

void reduce_case(bool cuda0, bool cuda1, DataType type, std::size_t count)
{
    World w(2);
    w.run([&](std::size_t rank, tbccl::Communicator &comm) {
        const std::size_t bytes = count * element_size(type);
        Buffer b(rank == 0 ? cuda0 : cuda1, bytes);
        std::vector<std::uint8_t> mine(bytes), want(bytes);
        for (std::size_t i = 0; i < count; ++i)
        {
            put(mine, type, i, value(rank, i));
            put(want, type, i, value(0, i) + value(1, i));
        }
        const std::string what = std::string(name_of(type)) + " x" + std::to_string(count) + (cuda0 ? " cuda" : " host") + "/" + (cuda1 ? "cuda" : "host");
        for (int round = 0; round < 3; ++round)
        {
            b.set(mine);
            auto work = comm.all_reduce(b.view(), b.view(), count, type, tbccl::ReduceOp::Sum);
            expect(work.wait_for(60s), what + ": all_reduce did not finish");
            expect(!work.has_error(), what + ": " + work.error());
            const auto got = b.get();
            for (std::size_t i = 0; i < count; ++i) expect(same_element(got, want, type, i), what + ": element " + std::to_string(i) + " is wrong");
        }
    });
}

void abort_case(bool cuda0, bool cuda1, bool rank1_aborts)
{
    World w(2);
    w.run([&](std::size_t rank, tbccl::Communicator &comm) {
        const std::size_t count = (std::size_t{48} << 20) / 4;
        Buffer b(rank == 0 ? cuda0 : cuda1, count * 4);
        b.sentinel(static_cast<std::uint8_t>(rank + 1));
        std::optional<tbccl::Work> work;
        try
        {
            work = comm.all_reduce(b.view(), b.view(), count, DataType::Float32, tbccl::ReduceOp::Sum);
        }
        catch (const std::exception &)
        {
            expect(comm.aborted(), "a refused submission requires an aborted communicator");
        }
        const bool aborter = rank == (rank1_aborts ? 1u : 0u);
        if (aborter) comm.abort("test abort during a CUDA all_reduce");
        if (work)
        {
            expect(work->wait_for(60s), "rank " + std::to_string(rank) + ": all_reduce Work not terminal within 60 s after abort");
            if (aborter) expect(work->has_error(), "an aborted all_reduce must not succeed on the aborting rank");
        }
        for (int i = 0; i < 300 && !comm.aborted(); ++i) std::this_thread::sleep_for(100ms);
        expect(comm.aborted(), "rank " + std::to_string(rank) + " did not learn of the abort");
        expect_no_late_access(b, "abort");
    });
}

// The peer's communicator is destroyed while the all_reduce is in flight.
void peer_exit_case(bool cuda_survivor, bool cuda_leaver, bool survivor_is_root)
{
    auto slots = bootstrap(healthy_slots(2));
    for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
    const std::size_t survivor_rank = survivor_is_root ? 0 : 1;
    std::unique_ptr<tbccl::Communicator> survivor = std::move(slots[survivor_rank].comm), leaver = std::move(slots[1 - survivor_rank].comm);
    const std::size_t count = (std::size_t{64} << 20) / 4;
    Buffer sb(cuda_survivor, count * 4), lb(cuda_leaver, count * 4);
    sb.sentinel(1);
    lb.sentinel(2);
    auto work = survivor->all_reduce(sb.view(), sb.view(), count, DataType::Float32, tbccl::ReduceOp::Sum);
    auto lwork = leaver->all_reduce(lb.view(), lb.view(), count, DataType::Float32, tbccl::ReduceOp::Sum);
    auto destroyer = std::async(std::launch::async, [&] { leaver.reset(); });
    expect(work.wait_for(60s), "the survivor's all_reduce never became terminal after its peer was destroyed");
    expect(destroyer.wait_for(60s) == std::future_status::ready, "the leaving communicator's destruction hung");
    expect(lwork.wait_for(5s), "the leaver's Work was left non-terminal by destruction");
    expect_no_late_access(sb, "peer exit (survivor)");
    expect_no_late_access(lb, "peer exit (leaver, after destruction)");
    auto closer = std::async(std::launch::async, [&] { survivor.reset(); });
    expect(closer.wait_for(60s) == std::future_status::ready, "the survivor's destruction hung");
}

void destruction_case(bool cuda0, bool cuda1)
{
    auto slots = bootstrap(healthy_slots(2));
    for (auto &s : slots) expect(s.comm != nullptr, "bootstrap: " + s.error);
    const std::size_t count = (std::size_t{64} << 20) / 4;
    Buffer b0(cuda0, count * 4), b1(cuda1, count * 4);
    b0.sentinel(1);
    b1.sentinel(2);
    auto w0 = slots[0].comm->all_reduce(b0.view(), b0.view(), count, DataType::Float32, tbccl::ReduceOp::Sum);
    auto w1 = slots[1].comm->all_reduce(b1.view(), b1.view(), count, DataType::Float32, tbccl::ReduceOp::Sum);
    auto d0 = std::async(std::launch::async, [&] { slots[0].comm.reset(); });
    auto d1 = std::async(std::launch::async, [&] { slots[1].comm.reset(); });
    expect(d0.wait_for(90s) == std::future_status::ready && d1.wait_for(90s) == std::future_status::ready, "destruction with an outstanding CUDA all_reduce hung");
    expect(w0.wait_for(5s) && w1.wait_for(5s), "an outstanding all_reduce was left non-terminal by destruction");
    expect_no_late_access(b0, "destruction");
    expect_no_late_access(b1, "destruction");
}
} // namespace

int main()
{
    tbccl::register_cuda_support();
    std::thread([] {
        std::this_thread::sleep_for(900s);
        std::fprintf(stderr, "STALLED: communicator_n2_all_reduce_cuda_test exceeded 900 s\n");
        std::fflush(stderr);
        std::_Exit(124);
    }).detach();
    try
    {
        for (DataType type : {DataType::Float32, DataType::Float64, DataType::Int32, DataType::Int64, DataType::Int8, DataType::UInt8, DataType::Float16, DataType::BFloat16})
            for (std::size_t count : {std::size_t{1}, std::size_t{4099}, std::size_t{100003}, std::size_t{262145}, std::size_t{1000003}})
            {
                reduce_case(true, true, type, count);
                reduce_case(true, false, type, count);
                reduce_case(false, true, type, count);
            }
        std::printf("cuda contents ok\n");
        abort_case(true, true, false);
        abort_case(true, true, true);
        abort_case(true, false, false);
        abort_case(false, true, true);
        std::printf("cuda abort ok\n");
        peer_exit_case(true, true, true);
        peer_exit_case(true, false, false);
        peer_exit_case(false, true, true);
        peer_exit_case(true, true, false);
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
    std::printf("communicator_n2_all_reduce_cuda_test passed\n");
    return 0;
}
