// Phase 44: persistent, communicator-scoped CUDA staging. Counters prove reuse (no steady-state pinned/device
// allocation), correctness is byte/exact for every collective and memory-kind pairing, and ownership is per
// communicator (independent capacity, clean destruction, failed growth keeps the old block usable).

#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include "cuda_external_async_backend.hpp"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using tbccl::BufferView;
using tbccl::MemoryKind;
constexpr std::size_t MiB = std::size_t{1} << 20;

void expect(bool c, const std::string &m) { if (!c) throw std::runtime_error("assertion failed: " + m); }
void cu(cudaError_t e, const char *w) { if (e != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e)); }

bool port_free(std::uint16_t port)
{
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(port);
    const bool ok = ::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) == 0;
    ::close(fd);
    return ok;
}
std::uint16_t next_port()
{
    static std::uint16_t p = 30100;
    for (;;)
    {
        p = static_cast<std::uint16_t>(p + 4);
        if (port_free(p) && port_free(p + 1) && port_free(p + 1000) && port_free(p + 1001)) return p;
    }
}

struct Pair
{
    std::unique_ptr<tbccl::Communicator> c[2];
    Pair()
    {
        const auto port = next_port();
        tbccl::CommunicatorOptions o0;
        o0.rank = 0;
        o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        tbccl::CommunicatorOptions o1 = o0;
        o1.rank = 1;
        std::exception_ptr e;
        std::thread t([&] { try { c[1] = tbccl::Communicator::create(o1); } catch (...) { e = std::current_exception(); } });
        c[0] = tbccl::Communicator::create(o0);
        t.join();
        if (e) std::rethrow_exception(e);
    }
    // Runs f0 on this thread (rank 0) and f1 on a helper (rank 1).
    void both(const std::function<void(tbccl::Communicator &)> &f0, const std::function<void(tbccl::Communicator &)> &f1)
    {
        std::exception_ptr e1;
        std::thread t([&] { try { f1(*c[1]); } catch (...) { e1 = std::current_exception(); } });
        std::exception_ptr e0;
        try { f0(*c[0]); } catch (...) { e0 = std::current_exception(); }
        t.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
    }
    tbccl::CudaStagingStats stats(std::size_t rank = 0) const
    {
        tbccl::CudaStagingStats s;
        expect(tbccl::cuda_staging_stats(*c[rank], s), "staging stats available");
        return s;
    }
};

// A rank's buffer of `kind` (device memory initialised through a user stream, no host sync before submit).
struct Buf
{
    bool cuda;
    std::size_t bytes;
    void *dev = nullptr;
    std::vector<unsigned char> host;
    cudaStream_t s = nullptr;
    Buf(bool cuda_, std::size_t n, const std::vector<unsigned char> &init, cudaStream_t stream) : cuda(cuda_), bytes(n), s(stream)
    {
        if (cuda) { cu(cudaMalloc(&dev, n), "malloc"); cu(cudaMemcpyAsync(dev, init.data(), n, cudaMemcpyHostToDevice, s), "upload"); }
        else host = init;
    }
    ~Buf() { if (dev) cudaFree(dev); }
    BufferView view() { return cuda ? BufferView{MemoryKind::Cuda, dev, bytes, 0} : BufferView{MemoryKind::Host, host.data(), bytes, -1}; }
    tbccl::ExecutionContext ctx() const { return cuda ? tbccl::ExecutionContext{tbccl::ExecutionContextKind::CudaStream, s} : tbccl::ExecutionContext{}; }
    std::vector<unsigned char> read()
    {
        if (!cuda) return host;
        std::vector<unsigned char> out(bytes);
        cudaStream_t other;
        cu(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "stream");
        cu(cudaMemcpyAsync(out.data(), dev, bytes, cudaMemcpyDeviceToHost, other), "download");
        cu(cudaStreamSynchronize(other), "sync other"); // consumer on an independent stream; no device-wide sync
        cudaStreamDestroy(other);
        return out;
    }
};

std::vector<float> f32pattern(std::size_t count, int rank) { std::vector<float> v(count); for (std::size_t i = 0; i < count; ++i) v[i] = float((i % 100) + 1 + rank * 7); return v; }
std::vector<unsigned char> bytes_of(const std::vector<float> &v) { const auto *p = reinterpret_cast<const unsigned char *>(v.data()); return {p, p + v.size() * 4}; }
std::vector<unsigned char> bpattern(std::size_t n, unsigned seed) { std::vector<unsigned char> v(n); for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<unsigned char>((i * 131u + seed * 17u + 3u) & 0xffu); return v; }
void ok(tbccl::Work w, const std::string &what) { w.wait(); expect(!w.has_error(), what + ": " + w.error()); }

enum class Op { AllReduce, Broadcast0, Broadcast1, AllGather };

// One synchronous collective of `bytes` on both ranks; kinds[r] true => rank r holds CUDA memory.
void run_op(Pair &p, Op op, std::size_t bytes, bool cuda0, bool cuda1)
{
    const bool cuda[2] = {cuda0, cuda1};
    auto rank_fn = [&](std::size_t r) {
        return [&, r](tbccl::Communicator &c) {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            if (op == Op::AllReduce)
            {
                const std::size_t count = bytes / 4;
                Buf b(cuda[r], bytes, bytes_of(f32pattern(count, int(r))), s);
                ok(c.all_reduce(b.view(), b.view(), count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, b.ctx()), "all_reduce");
                const auto got = b.read();
                const auto a = f32pattern(count, 0), bb = f32pattern(count, 1);
                const float *g = reinterpret_cast<const float *>(got.data());
                for (std::size_t i = 0; i < count; i += (count > 4096 ? 1021 : 1)) expect(g[i] == a[i] + bb[i], "all_reduce value");
            }
            else if (op == Op::Broadcast0 || op == Op::Broadcast1)
            {
                const std::size_t root = op == Op::Broadcast0 ? 0 : 1;
                const auto src = bpattern(bytes, 9);
                Buf b(cuda[r], bytes, r == root ? src : std::vector<unsigned char>(bytes, 0xEE), s);
                ok(c.broadcast(b.view(), root, b.ctx()), "broadcast");
                expect(b.read() == src, "broadcast bytes");
            }
            else
            {
                const auto in0 = bpattern(bytes, 21), in1 = bpattern(bytes, 42);
                Buf in(cuda[r], bytes, r == 0 ? in0 : in1, s), o0(cuda[r], bytes, std::vector<unsigned char>(bytes, 0xEE), s), o1(cuda[r], bytes, std::vector<unsigned char>(bytes, 0xEE), s);
                ok(c.all_gather(in.view(), {o0.view(), o1.view()}, in.ctx()), "all_gather");
                expect(o0.read() == in0 && o1.read() == in1, "all_gather outputs [rank0, rank1]");
            }
            cudaStreamDestroy(s);
        };
    };
    p.both(rank_fn(0), rank_fn(1));
}

void test_correctness_all_pairings()
{
    for (auto kinds : {std::pair<bool, bool>{true, false}, {false, true}, {true, true}})
    {
        Pair p;
        for (std::size_t bytes : {std::size_t{16}, std::size_t{4096}, MiB, 16 * MiB, 25 * MiB})
        {
            run_op(p, Op::AllReduce, bytes, kinds.first, kinds.second);
            run_op(p, Op::Broadcast0, bytes, kinds.first, kinds.second);
            run_op(p, Op::Broadcast1, bytes, kinds.first, kinds.second);
            run_op(p, Op::AllGather, bytes, kinds.first, kinds.second);
        }
    }
}

void test_steady_state_reuse()
{
    for (std::size_t size : {25 * MiB, 16 * MiB})
    {
        Pair p;
        run_op(p, Op::AllReduce, size, true, false); // warmup
        const auto warm = p.stats();
        expect(warm.pinned_alloc_count == 1 && warm.pinned_capacity == size, "one pinned block after warmup");
        expect(warm.device_scratch_alloc_count == 1, "one device scratch after warmup");
        for (int i = 0; i < 100; ++i) run_op(p, Op::AllReduce, size, true, false);
        const auto s = p.stats();
        expect(s.pinned_alloc_count == 1 && s.pinned_free_count == 0, "no pinned alloc/free in steady state");
        expect(s.device_scratch_alloc_count == 1 && s.device_scratch_free_count == 0, "no device scratch alloc/free in steady state");
        expect(s.pinned_capacity == size && s.pinned_peak_capacity == size, "capacity stable");
    }
}

void test_varying_sizes_grow_only()
{
    Pair p;
    for (std::size_t mib : {1, 4, 16, 25, 4, 1, 16, 25}) run_op(p, Op::AllReduce, mib * MiB, true, false);
    auto s = p.stats();
    expect(s.pinned_alloc_count == 4, "4 growth allocations for 1,4,16,25,4,1,16,25 (got " + std::to_string(s.pinned_alloc_count) + ")");
    expect(s.pinned_free_count == 3, "each growth frees the previous block (3)");
    expect(s.pinned_capacity == 25 * MiB && s.pinned_peak_capacity == 25 * MiB, "capacity never shrinks");
    // oscillation, with mixed collectives: no further growth
    for (int round = 0; round < 3; ++round)
        for (std::size_t mib : {1, 25, 4, 16, 1, 25})
        {
            run_op(p, Op::AllReduce, mib * MiB, true, round == 1 ? true : false);
            run_op(p, Op::Broadcast1, mib * MiB, true, false);
            run_op(p, Op::AllGather, (mib * MiB) / 2, true, false);
        }
    s = p.stats();
    expect(s.pinned_alloc_count == 4, "oscillation caused no growth");
}

void test_queued_and_dropped_work()
{
    Pair p;
    constexpr int N = 6;
    constexpr std::size_t bytes = 3 * MiB;
    auto rank_fn = [&](std::size_t r) {
        return [&, r](tbccl::Communicator &c) {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            std::vector<std::unique_ptr<Buf>> bufs;
            std::vector<tbccl::Work> works;
            const std::size_t count = bytes / 4;
            for (int i = 0; i < N; ++i)
            {
                auto pat = f32pattern(count, int(r));
                for (auto &x : pat) x += float(i * 3);
                bufs.push_back(std::make_unique<Buf>(r == 0, bytes, bytes_of(pat), s));
                works.push_back(c.all_reduce(bufs[i]->view(), bufs[i]->view(), count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, bufs[i]->ctx()));
            }
            // drop the first Work handles without waiting; FIFO means waiting for the last proves all completed
            for (int i = 0; i < N - 1; ++i) works[i] = works[N - 1];
            ok(works[N - 1], "queued last");
            for (int i = 0; i < N; ++i)
            {
                const auto got = bufs[i]->read();
                const float *g = reinterpret_cast<const float *>(got.data());
                for (std::size_t k = 0; k < count; k += 997) expect(g[k] == float((k % 100) + 1) + float((k % 100) + 8) + float(i * 6), "queued op " + std::to_string(i));
            }
            cudaStreamDestroy(s);
        };
    };
    p.both(rank_fn(0), rank_fn(1));
}

void test_two_communicators_independent()
{
    Pair a, b;
    run_op(a, Op::AllReduce, 25 * MiB, true, false);
    run_op(b, Op::AllReduce, 4 * MiB, true, false);
    const auto sa = a.stats(), sb = b.stats();
    expect(sa.pinned_capacity == 25 * MiB && sb.pinned_capacity == 4 * MiB, "independent capacities");
    expect(sa.pinned_alloc_count == 1 && sb.pinned_alloc_count == 1, "independent allocation counts");
    run_op(b, Op::AllReduce, 4 * MiB, true, false);
    expect(a.stats().pinned_alloc_count == 1, "communicator A unaffected by B");
}

void test_destroy_frees_exactly_once()
{
    for (int round = 0; round < 6; ++round)
    {
        std::shared_ptr<void> held;
        std::shared_ptr<tbccl_bench::tensor::CudaStagingResources> res;
        {
            Pair p;
            run_op(p, Op::AllReduce, 4 * MiB, true, false);
            run_op(p, Op::AllReduce, 8 * MiB, true, false);
            res = std::static_pointer_cast<tbccl_bench::tensor::CudaStagingResources>(p.c[0]->provider_resources(MemoryKind::Cuda));
            expect(res != nullptr, "resources exist");
            const auto before = res->stats();
            expect(before.pinned_alloc_count == 2 && before.pinned_free_count == 1, "one live block before destroy");
        }
        // The communicator (and every provider/outstanding Work) must have dropped its reference: ours is the last, so
        // resetting it runs the single final cudaFreeHost/cudaFree. (Exact free counts are cross-checked in the
        // interposer run: total cudaMallocHost == total cudaFreeHost.)
        expect(res.use_count() == 1, "communicator released its staging resources on destruction");
        res.reset();
    }
}

void test_failed_growth_keeps_old_block()
{
    Pair p;
    run_op(p, Op::AllReduce, MiB, true, false);
    const auto before = p.stats();
    // rank 0 (CUDA) is the broadcast receiver: the data arrives, then committing into a larger-than-capacity
    // staging block must grow, and that growth is made to fail.
    tbccl::cuda_staging_test_fail_next_pinned_growth(*p.c[0]);
    bool failed = false;
    p.both(
        [&](tbccl::Communicator &c) {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            Buf b(true, 4 * MiB, std::vector<unsigned char>(4 * MiB, 0), s);
            auto w = c.broadcast(b.view(), 1, b.ctx());
            w.wait();
            failed = w.has_error();
            expect(failed && w.error().find("device_error") != std::string::npos, "growth failure reported as device_error: " + w.error());
            cudaStreamDestroy(s);
        },
        [&](tbccl::Communicator &c) {
            std::vector<unsigned char> h(4 * MiB, 7);
            ok(c.broadcast({MemoryKind::Host, h.data(), h.size(), -1}, 1), "root send");
        });
    const auto mid = p.stats();
    expect(mid.pinned_alloc_count == before.pinned_alloc_count && mid.pinned_capacity == before.pinned_capacity, "old block preserved");
    // the same communicator keeps working at the preserved capacity and (after the injected failure) can grow again
    run_op(p, Op::AllReduce, MiB, true, false);
    run_op(p, Op::Broadcast1, 4 * MiB, true, false);
    expect(p.stats().pinned_capacity == 4 * MiB, "growth succeeds again");
}

// Pure-device producer (a delayed kernel on its own stream, no host involvement) feeds all_reduce through the
// ExecutionContext. Positive: producer stream => correct sum. Negative control: an idle stream => the stale zeros
// are read, proving the test can observe a missing dependency. The first (warm-up) collective keeps one-time
// setup from masking the delay.
void test_stream_ordering_delayed_producer()
{
    constexpr std::size_t bytes = 64 * 1024, count = bytes / 4; // small: every kernel thread spins
    constexpr std::int32_t one_f = 0x3f800000; // bit pattern of 1.0f
    Pair p;
    run_op(p, Op::AllReduce, bytes, true, false); // warm-up
    for (int control = 0; control < 2; ++control)
    {
        void *dev = nullptr;
        cu(cudaMalloc(&dev, bytes), "malloc");
        cu(cudaMemset(dev, 0, bytes), "memset");
        void *producer = tbccl_bench::tensor::cuda_external_test_launch_delayed_write_i32(dev, count, one_f, 3'000'000ull, /*non_blocking=*/true);
        cudaStream_t idle;
        cu(cudaStreamCreateWithFlags(&idle, cudaStreamNonBlocking), "idle stream");
        void *submit_stream = control == 0 ? producer : static_cast<void *>(idle);
        std::vector<float> two(count, 2.0f), got(count);
        p.both(
            [&](tbccl::Communicator &c) {
                tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, submit_stream};
                ok(c.all_reduce({MemoryKind::Cuda, dev, bytes, 0}, {MemoryKind::Cuda, dev, bytes, 0}, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, ctx), "all_reduce");
            },
            [&](tbccl::Communicator &c) {
                std::vector<float> h = two;
                ok(c.all_reduce({MemoryKind::Host, h.data(), bytes, -1}, {MemoryKind::Host, h.data(), bytes, -1}, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum), "peer all_reduce");
            });
        cu(cudaStreamSynchronize(static_cast<cudaStream_t>(producer)), "producer sync");
        cu(cudaMemcpy(got.data(), dev, bytes, cudaMemcpyDeviceToHost), "readback");
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < count; ++i) wrong += got[i] != 3.0f;
        if (control == 0) expect(wrong == 0, "producer-stream submission must see the produced data (wrong=" + std::to_string(wrong) + ")");
        else expect(wrong > 0, "negative control must observe stale data when the wrong stream is supplied");
        cudaStreamDestroy(static_cast<cudaStream_t>(producer));
        cudaStreamDestroy(idle);
        cudaFree(dev);
    }
}

} // namespace

int main(int argc, char **argv)
{
    tbccl::register_cuda_support();
    struct T { const char *n; void (*f)(); } tests[] = {
        {"stream ordering: delayed producer + negative control", test_stream_ordering_delayed_producer},
        {"correctness (all pairings, all collectives, 16 B..25 MiB)", test_correctness_all_pairings},
        {"steady state reuse (100 x 25 MiB, 100 x 16 MiB)", test_steady_state_reuse},
        {"varying sizes grow-only + oscillation", test_varying_sizes_grow_only},
        {"queued + dropped Work", test_queued_and_dropped_work},
        {"two communicators independent", test_two_communicators_independent},
        {"destroy frees exactly once (repeated create/destroy)", test_destroy_frees_exactly_once},
        {"failed growth keeps old block", test_failed_growth_keeps_old_block},
    };
    try
    {
        for (const auto &t : tests) { if (argc > 1 && std::string(t.n).find(argv[1]) == std::string::npos) continue; std::cerr << "[RUN ] " << t.n << std::endl; t.f(); std::cout << "[PASS] " << t.n << "\n"; }
    }
    catch (const std::exception &e) { std::cerr << "[FAIL] " << e.what() << "\n"; return 1; }
    std::cout << "All CUDA staging tests passed.\n";
    return 0;
}
