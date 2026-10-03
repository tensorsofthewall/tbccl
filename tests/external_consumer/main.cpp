// Phase 49 external-consumer check: uses ONLY the installed public headers (<tbccl/...>) and the installed TBCCL::tbccl[_cuda] targets.
// Two Communicator ranks on loopback (threads) run N=2 SUM all_reduce for every Phase 49 reduction type on host memory; with CONSUMER_WITH_CUDA, rank 0
// also holds BFloat16 / Float16 data on the device (register_cuda_support), and the arithmetic is checked bit for bit against a self-contained reference.
#include <tbccl/communicator.hpp>
#include <tbccl/reduction.hpp>
#ifdef CONSUMER_WITH_CUDA
#include <tbccl/cuda_support.hpp>
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
    void expect(bool ok, const std::string &what)
    {
        if (!ok) throw std::runtime_error("consumer check failed: " + what);
    }

    // Independent references (no TBCCL internals): bfloat16 / float16 widen to float, add, round to nearest even; ints modulo 256.
    float bf16_to_f(std::uint16_t b) { std::uint32_t u = std::uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
    std::uint16_t f_to_bf16(float f)
    {
        std::uint32_t u; std::memcpy(&u, &f, 4);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u) return std::uint16_t((u >> 16) | 0x40u);
        u += 0x7FFFu + ((u >> 16) & 1u);
        return std::uint16_t(u >> 16);
    }
    float h_to_f(std::uint16_t h)
    {
        const std::uint32_t s = (std::uint32_t(h) & 0x8000u) << 16; std::uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, u;
        if (e == 0) { if (!m) u = s; else { int sh = 0; while (!(m & 0x400)) { m <<= 1; ++sh; } m &= 0x3FF; u = s | (std::uint32_t(113 - sh) << 23) | (m << 13); } }
        else if (e == 31) u = s | 0x7F800000u | (m << 13);
        else u = s | ((e + 112) << 23) | (m << 13);
        float f; std::memcpy(&f, &u, 4); return f;
    }
    std::uint16_t f_to_h(float f)
    {
        std::uint32_t b; std::memcpy(&b, &f, 4);
        const std::uint32_t s = (b >> 16) & 0x8000u, a = b & 0x7FFFFFFFu;
        if (a >= 0x7F800000u) return std::uint16_t(a == 0x7F800000u ? (s | 0x7C00u) : (s | 0x7E00u | ((a >> 13) & 0x3FFu)));
        if (a >= 0x477FF000u) return std::uint16_t(s | 0x7C00u);
        if (a < 0x38800000u)
        {
            const std::uint32_t ex = a >> 23; if (ex < 102) return std::uint16_t(s);
            const std::uint32_t man = (a & 0x7FFFFFu) | 0x800000u, sh = 126 - ex; std::uint32_t q = man >> sh; const std::uint32_t r = man & ((1u << sh) - 1u), hf = 1u << (sh - 1);
            if (r > hf || (r == hf && (q & 1u))) ++q;
            return std::uint16_t(s | q);
        }
        std::uint32_t h = (((a >> 23) - 112u) << 10) | ((a & 0x7FFFFFu) >> 13); const std::uint32_t r = a & 0x1FFFu;
        if (r > 0x1000u || (r == 0x1000u && (h & 1u))) ++h;
        return std::uint16_t(s | h);
    }
    bool is_nan_bf16(std::uint16_t b) { return (b & 0x7F80) == 0x7F80 && (b & 0x7F); }
    bool is_nan_h(std::uint16_t b) { return (b & 0x7C00) == 0x7C00 && (b & 0x3FF); }

    void run_pair(std::uint16_t port, const std::function<void(tbccl::Communicator &)> &f0, const std::function<void(tbccl::Communicator &)> &f1)
    {
        tbccl::CommunicatorOptions o0; o0.rank = 0; o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", std::uint16_t(port + 1)}};
        tbccl::CommunicatorOptions o1 = o0; o1.rank = 1;
        std::exception_ptr e0, e1;
        std::thread t([&] { try { auto c = tbccl::Communicator::create(o1); f1(*c); } catch (...) { e1 = std::current_exception(); } });
        try { auto c = tbccl::Communicator::create(o0); f0(*c); } catch (...) { e0 = std::current_exception(); }
        t.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
    }

    // `kind0` is where rank 0's buffer lives; rank 1 is always host memory.
    void sum16(tbccl::DataType dt, bool cuda_rank0, std::uint16_t port)
    {
        const std::size_t n = 4099;
        std::vector<std::uint16_t> a(n), b(n);
        std::uint32_t st = 12345;
        for (auto &x : a) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; x = std::uint16_t(st >> 8); }
        for (auto &x : b) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; x = std::uint16_t(st >> 8); }
        std::vector<std::uint16_t> r0(n), r1 = b;
        const bool bf = dt == tbccl::DataType::BFloat16;
        run_pair(port,
            [&](tbccl::Communicator &c) {
                const std::size_t bytes = n * 2;
#ifdef CONSUMER_WITH_CUDA
                if (cuda_rank0)
                {
                    void *d = nullptr; expect(cudaMalloc(&d, bytes) == cudaSuccess, "cudaMalloc");
                    expect(cudaMemcpy(d, a.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess, "H2D");
                    tbccl::BufferView v{tbccl::MemoryKind::Cuda, d, bytes, 0};
                    auto w = c.all_reduce(v, v, n, dt, tbccl::ReduceOp::Sum); w.wait(); expect(!w.has_error(), "cuda all_reduce: " + w.error());
                    expect(cudaMemcpy(r0.data(), d, bytes, cudaMemcpyDeviceToHost) == cudaSuccess, "D2H"); cudaFree(d);
                    return;
                }
#endif
                (void)cuda_rank0; r0 = a;
                tbccl::BufferView v{tbccl::MemoryKind::Host, r0.data(), bytes, -1};
                auto w = c.all_reduce(v, v, n, dt, tbccl::ReduceOp::Sum); w.wait(); expect(!w.has_error(), "host all_reduce: " + w.error());
            },
            [&](tbccl::Communicator &c) {
                tbccl::BufferView v{tbccl::MemoryKind::Host, r1.data(), n * 2, -1};
                auto w = c.all_reduce(v, v, n, dt, tbccl::ReduceOp::Sum); w.wait(); expect(!w.has_error(), "host peer all_reduce: " + w.error());
            });
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint16_t want = bf ? f_to_bf16(bf16_to_f(a[i]) + bf16_to_f(b[i])) : f_to_h(h_to_f(a[i]) + h_to_f(b[i]));
            const bool nan = bf ? is_nan_bf16(want) : is_nan_h(want);
            const bool ok = nan ? (bf ? is_nan_bf16(r0[i]) && is_nan_bf16(r1[i]) : is_nan_h(r0[i]) && is_nan_h(r1[i])) : (r0[i] == want && r1[i] == want);
            expect(ok, std::string(bf ? "bfloat16" : "float16") + (cuda_rank0 ? " (cuda rank 0)" : " (host)") + " sum mismatch at " + std::to_string(i));
        }
    }

    void sum8(tbccl::DataType dt, std::uint16_t port)
    {
        const std::size_t n = 4099;
        std::vector<std::uint8_t> a(n), b(n), r0, r1;
        for (std::size_t i = 0; i < n; ++i) { a[i] = std::uint8_t(i * 37 + 11); b[i] = std::uint8_t(i * 101 + 7); }
        r0 = a; r1 = b;
        run_pair(port,
            [&](tbccl::Communicator &c) { tbccl::BufferView v{tbccl::MemoryKind::Host, r0.data(), n, -1}; auto w = c.all_reduce(v, v, n, dt, tbccl::ReduceOp::Sum); w.wait(); expect(!w.has_error(), w.error()); },
            [&](tbccl::Communicator &c) { tbccl::BufferView v{tbccl::MemoryKind::Host, r1.data(), n, -1}; auto w = c.all_reduce(v, v, n, dt, tbccl::ReduceOp::Sum); w.wait(); expect(!w.has_error(), w.error()); });
        for (std::size_t i = 0; i < n; ++i) expect(r0[i] == std::uint8_t(a[i] + b[i]) && r1[i] == r0[i], "8-bit sum mismatch at " + std::to_string(i));
    }

    // Phase 50: the installed headers expose the N-rank API (rank directory, communicator id, pre-bound listeners, barrier). Three ranks on loopback with
    // kernel-assigned ports; all_reduce Float32 SUM of 1+2+3, plus a rejected Float16 reduction (N>2 semantics are not defined).
    void three_ranks()
    {
        constexpr std::size_t n = 3;
        const auto id = tbccl::CommunicatorId::generate();
        std::vector<std::shared_ptr<tbccl::CommunicatorListeners>> listeners(n);
        tbccl::RankDirectory dir;
        for (std::size_t r = 0; r < n; ++r)
        {
            tbccl::RankEndpoint e;
            e.rank = r;
            if (tbccl::rank_accepts_connections(r, n))
            {
                listeners[r] = tbccl::CommunicatorListeners::bind("127.0.0.1");
                e.control = listeners[r]->control();
                e.data = listeners[r]->data();
            }
            dir.entries.push_back(e);
        }
        std::vector<std::string> errors(n);
        std::vector<std::thread> threads;
        for (std::size_t r = 0; r < n; ++r)
            threads.emplace_back([&, r] {
                try
                {
                    tbccl::CommunicatorOptions o;
                    o.rank = r;
                    o.world_size = n;
                    o.communicator_id = id;
                    o.rank_directory = dir;
                    o.listeners = listeners[r];
                    auto comm = tbccl::Communicator::create(o);
                    expect(comm->world_size() == n && comm->capabilities().world_size() == n, "world size 3");
                    comm->barrier().wait();
                    std::vector<float> v(64, static_cast<float>(r + 1));
                    tbccl::BufferView view{tbccl::MemoryKind::Host, v.data(), v.size() * 4, 0};
                    auto w = comm->all_reduce(view, view, v.size(), tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                    w.wait();
                    expect(!w.has_error() && v[0] == 6.0f && v[63] == 6.0f, "3-rank all_reduce");
                    bool rejected = false;
                    try { comm->all_reduce(view, view, 32, tbccl::DataType::Float16, tbccl::ReduceOp::Sum); }
                    catch (const std::runtime_error &e) { rejected = std::string(e.what()).find("N>2") != std::string::npos; }
                    expect(rejected, "Float16 N>2 rejected");
                    comm->barrier().wait();
                }
                catch (const std::exception &e)
                {
                    errors[r] = e.what();
                }
            });
        for (auto &t : threads) t.join();
        for (std::size_t r = 0; r < n; ++r) expect(errors[r].empty(), "rank " + std::to_string(r) + ": " + errors[r]);
    }
} // namespace

int main()
{
    try
    {
        // the installed headers expose the Phase 49 types and the single support predicate
        expect(static_cast<int>(tbccl::DataType::BFloat16) == 7 && static_cast<int>(tbccl::DataType::Float32) == 2, "installed enum values");
        expect(tbccl::datatype_size(tbccl::DataType::Float16) == 2 && tbccl::datatype_size(tbccl::DataType::UInt8) == 1, "installed datatype_size");
        expect(tbccl::reduction_supported(tbccl::DataType::BFloat16, tbccl::ReduceOp::Sum) && !tbccl::reduction_supported(tbccl::DataType::BFloat16, tbccl::ReduceOp::Max), "support matrix");
        bool rejected = false;
        try { tbccl::validate_reduction(tbccl::DataType::Float16, tbccl::ReduceOp::Product); } catch (const std::runtime_error &e) { rejected = std::string(e.what()).rfind("unsupported:", 0) == 0; }
        expect(rejected, "validate_reduction message");

        three_ranks();
        std::uint16_t port = 29270;
        sum16(tbccl::DataType::BFloat16, false, port); port += 2;
        sum16(tbccl::DataType::Float16, false, port); port += 2;
        sum8(tbccl::DataType::Int8, port); port += 2;
        sum8(tbccl::DataType::UInt8, port);
#ifdef CONSUMER_WITH_CUDA
        tbccl::register_cuda_support();
        sum16(tbccl::DataType::BFloat16, true, 29270);
        sum16(tbccl::DataType::Float16, true, 29272);
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "external consumer ok (host"
#ifdef CONSUMER_WITH_CUDA
              << " + cuda"
#endif
              << ")\n";
    return 0;
}
