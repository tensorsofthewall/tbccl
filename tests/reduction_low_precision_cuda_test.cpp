// real-GPU correctness of Float16 / BFloat16 SUM and Int8 / UInt8 modulo-256 SUM.
//
//   1. Kernel level (CudaExternalReduceBackend, the provider behind Communicator::all_reduce on MemoryKind::Cuda): the device result equals the
//      documented host semantics (float32 widen, one float32 add, round once, ties-to-even) BIT FOR BIT, for an edge-pattern cross product, random
//      vectors of assorted lengths and, when TBCCL_EXHAUSTIVE_LOWP=1, every one of the 2^32 (a, b) pairs of each format. NaN results are compared
//      as "is NaN" (payloads are not preserved identically across host and device).
//   2. Communicator level on loopback with real devices: CUDA <-> Host in both rank orientations and CUDA <-> CUDA (same GPU), for the vLLM pipeline-parallel study
//      TP-audit shapes ([1,1024] .. [512,1024]) and odd sizes, in place.
//   3. Producer readiness: the CUDA operand is written by a delayed kernel on a non-blocking user stream and all_reduce is submitted immediately
//      (no host synchronization); the result must contain the delayed write.

#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include "tensor/cuda_external_async_backend.hpp"

#include "low_precision.hpp"
#include "lowp_test_support.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    using lowp_test::Bf16;
    using lowp_test::edge_patterns;
    using lowp_test::expected_sum;
    using lowp_test::Fp16;
    using lowp_test::random_patterns;
    using lowp_test::same;

    void expect(bool c, const std::string &m)
    {
        if (!c) throw std::runtime_error("assertion failed: " + m);
    }

    void cu(cudaError_t e, const char *what)
    {
        if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }

    struct DevBuf
    {
        void *p = nullptr;
        std::size_t bytes;
        explicit DevBuf(std::size_t n) : bytes(n) { cu(cudaMalloc(&p, n), "cudaMalloc"); }
        ~DevBuf() { cudaFree(p); }
        DevBuf(const DevBuf &) = delete;
        DevBuf &operator=(const DevBuf &) = delete;
        void upload(const std::vector<std::uint16_t> &h) { cu(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice), "H2D"); }
        std::vector<std::uint16_t> download(std::size_t count) const
        {
            std::vector<std::uint16_t> h(count);
            cu(cudaMemcpy(h.data(), p, count * 2, cudaMemcpyDeviceToHost), "D2H");
            return h;
        }
    };

    // ----- 1. kernel level ---------------------------------------------------------------------------------------------------------------
    template <class F>
    void backend_sum_and_check(const std::vector<std::uint16_t> &a, const std::vector<std::uint16_t> &b, const char *what)
    {
        const std::size_t count = a.size();
        DevBuf dst(count * 2), src(count * 2);
        dst.upload(a);
        src.upload(b);
        tbccl_bench::tensor::CudaExternalReduceBackend backend(dst.p, src.p, /*stream=*/nullptr);
        backend.reduce_sum(count, F::dt);
        const auto got = dst.download(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::uint16_t want = expected_sum<F>(a[i], b[i]);
            if (!same<F>(got[i], want))
            {
                throw std::runtime_error(std::string("assertion failed: ") + F::name + " CUDA sum differs from host semantics (" + what + "): a=" +
                                         std::to_string(a[i]) + " b=" + std::to_string(b[i]) + " got=" + std::to_string(got[i]) + " want=" +
                                         std::to_string(want));
            }
        }
    }

    template <class F>
    void test_kernel_edge_and_random()
    {
        const auto patterns = edge_patterns<F>();
        std::vector<std::uint16_t> a, b;
        for (std::uint16_t x : patterns)
            for (std::uint16_t y : patterns)
            {
                a.push_back(x);
                b.push_back(y);
            }
        backend_sum_and_check<F>(a, b, "edge cross product");
        for (std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{255}, std::size_t{2048}, std::size_t{1} << 20})
        {
            backend_sum_and_check<F>(random_patterns<F>(count, 424242u + static_cast<std::uint32_t>(count)),
                                     random_patterns<F>(count, 8675309u + static_cast<std::uint32_t>(count)), "random");
        }
        std::cout << "[PASS] test_kernel_edge_and_random<" << F::name << ">\n";
    }

    // All 2^32 ordered pairs: for each a, dst = a repeated 65536 times, src = every pattern b.
    template <class F>
    void test_kernel_exhaustive()
    {
        std::vector<std::uint16_t> all(65536);
        for (std::uint32_t b = 0; b < 65536; ++b) all[b] = static_cast<std::uint16_t>(b);
        std::vector<float> dec(65536);
        for (std::uint32_t b = 0; b < 65536; ++b) dec[b] = F::dec(static_cast<std::uint16_t>(b));
        DevBuf dst(65536 * 2), src(65536 * 2);
        src.upload(all);
        tbccl_bench::tensor::CudaExternalReduceBackend backend(dst.p, src.p, nullptr);
        std::vector<std::uint16_t> a_rep(65536), got;
        std::uint64_t mismatches = 0;
        for (std::uint32_t a = 0; a < 65536; ++a)
        {
            std::fill(a_rep.begin(), a_rep.end(), static_cast<std::uint16_t>(a));
            dst.upload(a_rep);
            backend.reduce_sum(65536, F::dt);
            got = dst.download(65536);
            for (std::uint32_t b = 0; b < 65536; ++b)
            {
                const std::uint16_t want = F::enc(dec[a] + dec[b]);
                if (!same<F>(got[b], want)) ++mismatches;
            }
        }
        expect(mismatches == 0, std::string(F::name) + ": exhaustive CUDA vs host sum, " + std::to_string(mismatches) + " mismatching pairs");
        std::cout << "[PASS] test_kernel_exhaustive<" << F::name << "> (all 2^32 pairs)\n";
    }

    // ----- 2. communicator level ---------------------------------------------------------------------------------------------------------------
    // Control ports 29240-29251 (pairs of two), data plane = control + 1000 = 30240-30251: both below the kernel's ephemeral range (32768+), where a
    // stray closed client connection holding the same port in TIME_WAIT would make bind() fail. Shared with the host low-precision tests (ctest
    // serializes them); cases run one after another and the listener sets SO_REUSEADDR, so the six slots are reused in turn.
    std::uint16_t g_next_port = 29240;

    std::uint16_t take_port()
    {
        const std::uint16_t p = g_next_port;
        g_next_port = static_cast<std::uint16_t>(g_next_port + 2);
        if (g_next_port >= 29252) g_next_port = 29240;
        return p;
    }

    void run_pair(std::uint16_t port, const std::function<void(tbccl::Communicator &)> &f0, const std::function<void(tbccl::Communicator &)> &f1)
    {
        tbccl::CommunicatorOptions o0;
        o0.rank = 0;
        o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        tbccl::CommunicatorOptions o1 = o0;
        o1.rank = 1;
        std::exception_ptr e0, e1;
        std::thread t([&] {
            try { auto c = tbccl::Communicator::create(o1); f1(*c); } catch (...) { e1 = std::current_exception(); }
        });
        try { auto c = tbccl::Communicator::create(o0); f0(*c); } catch (...) { e0 = std::current_exception(); }
        t.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
    }

    enum class Kind { Host, Cuda };

    // Each rank holds `kind[rank]` memory with its own random input; both ranks must end with the host-semantics sum.
    template <class F>
    void communicator_case(Kind k0, Kind k1, std::size_t count)
    {
        const Kind kinds[2] = {k0, k1};
        const auto in0 = random_patterns<F>(count, 11u + static_cast<std::uint32_t>(count));
        const auto in1 = random_patterns<F>(count, 6007u + static_cast<std::uint32_t>(count));
        std::vector<std::uint16_t> result[2];
        auto rank_fn = [&](std::size_t rank) {
            return [&, rank](tbccl::Communicator &comm) {
                const auto &mine = rank == 0 ? in0 : in1;
                const std::size_t bytes = count * 2;
                if (kinds[rank] == Kind::Cuda)
                {
                    cudaStream_t s;
                    cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
                    DevBuf d(bytes);
                    cu(cudaMemcpyAsync(d.p, mine.data(), bytes, cudaMemcpyHostToDevice, s), "H2D async");
                    tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
                    tbccl::BufferView view{tbccl::MemoryKind::Cuda, d.p, bytes, 0};
                    auto w = comm.all_reduce(view, view, count, F::dt, tbccl::ReduceOp::Sum, ctx);
                    w.wait();
                    expect(!w.has_error(), std::string(F::name) + " cuda all_reduce: " + w.error());
                    result[rank] = d.download(count);
                    cudaStreamDestroy(s);
                }
                else
                {
                    std::vector<std::uint16_t> buf = mine;
                    tbccl::BufferView view{tbccl::MemoryKind::Host, buf.data(), bytes, -1};
                    auto w = comm.all_reduce(view, view, count, F::dt, tbccl::ReduceOp::Sum);
                    w.wait();
                    expect(!w.has_error(), std::string(F::name) + " host all_reduce: " + w.error());
                    result[rank] = buf;
                }
            };
        };
        run_pair(take_port(), rank_fn(0), rank_fn(1));
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::uint16_t want = expected_sum<F>(in0[i], in1[i]);
            expect(same<F>(result[0][i], want) && same<F>(result[1][i], want),
                   std::string(F::name) + ": all_reduce mismatch (rank0=" + (k0 == Kind::Cuda ? "cuda" : "host") + ", rank1=" + (k1 == Kind::Cuda ? "cuda" : "host") +
                       ", count " + std::to_string(count) + ", index " + std::to_string(i) + ")");
        }
    }

    template <class F>
    void test_communicator_shapes()
    {
        // The vLLM pipeline-parallel study work TP-audit shapes [T, 1024] for T = 1, 4, 12, 128, 512, plus 1 element, an odd count
        // and 1 MiB of 16-bit elements.
        const std::size_t counts[] = {1, 17, 1 * 1024, 4 * 1024, 12 * 1024, 128 * 1024, 512 * 1024};
        for (std::size_t count : counts)
        {
            communicator_case<F>(Kind::Cuda, Kind::Host, count);
            communicator_case<F>(Kind::Host, Kind::Cuda, count);
            communicator_case<F>(Kind::Cuda, Kind::Cuda, count);
        }
        std::cout << "[PASS] test_communicator_shapes<" << F::name << ">\n";
    }


    // ----- integer types (Int8 / UInt8): modulo-256 SUM, bit exact on device ----------------------------------------------------------------
    template <class T>
    struct IntTraits;
    template <>
    struct IntTraits<std::int8_t>
    {
        static constexpr const char *name = "int8";
        static constexpr tbccl::DataType dt = tbccl::DataType::Int8;
    };
    template <>
    struct IntTraits<std::uint8_t>
    {
        static constexpr const char *name = "uint8";
        static constexpr tbccl::DataType dt = tbccl::DataType::UInt8;
    };

    template <class T>
    T int_reference_sum(T a, T b)
    {
        return static_cast<T>(static_cast<std::uint8_t>((static_cast<unsigned>(static_cast<std::uint8_t>(a)) + static_cast<unsigned>(static_cast<std::uint8_t>(b))) & 0xFFu));
    }

    template <class T>
    void test_int_kernel_all_pairs()
    {
        std::vector<T> a(65536), b(65536);
        for (std::uint32_t i = 0; i < 65536; ++i)
        {
            a[i] = static_cast<T>(static_cast<std::uint8_t>(i >> 8));
            b[i] = static_cast<T>(static_cast<std::uint8_t>(i & 0xFF));
        }
        DevBuf dst(a.size()), src(b.size());
        cu(cudaMemcpy(dst.p, a.data(), a.size(), cudaMemcpyHostToDevice), "H2D");
        cu(cudaMemcpy(src.p, b.data(), b.size(), cudaMemcpyHostToDevice), "H2D");
        tbccl_bench::tensor::CudaExternalReduceBackend backend(dst.p, src.p, nullptr);
        backend.reduce_sum(a.size(), IntTraits<T>::dt);
        std::vector<T> got(a.size());
        cu(cudaMemcpy(got.data(), dst.p, got.size(), cudaMemcpyDeviceToHost), "D2H");
        for (std::size_t i = 0; i < got.size(); ++i)
        {
            expect(got[i] == int_reference_sum(a[i], b[i]), std::string(IntTraits<T>::name) + ": CUDA sum differs from the modulo-256 reference at pair " + std::to_string(i));
        }
        // plan overflow cases on device
        const std::vector<T> x = {static_cast<T>(120), static_cast<T>(127), static_cast<T>(-128), static_cast<T>(-100), static_cast<T>(255), static_cast<T>(200)};
        const std::vector<T> y = {static_cast<T>(100), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-100), static_cast<T>(1), static_cast<T>(100)};
        DevBuf dx(x.size()), dy(y.size());
        cu(cudaMemcpy(dx.p, x.data(), x.size(), cudaMemcpyHostToDevice), "H2D");
        cu(cudaMemcpy(dy.p, y.data(), y.size(), cudaMemcpyHostToDevice), "H2D");
        tbccl_bench::tensor::CudaExternalReduceBackend small(dx.p, dy.p, nullptr);
        small.reduce_sum(x.size(), IntTraits<T>::dt);
        std::vector<T> r(x.size());
        cu(cudaMemcpy(r.data(), dx.p, r.size(), cudaMemcpyDeviceToHost), "D2H");
        for (std::size_t i = 0; i < r.size(); ++i) expect(r[i] == int_reference_sum(x[i], y[i]), std::string(IntTraits<T>::name) + ": CUDA overflow case " + std::to_string(i));
        std::cout << "[PASS] test_int_kernel_all_pairs<" << IntTraits<T>::name << "> (65536 pairs + overflow cases)\n";
    }

    template <class T>
    void int_communicator_case(Kind k0, Kind k1, std::size_t count)
    {
        const Kind kinds[2] = {k0, k1};
        std::vector<T> in0(count), in1(count);
        std::uint32_t st0 = 5u + static_cast<std::uint32_t>(count), st1 = 71u + static_cast<std::uint32_t>(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            in0[i] = static_cast<T>(lowp_test::next_random(st0) >> 9);
            in1[i] = static_cast<T>(lowp_test::next_random(st1) >> 9);
        }
        std::vector<T> result[2];
        auto rank_fn = [&](std::size_t rank) {
            return [&, rank](tbccl::Communicator &comm) {
                const auto &mine = rank == 0 ? in0 : in1;
                if (kinds[rank] == Kind::Cuda)
                {
                    cudaStream_t s;
                    cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
                    DevBuf d(count);
                    cu(cudaMemcpyAsync(d.p, mine.data(), count, cudaMemcpyHostToDevice, s), "H2D async");
                    tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
                    tbccl::BufferView view{tbccl::MemoryKind::Cuda, d.p, count, 0};
                    auto w = comm.all_reduce(view, view, count, IntTraits<T>::dt, tbccl::ReduceOp::Sum, ctx);
                    w.wait();
                    expect(!w.has_error(), std::string(IntTraits<T>::name) + " cuda all_reduce: " + w.error());
                    result[rank].resize(count);
                    cu(cudaMemcpy(result[rank].data(), d.p, count, cudaMemcpyDeviceToHost), "D2H");
                    cudaStreamDestroy(s);
                }
                else
                {
                    std::vector<T> buf = mine;
                    tbccl::BufferView view{tbccl::MemoryKind::Host, buf.data(), count, -1};
                    auto w = comm.all_reduce(view, view, count, IntTraits<T>::dt, tbccl::ReduceOp::Sum);
                    w.wait();
                    expect(!w.has_error(), std::string(IntTraits<T>::name) + " host all_reduce: " + w.error());
                    result[rank] = buf;
                }
            };
        };
        run_pair(take_port(), rank_fn(0), rank_fn(1));
        for (std::size_t i = 0; i < count; ++i)
        {
            const T want = int_reference_sum(in0[i], in1[i]);
            expect(result[0][i] == want && result[1][i] == want, std::string(IntTraits<T>::name) + ": all_reduce mismatch (rank0=" + (k0 == Kind::Cuda ? "cuda" : "host") +
                                                                     ", rank1=" + (k1 == Kind::Cuda ? "cuda" : "host") + ", count " + std::to_string(count) + ", index " + std::to_string(i) + ")");
        }
    }

    template <class T>
    void test_int_communicator_shapes()
    {
        // Element counts matching the [T, 1024] shapes (1 byte per element), odd sizes and 1 MiB.
        for (std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{1024}, std::size_t{4 * 1024}, std::size_t{12 * 1024}, std::size_t{128 * 1024}, std::size_t{1} << 20})
        {
            int_communicator_case<T>(Kind::Cuda, Kind::Host, count);
            int_communicator_case<T>(Kind::Host, Kind::Cuda, count);
            int_communicator_case<T>(Kind::Cuda, Kind::Cuda, count);
        }
        std::cout << "[PASS] test_int_communicator_shapes<" << IntTraits<T>::name << ">\n";
    }

    // ----- 3. producer readiness ---------------------------------------------------------------------------------------------------------------
    template <class F>
    void test_delayed_producer()
    {
        constexpr std::size_t count = 1 << 17;                // 16-bit elements; the delayed kernel writes 32-bit words
        const std::uint16_t h0 = F::enc(1.5f), h1 = F::enc(2.25f);
        const std::uint16_t want = F::enc(F::dec(h0) + F::dec(h1));
        const std::int32_t word = static_cast<std::int32_t>((static_cast<std::uint32_t>(h0) << 16) | h0);
        for (bool cuda_is_rank0 : {true, false})
        {
            DevBuf d(count * 2);
            cu(cudaMemset(d.p, 0, count * 2), "memset");
            // Setup only (before the readiness window): cudaMemset runs on the legacy default stream, which is NOT ordered against the
            // non-blocking producer stream below. Without this, a delayed memset can zero part of the producer's output (found under
            // heavy GPU load: it corrupted the operand even when the producer was synchronized before all_reduce was submitted).
            cu(cudaDeviceSynchronize(), "device sync after the setup memset");
            void *producer = tbccl_bench::tensor::cuda_external_test_launch_delayed_write_i32(d.p, count / 2, word, 2'000'000ull, /*non_blocking=*/true);
            std::vector<std::uint16_t> host(count, h1);
            auto cuda_rank = [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Cuda, d.p, count * 2, 0};
                tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, producer};
                auto w = comm.all_reduce(view, view, count, F::dt, tbccl::ReduceOp::Sum, ctx);
                w.wait();
                expect(!w.has_error(), std::string(F::name) + " delayed-producer cuda rank: " + w.error());
            };
            auto host_rank = [&](tbccl::Communicator &comm) {
                tbccl::BufferView view{tbccl::MemoryKind::Host, host.data(), count * 2, -1};
                auto w = comm.all_reduce(view, view, count, F::dt, tbccl::ReduceOp::Sum);
                w.wait();
                expect(!w.has_error(), std::string(F::name) + " delayed-producer host rank: " + w.error());
            };
            run_pair(take_port(), cuda_is_rank0 ? std::function<void(tbccl::Communicator &)>(cuda_rank) : std::function<void(tbccl::Communicator &)>(host_rank),
                     cuda_is_rank0 ? std::function<void(tbccl::Communicator &)>(host_rank) : std::function<void(tbccl::Communicator &)>(cuda_rank));
            // Independent consumer: a fresh stream, only that stream is synchronized (no device-wide synchronization).
            cudaStream_t other;
            cu(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "stream");
            std::vector<std::uint16_t> got(count);
            cu(cudaMemcpyAsync(got.data(), d.p, count * 2, cudaMemcpyDeviceToHost, other), "D2H");
            cu(cudaStreamSynchronize(other), "sync");
            cudaStreamDestroy(other);
            tbccl_bench::tensor::cuda_external_test_destroy_stream(producer);
            std::size_t bad_cuda = 0, bad_host = 0, first_bad = count;
            std::uint16_t first_got = 0;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (got[i] != want) { if (first_bad == count) { first_bad = i; first_got = got[i]; } ++bad_cuda; }
                if (host[i] != want) ++bad_host;
            }
            expect(bad_cuda == 0 && bad_host == 0,
                   std::string(F::name) + ": delayed producer not honored (cuda_rank=" + (cuda_is_rank0 ? "0" : "1") + "): " + std::to_string(bad_cuda) + " of " +
                       std::to_string(count) + " CUDA elements and " + std::to_string(bad_host) + " host elements wrong; first CUDA bad index " + std::to_string(first_bad) +
                       " (got " + std::to_string(first_got) + ", want " + std::to_string(want) + ", producer h0 " + std::to_string(h0) + ", host h1 " + std::to_string(h1) + ")");
        }
        std::cout << "[PASS] test_delayed_producer<" << F::name << ">\n";
    }

} // namespace

namespace
{
    template <class Fn>
    void timed(const char *label, Fn &&fn)
    {
        const auto start = std::chrono::steady_clock::now();
        fn();
        std::cout << "    (" << label << ": " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s)\n";
    }
} // namespace

int main()
{
    tbccl::register_cuda_support();
    try
    {
        const char *only = std::getenv("TBCCL_LOWP_ONLY");
        const int repeat = std::getenv("TBCCL_LOWP_REPEAT") ? std::atoi(std::getenv("TBCCL_LOWP_REPEAT")) : 1;
        if (only != nullptr && std::string(only) == "delayed")
        {
            for (int r = 0; r < repeat; ++r)
            {
                test_delayed_producer<Fp16>();
                test_delayed_producer<Bf16>();
            }
            std::cout << "All delayed-producer repeats passed (" << repeat << ")\n";
            return 0;
        }
        timed("kernel fp16", [] { test_kernel_edge_and_random<Fp16>(); });
        timed("kernel bf16", [] { test_kernel_edge_and_random<Bf16>(); });
        const char *exhaustive = std::getenv("TBCCL_EXHAUSTIVE_LOWP");
        if (exhaustive != nullptr && std::string(exhaustive) == "1")
        {
            timed("exhaustive fp16", [] { test_kernel_exhaustive<Fp16>(); });
            timed("exhaustive bf16", [] { test_kernel_exhaustive<Bf16>(); });
        }
        timed("communicator fp16", [] { test_communicator_shapes<Fp16>(); });
        timed("communicator bf16", [] { test_communicator_shapes<Bf16>(); });
        timed("int8 kernel", [] { test_int_kernel_all_pairs<std::int8_t>(); });
        timed("uint8 kernel", [] { test_int_kernel_all_pairs<std::uint8_t>(); });
        timed("int8 communicator", [] { test_int_communicator_shapes<std::int8_t>(); });
        timed("uint8 communicator", [] { test_int_communicator_shapes<std::uint8_t>(); });
        timed("delayed fp16", [] { test_delayed_producer<Fp16>(); });
        timed("delayed bf16", [] { test_delayed_producer<Bf16>(); });
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
    std::cout << "All low-precision CUDA reduction tests passed.\n";
    return 0;
}
