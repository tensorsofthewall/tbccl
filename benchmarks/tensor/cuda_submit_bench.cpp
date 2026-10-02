// The persistent CUDA staging work microbenchmark: Communicator::all_reduce call-return ("submit"), wait
// remainder and total, for a CUDA rank 0 against a Host rank 1 (in-process loopback). Per size: one COLD
// operation (may allocate/grow staging) then --warm WARM operations. If the cuda_alloc_interpose shim is
// preloaded, per-phase cudaMallocHost/cudaFreeHost/cudaMalloc/cudaFree counts are printed too. Never
// averages cold with warm.
#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
static void cu(cudaError_t e, const char *w) { if (e) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e)); }
using SnapFn = void (*)(unsigned long long *);
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }


static bool port_free(std::uint16_t port)
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

int main(int argc, char **argv)
{
    std::vector<std::size_t> sizes = {1, 4, 16, 25};
    int warm = 20;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--sizes")) { sizes.clear(); for (char *t = std::strtok(argv[++i], ","); t; t = std::strtok(nullptr, ",")) sizes.push_back(std::strtoul(t, nullptr, 10)); }
        else if (!std::strcmp(argv[i], "--warm")) warm = std::atoi(argv[++i]);
    }
    auto snap = reinterpret_cast<SnapFn>(dlsym(RTLD_DEFAULT, "tbccl_interpose_snapshot"));
    tbccl::register_cuda_support();
    std::uint16_t port = 31000;
    while (!(port_free(port) && port_free(port + 1) && port_free(port + 1000) && port_free(port + 1001))) port += 4;

    const std::size_t max_bytes = *std::max_element(sizes.begin(), sizes.end()) << 20;
    const std::size_t ops_per_size = 1 + warm;
    tbccl::CommunicatorOptions o0;
    o0.rank = 0;
    o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
    tbccl::CommunicatorOptions o1 = o0;
    o1.rank = 1;

    std::exception_ptr e1;
    std::thread peer([&] {
        try
        {
            auto c = tbccl::Communicator::create(o1);
            std::vector<float> h(max_bytes / 4, 1.0f);
            for (std::size_t mib : sizes)
                for (std::size_t k = 0; k < ops_per_size; ++k)
                {
                    std::fill(h.begin(), h.begin() + (mib << 20) / 4, 1.0f);
                    auto w = c->all_reduce({tbccl::MemoryKind::Host, h.data(), (mib << 20), -1}, {tbccl::MemoryKind::Host, h.data(), (mib << 20), -1}, (mib << 20) / 4, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                    w.wait();
                }
        }
        catch (...) { e1 = std::current_exception(); }
    });

    int rc = 0;
    try
    {
        auto c = tbccl::Communicator::create(o0);
        cudaStream_t s;
        cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        void *d;
        cu(cudaMalloc(&d, max_bytes), "malloc");
        tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
        std::printf("%6s | %-5s %9s %9s %9s | counts per phase: mallocHost freeHost malloc free\n", "MiB", "phase", "submit_us", "rest_us", "total_us");
        for (std::size_t mib : sizes)
        {
            const std::size_t bytes = mib << 20, count = bytes / 4;
            std::vector<double> sub, rest, tot;
            unsigned long long before[8] = {}, after_cold[8] = {}, after_warm[8] = {};
            for (std::size_t k = 0; k < ops_per_size; ++k)
            {
                std::vector<float> ones(count, 1.0f);
                cu(cudaMemcpyAsync(d, ones.data(), bytes, cudaMemcpyHostToDevice, s), "fill");
                cu(cudaStreamSynchronize(s), "fillsync");
                if (k == 0 && snap) snap(before);
                auto t0 = clk::now();
                auto w = c->all_reduce({tbccl::MemoryKind::Cuda, d, bytes, 0}, {tbccl::MemoryKind::Cuda, d, bytes, 0}, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, ctx);
                auto t1 = clk::now();
                w.wait();
                auto t2 = clk::now();
                if (w.has_error()) throw std::runtime_error("all_reduce failed: " + w.error());
                if (k == 0 && snap) snap(after_cold);
                if (k == 0 || k == ops_per_size - 1)
                {
                    std::vector<float> out(count);
                    cu(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost), "readback");
                    for (std::size_t i = 0; i < count; i += 4097) if (out[i] != 2.0f) throw std::runtime_error("result mismatch");
                }
                if (k == 0) { std::printf("%6zu | %-5s %9.0f %9.0f %9.0f |", mib, "cold", us(t0, t1), us(t1, t2), us(t0, t2)); if (snap) std::printf(" %llu %llu %llu %llu", after_cold[0] - before[0], after_cold[3] - before[3], after_cold[5] - before[5], after_cold[6] - before[6]); std::printf("\n"); }
                else { sub.push_back(us(t0, t1)); rest.push_back(us(t1, t2)); tot.push_back(us(t0, t2)); }
            }
            if (snap) snap(after_warm);
            std::printf("%6zu | %-5s %9.0f %9.0f %9.0f |", mib, "warm", med(sub), med(rest), med(tot));
            if (snap) std::printf(" %llu %llu %llu %llu (all %d warm ops)", after_warm[0] - after_cold[0], after_warm[3] - after_cold[3], after_warm[5] - after_cold[5], after_warm[6] - after_cold[6], warm);
            std::printf("\n");
        }
        cudaFree(d);
        cudaStreamDestroy(s);
    }
    catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); rc = 1; }
    peer.join();
    if (e1) { try { std::rethrow_exception(e1); } catch (const std::exception &e) { std::fprintf(stderr, "peer FAIL: %s\n", e.what()); rc = 1; } }
    return rc;
}
