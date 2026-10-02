// Diagnostic LD_PRELOAD shim: counts and times cudaMallocHost/cudaFreeHost/cudaMalloc/cudaFree calls
// made by anything in the process (independent of TBCCL's own counters). Phase 44 uses it to prove
// that steady-state collectives no longer allocate/free payload-sized pinned memory.
// Read with dlsym(RTLD_DEFAULT, "tbccl_interpose_snapshot").
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace
{
std::atomic<std::uint64_t> c[8]; // mallocHost n, bytes, ns | freeHost n, ns | malloc n | free n | (unused)
using clk = std::chrono::steady_clock;
std::uint64_t since(clk::time_point t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count(); }
template <typename F> F next(const char *name) { return reinterpret_cast<F>(dlsym(RTLD_NEXT, name)); }
} // namespace

extern "C"
{
void tbccl_interpose_snapshot(std::uint64_t out[8])
{
    for (int i = 0; i < 8; ++i) out[i] = c[i].load();
}

int cudaMallocHost(void **p, std::size_t n)
{
    static auto f = next<int (*)(void **, std::size_t)>("cudaMallocHost");
    auto t = clk::now();
    int r = f(p, n);
    c[0]++; c[1] += n; c[2] += since(t);
    return r;
}
int cudaFreeHost(void *p)
{
    static auto f = next<int (*)(void *)>("cudaFreeHost");
    auto t = clk::now();
    int r = f(p);
    c[3]++; c[4] += since(t);
    return r;
}
int cudaMalloc(void **p, std::size_t n)
{
    static auto f = next<int (*)(void **, std::size_t)>("cudaMalloc");
    c[5]++;
    return f(p, n);
}
int cudaFree(void *p)
{
    static auto f = next<int (*)(void *)>("cudaFree");
    c[6]++;
    return f(p);
}
}
