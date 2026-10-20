// Producer/consumer ordering of CUDA memory through the Communicator, run as TWO PROCESSES (one rank each, same GPU, loopback) so that every rank can have its own
// TBCCL_STAGED_CHUNK_BYTES. Kernels produce the send data asynchronously on a user stream and the send is submitted without synchronising; received data is consumed by
// kernels without any host synchronisation after Work::wait(); a slow kernel that still reads a buffer is ordered before the receive that overwrites it.
//
//   communicator_cuda_stream_chunk_test --rank <0|1> --port <p> [--scenarios a,b,...]
//
// Scenarios: zero, producer, delayed, outstanding, war, repeated, collectives, abort. Every check prints one line; the exit status is non-zero if any check failed.
// Both ranks must run the same scenario list. See tests/communicator_cuda_stream_chunk_test.sh for the driver that starts the two ranks.

#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
constexpr std::size_t KiB = 1024, MiB = 1024 * KiB;

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string &message)
{
    ++g_checks;
    if (!condition)
    {
        ++g_failures;
        std::printf("FAIL %s\n", message.c_str());
        std::fflush(stdout);
    }
}

void cu(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(status));
}
#define CU(x) cu((x), #x)

__device__ __forceinline__ std::uint8_t pat(std::uint32_t seed, std::size_t i) { return static_cast<std::uint8_t>(seed * 131u + i * 7u + (i >> 9) * 3u + 5u); }

__global__ void k_spin(unsigned long long cycles)
{
    const long long start = clock64();
    while (clock64() - start < static_cast<long long>(cycles)) {}
}

__global__ void k_fill(std::uint8_t *p, std::size_t n, std::uint32_t seed)
{
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) p[i] = pat(seed, i);
}

__global__ void k_check(const std::uint8_t *p, std::size_t n, std::uint32_t seed, unsigned long long *bad)
{
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        if (p[i] != pat(seed, i)) atomicAdd(bad, 1ull);
}

__global__ void k_fill_f32(float *p, std::size_t n, float scale)
{
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) p[i] = scale * static_cast<float>((i % 251) + 1);
}

__global__ void k_check_f32(const float *p, std::size_t n, float scale, unsigned long long *bad)
{
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        if (p[i] != scale * static_cast<float>((i % 251) + 1)) atomicAdd(bad, 1ull);
}

// a slow reader: spins, then copies src to dst (used to prove that a receive does not overwrite a buffer an earlier kernel on the same stream still reads)
__global__ void k_slow_copy(const std::uint8_t *src, std::uint8_t *dst, std::size_t n, unsigned long long cycles)
{
    const long long start = clock64();
    while (clock64() - start < static_cast<long long>(cycles)) {} // every thread waits, so no byte is read before the delay is over
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) dst[i] = src[i];
}

const unsigned long long kCyclesPerMs = 1'500'000ull; // conservative: the GPU idles at a lower clock, so the delay is at least this many milliseconds

struct Stream
{
    cudaStream_t s = nullptr;
    Stream() { CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
};

struct Dev
{
    void *p = nullptr;
    std::size_t bytes;
    explicit Dev(std::size_t n) : bytes(n) { CU(cudaMalloc(&p, n ? n : 1)); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    std::uint8_t *u8() { return static_cast<std::uint8_t *>(p); }
    float *f32() { return static_cast<float *>(p); }
};

tbccl::BufferView view(Dev &d) { return {tbccl::MemoryKind::Cuda, d.p, d.bytes, 0}; }
tbccl::ExecutionContext ctx(Stream &s) { return {tbccl::ExecutionContextKind::CudaStream, s.s}; }

void fill(Dev &d, std::uint32_t seed, Stream &s, unsigned long long spin_cycles = 0)
{
    if (spin_cycles) k_spin<<<1, 1, 0, s.s>>>(spin_cycles);
    k_fill<<<256, 256, 0, s.s>>>(d.u8(), d.bytes, seed);
    CU(cudaGetLastError());
}

// Consume `d` with a kernel on its own stream, with NO synchronisation of the stream the Work used: the Work's completion alone must make the data visible.
unsigned long long bad_bytes(Dev &d, std::uint32_t seed)
{
    Stream consumer;
    Dev bad(sizeof(unsigned long long));
    CU(cudaMemsetAsync(bad.p, 0, sizeof(unsigned long long), consumer.s));
    k_check<<<256, 256, 0, consumer.s>>>(d.u8(), d.bytes, seed, static_cast<unsigned long long *>(bad.p));
    unsigned long long host = 0;
    CU(cudaMemcpyAsync(&host, bad.p, sizeof(host), cudaMemcpyDeviceToHost, consumer.s));
    CU(cudaStreamSynchronize(consumer.s));
    return host;
}

unsigned long long bad_f32(Dev &d, float scale)
{
    Stream consumer;
    Dev bad(sizeof(unsigned long long));
    CU(cudaMemsetAsync(bad.p, 0, sizeof(unsigned long long), consumer.s));
    k_check_f32<<<256, 256, 0, consumer.s>>>(d.f32(), d.bytes / 4, scale, static_cast<unsigned long long *>(bad.p));
    unsigned long long host = 0;
    CU(cudaMemcpyAsync(&host, bad.p, sizeof(host), cudaMemcpyDeviceToHost, consumer.s));
    CU(cudaStreamSynchronize(consumer.s));
    return host;
}

std::string str(std::size_t n) { return std::to_string(n); }

double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

void wait_ok(tbccl::Work &w, const std::string &what)
{
    w.wait();
    check(!w.has_error(), what + ": " + w.error());
}

const std::vector<std::size_t> kSizes = {1, 4097, 64 * KiB, 256 * KiB, 300000, 512 * KiB + 3, MiB, MiB + 12347, 3 * MiB + 1, 9 * MiB};

// The sender produces with a (delayed) kernel and submits at once; the receiver consumes with a kernel right after wait(). Both directions.
void scenario_producer(tbccl::Communicator &comm, std::size_t rank, bool delayed)
{
    const unsigned long long spin = (delayed ? 250 : 20) * kCyclesPerMs;
    std::uint32_t seed = delayed ? 40 : 10;
    for (int direction = 0; direction < 2; ++direction)
    {
        const std::size_t sender = static_cast<std::size_t>(direction), receiver = 1 - sender;
        for (std::size_t size : delayed ? std::vector<std::size_t>{64 * KiB, MiB + 12347, 9 * MiB} : kSizes)
        {
            ++seed;
            comm.barrier().wait();
            if (rank == sender)
            {
                Stream a;
                Dev buf(size);
                cudaEvent_t done;
                CU(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
                fill(buf, seed, a, spin);
                CU(cudaEventRecord(done, a.s));
                const auto t0 = Clock::now();
                auto w = comm.send(view(buf), size, tbccl::DataType::UInt8, receiver, ctx(a)); // no synchronisation between the kernel launch and the submission
                const double submit_us = us_since(t0);
                wait_ok(w, "send " + str(size));
                const double total_us = us_since(t0);
                check(cudaEventQuery(done) == cudaSuccess, "send " + str(size) + " completed before its producer kernel finished");
                std::printf("ok producer%s dir=%d size=%zu submit_us=%.0f complete_us=%.0f\n", delayed ? "-delayed" : "", direction, size, submit_us, total_us);
                CU(cudaEventDestroy(done));
            }
            else
            {
                Stream b;
                Dev buf(size);
                CU(cudaMemsetAsync(buf.p, 0xEE, size, b.s));
                auto w = comm.recv(view(buf), size, tbccl::DataType::UInt8, sender, ctx(b));
                wait_ok(w, "recv " + str(size));
                check(bad_bytes(buf, seed) == 0, "recv " + str(size) + ": consumer kernel after wait() saw wrong bytes");
            }
        }
    }
}

// Several transfers in flight at once, each from its own producer stream; waited in a different order than submitted.
void scenario_outstanding(tbccl::Communicator &comm, std::size_t rank)
{
    const std::vector<std::size_t> sizes = {300000, MiB, MiB + 12347, 5 * MiB, 4097, 2 * MiB};
    comm.barrier().wait();
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<std::unique_ptr<Dev>> bufs;
    std::vector<tbccl::Work> works;
    for (std::size_t k = 0; k < sizes.size(); ++k)
    {
        streams.push_back(std::make_unique<Stream>());
        bufs.push_back(std::make_unique<Dev>(sizes[k]));
        if (rank == 0)
        {
            fill(*bufs[k], 70 + static_cast<std::uint32_t>(k), *streams[k], (5 + 10 * (sizes.size() - k)) * kCyclesPerMs);
            works.push_back(comm.send(view(*bufs[k]), sizes[k], tbccl::DataType::UInt8, 1, ctx(*streams[k])));
        }
        else
        {
            works.push_back(comm.recv(view(*bufs[k]), sizes[k], tbccl::DataType::UInt8, 0, ctx(*streams[k])));
        }
    }
    for (std::size_t k = sizes.size(); k-- > 0;)
    {
        wait_ok(works[k], std::string(rank == 0 ? "send" : "recv") + " #" + str(k));
        if (rank == 1) check(bad_bytes(*bufs[k], 70 + static_cast<std::uint32_t>(k)) == 0, "outstanding recv #" + str(k) + " has wrong bytes");
    }
    std::printf("ok outstanding %zu transfers\n", sizes.size());
}

// Write-after-read: a slow kernel on the receive's stream reads the buffer; the receive (same stream) must not overwrite it before that kernel is done.
void scenario_war(tbccl::Communicator &comm, std::size_t rank)
{
    for (std::size_t size : {std::size_t{300000}, MiB + 12347, 4 * MiB})
    {
        comm.barrier().wait();
        if (rank == 0)
        {
            Dev buf(size);
            Stream s;
            fill(buf, 91, s);
            CU(cudaStreamSynchronize(s.s));
            auto w = comm.send(view(buf), size, tbccl::DataType::UInt8, 1);
            wait_ok(w, "war send " + str(size));
        }
        else
        {
            Dev buf(size), snapshot(size);
            Stream s;
            fill(buf, 90, s);
            CU(cudaStreamSynchronize(s.s));
            k_slow_copy<<<64, 256, 0, s.s>>>(buf.u8(), snapshot.u8(), size, 60 * kCyclesPerMs);
            auto w = comm.recv(view(buf), size, tbccl::DataType::UInt8, 0, ctx(s));
            wait_ok(w, "war recv " + str(size));
            CU(cudaStreamSynchronize(s.s));
            check(bad_bytes(snapshot, 90) == 0, "war " + str(size) + ": the earlier kernel read data overwritten by the receive");
            check(bad_bytes(buf, 91) == 0, "war " + str(size) + ": the receive did not deliver the new data");
        }
    }
    std::printf("ok war\n");
}

// Many operations back to back, alternating direction and size; the staging block must stay bounded.
void scenario_repeated(tbccl::Communicator &comm, std::size_t rank)
{
    const std::vector<std::size_t> sizes = {MiB + 12347, 256 * KiB, 4 * MiB + 5, 4097, 2 * MiB};
    const char *env = std::getenv("TBCCL_STAGED_CHUNK_BYTES");
    const std::size_t chunk = env && *env ? static_cast<std::size_t>(std::strtoull(env, nullptr, 10)) : 0;
    std::size_t max_size = 9 * MiB; // the block is shared by every scenario of this run, so the largest payload of ANY scenario sets the unchunked capacity
    for (std::size_t n : sizes) max_size = std::max(max_size, n);
    comm.barrier().wait();
    for (int i = 0; i < 60; ++i)
    {
        const std::size_t size = sizes[static_cast<std::size_t>(i) % sizes.size()];
        const std::size_t sender = static_cast<std::size_t>(i % 2), receiver = 1 - sender;
        const std::uint32_t seed = 200 + static_cast<std::uint32_t>(i);
        Stream s;
        Dev buf(size);
        if (rank == sender)
        {
            fill(buf, seed, s);
            auto w = comm.send(view(buf), size, tbccl::DataType::UInt8, receiver, ctx(s));
            wait_ok(w, "repeated send " + str(static_cast<std::size_t>(i)));
        }
        else
        {
            auto w = comm.recv(view(buf), size, tbccl::DataType::UInt8, sender, ctx(s));
            wait_ok(w, "repeated recv " + str(static_cast<std::size_t>(i)));
            check(bad_bytes(buf, seed) == 0, "repeated recv " + str(static_cast<std::size_t>(i)) + " has wrong bytes");
        }
    }
    tbccl::CudaStagingStats st;
    check(tbccl::cuda_staging_stats(comm, st), "staging stats available");
    const std::size_t bound = chunk != 0 ? std::min(chunk, max_size) : max_size;
    check(st.pinned_peak_capacity <= bound, "pinned peak capacity " + str(st.pinned_peak_capacity) + " exceeds " + str(bound));
    std::printf("ok repeated rank=%zu pinned_alloc=%llu pinned_free=%llu pinned_peak=%llu scratch_alloc=%llu chunk_env=%zu\n", rank, static_cast<unsigned long long>(st.pinned_alloc_count),
                static_cast<unsigned long long>(st.pinned_free_count), static_cast<unsigned long long>(st.pinned_peak_capacity), static_cast<unsigned long long>(st.device_scratch_alloc_count), chunk);
}

// all_reduce / broadcast / all_gather on device memory, produced by kernels and consumed without host synchronisation.
void scenario_collectives(tbccl::Communicator &comm, std::size_t rank)
{
    for (std::size_t size : {std::size_t{4}, 256 * KiB, (3 * MiB + 12348), 9 * MiB})
    {
        comm.barrier().wait();
        {   // all_reduce float32 SUM, in place
            Stream s;
            Dev buf(size);
            k_spin<<<1, 1, 0, s.s>>>(15 * kCyclesPerMs);
            k_fill_f32<<<256, 256, 0, s.s>>>(buf.f32(), size / 4, static_cast<float>(rank + 1));
            auto w = comm.all_reduce(view(buf), view(buf), size / 4, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, ctx(s));
            wait_ok(w, "all_reduce " + str(size));
            check(bad_f32(buf, 3.0f) == 0, "all_reduce " + str(size) + ": wrong sum");
        }
        for (std::size_t root = 0; root < 2; ++root)
        {
            Stream s;
            Dev buf(size);
            if (rank == root) fill(buf, 120 + static_cast<std::uint32_t>(root), s, 15 * kCyclesPerMs);
            else CU(cudaMemsetAsync(buf.p, 0, size, s.s));
            auto w = comm.broadcast(view(buf), root, ctx(s));
            wait_ok(w, "broadcast " + str(size) + " root " + str(root));
            check(bad_bytes(buf, 120 + static_cast<std::uint32_t>(root)) == 0, "broadcast " + str(size) + " root " + str(root) + ": wrong bytes");
        }
        {   // all_gather
            Stream s;
            Dev in(size), o0(size), o1(size);
            fill(in, 140 + static_cast<std::uint32_t>(rank), s, 15 * kCyclesPerMs);
            auto w = comm.all_gather(view(in), {view(o0), view(o1)}, ctx(s));
            wait_ok(w, "all_gather " + str(size));
            check(bad_bytes(o0, 140) == 0 && bad_bytes(o1, 141) == 0, "all_gather " + str(size) + ": wrong bytes");
        }
    }
    std::printf("ok collectives\n");
}

// A zero-length device transfer: whatever the outcome, it must be the same for every chunk setting and must not wedge the communicator.
void scenario_zero(tbccl::Communicator &comm, std::size_t rank)
{
    comm.barrier().wait();
    Dev buf(1);
    std::string outcome = "completed";
    try
    {
        if (rank == 0)
        {
            tbccl::BufferView v{tbccl::MemoryKind::Cuda, buf.p, 0, 0};
            auto w = comm.send(v, 0, tbccl::DataType::UInt8, 1);
            w.wait();
            if (w.has_error()) outcome = "error: " + w.error();
        }
        else
        {
            tbccl::BufferView v{tbccl::MemoryKind::Cuda, buf.p, 0, 0};
            auto w = comm.recv(v, 0, tbccl::DataType::UInt8, 0);
            w.wait();
            if (w.has_error()) outcome = "error: " + w.error();
        }
    }
    catch (const std::exception &e)
    {
        outcome = std::string("exception: ") + e.what();
    }
    std::printf("info zero-length rank=%zu outcome=%s\n", rank, outcome.c_str());
    // the communicator must still work afterwards if the zero-length call was only rejected at admission
    if (!comm.failed())
    {
        Stream s;
        Dev b(4097);
        if (rank == 0)
        {
            fill(b, 5, s);
            auto w = comm.send(view(b), 4097, tbccl::DataType::UInt8, 1, ctx(s));
            wait_ok(w, "send after zero-length");
        }
        else
        {
            auto w = comm.recv(view(b), 4097, tbccl::DataType::UInt8, 0, ctx(s));
            wait_ok(w, "recv after zero-length");
            check(bad_bytes(b, 5) == 0, "bytes after zero-length");
        }
    }
    else
    {
        std::printf("info zero-length poisoned the communicator\n");
    }
}

// Abort while a chunked device send has no receiver. Last scenario: the communicator is dead afterwards.
void scenario_abort(tbccl::Communicator &comm, std::size_t rank)
{
    comm.barrier().wait();
    if (rank == 0)
    {
        const std::size_t size = 64 * MiB;
        Stream s;
        Dev buf(size);
        fill(buf, 7, s);
        CU(cudaStreamSynchronize(s.s));
        auto w = comm.send(view(buf), size, tbccl::DataType::UInt8, 1, ctx(s));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const auto t0 = Clock::now();
        comm.abort("test abort during a staged device send");
        const bool terminal = w.wait_for(std::chrono::seconds(20));
        check(terminal, "staged device send still pending 20 s after abort");
        check(terminal && w.has_error(), "the aborted staged send must report an error");
        std::printf("ok abort rank=0 terminal_after_us=%.0f error=%s\n", us_since(t0), w.error().c_str());
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        const auto t0 = Clock::now();
        while (!comm.failed() && us_since(t0) < 10e6) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(comm.failed(), "the surviving rank did not notice the abort within 10 s");
        std::printf("ok abort rank=1 noticed_after_us=%.0f\n", us_since(t0));
    }
    CU(cudaDeviceSynchronize());
}

} // namespace

int main(int argc, char **argv)
{
    long rank = -1, port = 0;
    std::string scenarios = "zero,producer,delayed,outstanding,war,repeated,collectives,abort";
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--rank") rank = std::atol(argv[i + 1]);
        else if (a == "--port") port = std::atol(argv[i + 1]);
        else if (a == "--scenarios") scenarios = argv[i + 1];
    }
    if ((rank != 0 && rank != 1) || port <= 0)
    {
        std::fprintf(stderr, "usage: communicator_cuda_stream_chunk_test --rank 0|1 --port P [--scenarios list]\n");
        return 2;
    }
    const char *env = std::getenv("TBCCL_STAGED_CHUNK_BYTES");
    std::printf("rank %ld TBCCL_STAGED_CHUNK_BYTES=%s\n", rank, env ? (*env ? env : "(empty)") : "(unset)");
    try
    {
        tbccl::register_cuda_support();
        CU(cudaSetDevice(0));
        CU(cudaFree(nullptr));
        tbccl::CommunicatorOptions opts;
        opts.rank = static_cast<std::size_t>(rank);
        opts.peers = {{"127.0.0.1", static_cast<std::uint16_t>(port)}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        auto comm = tbccl::Communicator::create(opts);
        const std::size_t r = static_cast<std::size_t>(rank);
        std::stringstream list(scenarios);
        std::string name;
        while (std::getline(list, name, ','))
        {
            const int before = g_failures;
            try
            {
                if (name == "zero") scenario_zero(*comm, r);
                else if (name == "producer") scenario_producer(*comm, r, false);
                else if (name == "delayed") scenario_producer(*comm, r, true);
                else if (name == "outstanding") scenario_outstanding(*comm, r);
                else if (name == "war") scenario_war(*comm, r);
                else if (name == "repeated") scenario_repeated(*comm, r);
                else if (name == "collectives") scenario_collectives(*comm, r);
                else if (name == "abort") scenario_abort(*comm, r);
                else throw std::runtime_error("unknown scenario " + name);
            }
            catch (const std::exception &e)
            {
                ++g_failures;
                std::printf("FAIL scenario %s threw: %s\n", name.c_str(), e.what());
            }
            std::printf("scenario %s %s\n", name.c_str(), g_failures == before ? "passed" : "FAILED");
            std::fflush(stdout);
            if (g_failures != before && name != "abort") break; // a corrupted or poisoned run: stop rather than pile up follow-on failures
        }
        comm.reset();
        CU(cudaDeviceSynchronize());
    }
    catch (const std::exception &e)
    {
        std::printf("FAIL setup/teardown: %s\n", e.what());
        ++g_failures;
    }
    std::printf("rank %ld: %d checks, %d failures\n", rank, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
