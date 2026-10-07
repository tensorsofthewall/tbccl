// real-hardware CUDA coverage for Communicator::broadcast()/all_gather(): CUDA<->Host and
// CUDA<->CUDA (same GPU), both roots, producer readiness through a user stream (no host sync before
// submission) and an independent consumer stream after wait() (no device-wide synchronization).

#include <tbccl/communicator.hpp>
#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

void expect(bool c, const std::string &m)
{
    if (!c) throw std::runtime_error("assertion failed: " + m);
}
void cu(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
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

std::vector<std::uint8_t> pattern(std::size_t n, unsigned seed)
{
    std::vector<std::uint8_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>((i * 131u + seed * 17u + 3u) & 0xffu);
    return v;
}

// Device buffer on `stream`-ordered init (no host synchronization before submission).
struct DevBuf
{
    void *p = nullptr;
    std::size_t n;
    explicit DevBuf(std::size_t bytes) : n(bytes) { cu(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~DevBuf() { cudaFree(p); }
    void upload_async(const std::vector<std::uint8_t> &h, cudaStream_t s) { cu(cudaMemcpyAsync(p, h.data(), n, cudaMemcpyHostToDevice, s), "H2D"); }
    // Independent consumer: a fresh stream, only that stream is synchronized.
    std::vector<std::uint8_t> download_on_other_stream()
    {
        cudaStream_t other;
        cu(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "stream create");
        std::vector<std::uint8_t> h(n);
        cu(cudaMemcpyAsync(h.data(), p, n, cudaMemcpyDeviceToHost, other), "D2H");
        cu(cudaStreamSynchronize(other), "stream sync");
        cudaStreamDestroy(other);
        return h;
    }
};

void ok(tbccl::Work w, const std::string &what)
{
    w.wait();
    expect(!w.has_error(), what + ": " + w.error());
}

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

// The control port, the next one, and the data ports (+1000) must all be free.
std::uint16_t next_port()
{
    static std::uint16_t p = 29650;
    for (;;)
    {
        p = static_cast<std::uint16_t>(p + 4);
        if (port_free(p) && port_free(p + 1) && port_free(p + 1000) && port_free(p + 1001)) return p;
    }
}

enum class Kind { Host, Cuda };

// rank r holds a buffer of `kind[r]`; broadcast from `root`.
void broadcast_case(Kind k0, Kind k1, std::size_t root, std::size_t bytes)
{
    const Kind kinds[2] = {k0, k1};
    std::cerr << "broadcast k0=" << int(k0) << " k1=" << int(k1) << " root=" << root << " bytes=" << bytes << std::endl;
    auto src = pattern(bytes, 9);
    auto rank_fn = [&](std::size_t rank) {
        return [&, rank](tbccl::Communicator &c) {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
            const auto init = rank == root ? src : std::vector<std::uint8_t>(bytes, 0xEE);
            if (kinds[rank] == Kind::Cuda)
            {
                DevBuf d(bytes);
                d.upload_async(init, s); // producer on `s`; no sync before broadcast
                ok(c.broadcast({tbccl::MemoryKind::Cuda, d.p, bytes, 0}, root, ctx), "cuda broadcast");
                expect(d.download_on_other_stream() == src, "cuda rank result, rank=" + std::to_string(rank));
            }
            else
            {
                auto h = init;
                ok(c.broadcast({tbccl::MemoryKind::Host, h.data(), bytes, 0}, root), "host broadcast");
                expect(h == src, "host rank result, rank=" + std::to_string(rank));
            }
            cudaStreamDestroy(s);
        };
    };
    run_pair(next_port(), rank_fn(0), rank_fn(1));
}

void all_gather_case(Kind k0, Kind k1, std::size_t bytes)
{
    const Kind kinds[2] = {k0, k1};
    std::cerr << "all_gather k0=" << int(k0) << " k1=" << int(k1) << " bytes=" << bytes << std::endl;
    const std::vector<std::uint8_t> in[2] = {pattern(bytes, 21), pattern(bytes, 42)};
    auto rank_fn = [&](std::size_t rank) {
        return [&, rank](tbccl::Communicator &c) {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
            if (kinds[rank] == Kind::Cuda)
            {
                DevBuf din(bytes), o0(bytes), o1(bytes);
                din.upload_async(in[rank], s);
                std::vector<tbccl::BufferView> outs = {
                    {tbccl::MemoryKind::Cuda, o0.p, bytes, 0}, {tbccl::MemoryKind::Cuda, o1.p, bytes, 0}};
                ok(c.all_gather({tbccl::MemoryKind::Cuda, din.p, bytes, 0}, outs, ctx), "cuda all_gather");
                expect(o0.download_on_other_stream() == in[0], "cuda outputs[0]");
                expect(o1.download_on_other_stream() == in[1], "cuda outputs[1]");
            }
            else
            {
                auto h = in[rank];
                std::vector<std::uint8_t> o0(bytes, 0xEE), o1(bytes, 0xEE);
                std::vector<tbccl::BufferView> outs = {
                    {tbccl::MemoryKind::Host, o0.data(), bytes, 0}, {tbccl::MemoryKind::Host, o1.data(), bytes, 0}};
                ok(c.all_gather({tbccl::MemoryKind::Host, h.data(), bytes, 0}, outs), "host all_gather");
                expect(o0 == in[0] && o1 == in[1], "host outputs");
            }
            cudaStreamDestroy(s);
        };
    };
    run_pair(next_port(), rank_fn(0), rank_fn(1));
}

} // namespace

int main(int argc, char **argv)
{
    tbccl::register_cuda_support();
    try
    {
        if (argc == 4 && std::string(argv[1]) == "ag")
        {
            all_gather_case(argv[2][0] == 'c' ? Kind::Cuda : Kind::Host, argv[2][1] == 'c' ? Kind::Cuda : Kind::Host, std::stoul(argv[3]));
            std::cout << "ok\n";
            return 0;
        }
        for (std::size_t bytes : {std::size_t{8}, std::size_t{4096}, std::size_t{1048583}})
        {
            for (std::size_t root : {std::size_t{0}, std::size_t{1}})
            {
                broadcast_case(Kind::Cuda, Kind::Host, root, bytes);
                broadcast_case(Kind::Host, Kind::Cuda, root, bytes);
                broadcast_case(Kind::Cuda, Kind::Cuda, root, bytes);
            }
            all_gather_case(Kind::Cuda, Kind::Host, bytes);
            all_gather_case(Kind::Host, Kind::Cuda, bytes);
            all_gather_case(Kind::Cuda, Kind::Cuda, bytes);
            std::cout << "[PASS] bytes=" << bytes << "\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All communicator CUDA collective tests passed.\n";
    return 0;
}
