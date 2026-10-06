// Mixed P2P + collective traffic on one communicator with rank 0 on CUDA buffers and rank 1 on Host (a heterogeneous W2 world on one machine, one GPU). all_reduce
// / broadcast / all_gather overlapped with a CUDA->Host send and a Host->CUDA receive, 4 KiB / 1 MiB / 16 MiB, in the same relative order on both ranks and in
// each opposite order. Producer readiness goes through a user stream (no host synchronization before submission); consumers read on an independent stream. This
// also exercises the CUDA provider's staging resources with a P2P and a collective transfer active at the same time (two workers per peer).

#include "mesh_test_support.hpp"

#include <tbccl/cuda_support.hpp>

#include "tensor/cuda_external_async_backend.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

using namespace mesh_test;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;
using Bytes = std::vector<std::uint8_t>;

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
    explicit Dev(std::size_t bytes) : n(bytes) { cu(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    void upload(const void *h, cudaStream_t s) { cu(cudaMemcpyAsync(p, h, n, cudaMemcpyHostToDevice, s), "H2D"); }
    Bytes download() const
    {
        cudaStream_t other;
        cu(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "stream");
        Bytes h(n);
        cu(cudaMemcpyAsync(h.data(), p, n, cudaMemcpyDeviceToHost, other), "D2H");
        cu(cudaStreamSynchronize(other), "sync");
        cudaStreamDestroy(other);
        return h;
    }
};

Bytes pattern(std::size_t from, std::size_t tag, std::size_t n)
{
    Bytes v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(from * 131 + tag * 17 + i * 7 + (i >> 8) + 1);
    return v;
}

tbccl::BufferView dview(const Dev &d) { return {MemoryKind::Cuda, d.p, d.n, 0}; }
tbccl::BufferView hview(Bytes &b) { return {MemoryKind::Host, b.data(), b.size(), 0}; }
tbccl::BufferView hraw(void *p, std::size_t n) { return {MemoryKind::Host, p, n, 0}; }

void wait_ok(std::vector<tbccl::Work> &works, const std::string &what)
{
    for (std::size_t i = 0; i < works.size(); ++i)
    {
        expect(works[i].wait_for(std::chrono::seconds(60)), what + ": Work #" + std::to_string(i) + " did not finish");
        expect(!works[i].has_error(), what + ": Work #" + std::to_string(i) + " failed: " + works[i].error());
    }
}

// variant 0: both ranks collective first; 1: CUDA rank collective first, Host rank P2P first; 2: the reverse of 1; 3: both P2P first;
// 4: the CUDA rank posts its P2P first and its collective second, while the Host rank finishes the collective BEFORE it posts any P2P (the CUDA rank's receive is pending,
//    holding whatever staging it holds, for the whole collective: independent domains must not let it block the collective)
void run_case(const std::string &family, std::size_t bytes, int variant)
{
    run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
        const bool cuda_rank = rank == 0;
        const bool coll_first = variant == 0 || (variant == 1 && cuda_rank) || (variant == 2 && !cuda_rank);
        cudaStream_t s = nullptr;
        if (cuda_rank) cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        const tbccl::ExecutionContext ctx = cuda_rank ? tbccl::ExecutionContext{tbccl::ExecutionContextKind::CudaStream, s} : tbccl::ExecutionContext{};

        std::vector<std::int32_t> ar_host(bytes / 4);
        for (std::size_t i = 0; i < ar_host.size(); ++i) ar_host[i] = static_cast<std::int32_t>(i % 1000 + rank * 77);
        Bytes bc = rank == 0 ? pattern(0, 5, bytes) : Bytes(bytes, 0xEE);
        Bytes ag_in = pattern(rank, 6, bytes), ag0(bytes, 0xEE), ag1(bytes, 0xEE);
        Bytes tx = pattern(rank, 9, bytes), rx(bytes, 0xEE); // rank 0 (CUDA) and rank 1 (Host) swap `tx`

        std::unique_ptr<Dev> d_coll, d_ag_in, d_ag0, d_ag1, d_tx, d_rx;
        if (cuda_rank)
        {
            d_coll = std::make_unique<Dev>(bytes);
            if (family == "AR") d_coll->upload(ar_host.data(), s);
            else d_coll->upload(bc.data(), s);
            d_ag_in = std::make_unique<Dev>(bytes);
            d_ag_in->upload(ag_in.data(), s);
            d_ag0 = std::make_unique<Dev>(bytes);
            d_ag1 = std::make_unique<Dev>(bytes);
            d_tx = std::make_unique<Dev>(bytes);
            d_tx->upload(tx.data(), s);
            d_rx = std::make_unique<Dev>(bytes);
        }
        std::vector<tbccl::Work> works;
        auto post_c = [&] {
            if (family == "AR")
                works.push_back(cuda_rank ? comm.all_reduce(dview(*d_coll), dview(*d_coll), bytes / 4, DataType::Int32, ReduceOp::Sum, ctx)
                                          : comm.all_reduce(hraw(ar_host.data(), bytes), hraw(ar_host.data(), bytes), bytes / 4, DataType::Int32, ReduceOp::Sum));
            else if (family == "BC")
                works.push_back(cuda_rank ? comm.broadcast(dview(*d_coll), 0, ctx) : comm.broadcast(hview(bc), 0));
            else
                works.push_back(cuda_rank ? comm.all_gather(dview(*d_ag_in), {dview(*d_ag0), dview(*d_ag1)}, ctx) : comm.all_gather(hview(ag_in), {hview(ag0), hview(ag1)}));
        };
        auto post_p = [&] {
            if (cuda_rank)
            {
                works.push_back(comm.recv(dview(*d_rx), bytes, DataType::UInt8, 1, ctx));
                works.push_back(comm.send(dview(*d_tx), bytes, DataType::UInt8, 1, ctx));
            }
            else
            {
                works.push_back(comm.recv(hview(rx), bytes, DataType::UInt8, 0));
                works.push_back(comm.send(hview(tx), bytes, DataType::UInt8, 0));
            }
        };
        if (variant == 4 && !cuda_rank)
        {
            post_c();
            wait_ok(works, "host collective before any P2P");
            post_p();
        }
        else if (coll_first)
        {
            post_c();
            post_p();
        }
        else
        {
            post_p();
            post_c();
        }
        const std::string what = family + " " + std::to_string(bytes) + "B variant " + std::to_string(variant) + " rank " + std::to_string(rank);
        wait_ok(works, what);

        if (cuda_rank) rx = d_rx->download();
        expect(rx == pattern(1 - rank, 9, bytes), what + ": P2P payload");
        if (family == "AR")
        {
            const std::vector<std::int32_t> got = [&] {
                if (!cuda_rank) return ar_host;
                const Bytes b = d_coll->download();
                std::vector<std::int32_t> v(bytes / 4);
                std::memcpy(v.data(), b.data(), bytes);
                return v;
            }();
            for (std::size_t i = 0; i < got.size(); ++i)
                expect(got[i] == static_cast<std::int32_t>((i % 1000) * 2 + 77), what + ": all_reduce element " + std::to_string(i));
        }
        else if (family == "BC") expect((cuda_rank ? d_coll->download() : bc) == pattern(0, 5, bytes), what + ": broadcast");
        else
        {
            expect((cuda_rank ? d_ag0->download() : ag0) == pattern(0, 6, bytes), what + ": all_gather slot 0");
            expect((cuda_rank ? d_ag1->download() : ag1) == pattern(1, 6, bytes), what + ": all_gather slot 1");
        }
        if (cuda_rank) cudaStreamDestroy(s);
    });
}

} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::minutes(20));
        std::cerr << "mixed_domain_cuda_test: watchdog expired\n";
        std::_Exit(2);
    }).detach();
    try
    {
        tbccl::register_cuda_support();
        for (const char *family : {"AR", "BC", "AG"})
            for (const std::size_t bytes : {std::size_t{4096}, std::size_t{1} << 20, std::size_t{16} << 20})
                for (int variant = 0; variant < 5; ++variant)
                {
                    run_case(family, bytes, variant);
                    std::cout << "[PASS] " << family << " " << bytes << "B variant " << variant << "\n" << std::flush;
                }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "mixed_domain_cuda_test: ok\n";
    return 0;
}
