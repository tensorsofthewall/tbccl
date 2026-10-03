// A heterogeneous N-rank world on one machine: exactly ONE rank owns CUDA buffers (the single GPU is never shared between ranks), the others use Host
// memory. world_size 3 and 4, the CUDA rank at rank 0 (the reference root) and at the last rank (a non-root contributor). Float32 / Int32 / Int8
// all_reduce, broadcast from a CUDA and from a Host root, all_gather with device input and outputs. The producer work is ordered on a user stream with
// no host synchronization before submission, as in the N=2 CUDA tests.

#include "mesh_test_support.hpp"

#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace mesh_test;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

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
        explicit Dev(std::size_t bytes) : n(bytes) { cu(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
        ~Dev() { cudaFree(p); }
        void upload(const void *h, cudaStream_t s) { cu(cudaMemcpyAsync(p, h, n, cudaMemcpyHostToDevice, s), "H2D"); }
        void download(void *h) const
        {
            cudaStream_t other;
            cu(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "stream");
            cu(cudaMemcpyAsync(h, p, n, cudaMemcpyDeviceToHost, other), "D2H");
            cu(cudaStreamSynchronize(other), "sync");
            cudaStreamDestroy(other);
        }
    };

    tbccl::BufferView dview(const Dev &d) { return {MemoryKind::Cuda, d.p, d.n, 0}; }
    tbccl::BufferView hview(void *p, std::size_t n) { return {MemoryKind::Host, p, n, 0}; }

    void ok(tbccl::Work w, const std::string &what)
    {
        w.wait();
        expect(!w.has_error(), what + ": " + w.error());
    }

    template <typename T> void all_reduce_case(std::size_t world, std::size_t cuda_rank, DataType dt, std::size_t count)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            cudaStream_t s;
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            const tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
            std::vector<T> mine(count);
            for (std::size_t i = 0; i < count; ++i) mine[i] = static_cast<T>((rank + 1) * 3 + i % 7);
            std::vector<T> want(count, 0);
            for (std::size_t r = 0; r < world; ++r)
                for (std::size_t i = 0; i < count; ++i) want[i] = static_cast<T>(want[i] + static_cast<T>((r + 1) * 3 + i % 7));
            std::vector<T> got(count);
            const std::size_t bytes = count * sizeof(T);
            if (rank == cuda_rank)
            {
                Dev d(bytes);
                d.upload(mine.data(), s);
                ok(comm.all_reduce(dview(d), dview(d), count, dt, ReduceOp::Sum, ctx), "cuda all_reduce");
                d.download(got.data());
            }
            else
            {
                got = mine;
                ok(comm.all_reduce(hview(got.data(), bytes), hview(got.data(), bytes), count, dt, ReduceOp::Sum), "host all_reduce");
            }
            expect(got == want, std::string("all_reduce ") + tbccl::datatype_label(dt) + " result on rank " + std::to_string(rank));
            cudaStreamDestroy(s);
        });
    }

    void test_all_reduce()
    {
        for (std::size_t world : {std::size_t{3}, std::size_t{4}})
            for (std::size_t cuda_rank : {std::size_t{0}, world - 1})
                for (std::size_t count : {std::size_t{1}, std::size_t{4097}, std::size_t{1} << 20})
                {
                    all_reduce_case<float>(world, cuda_rank, DataType::Float32, count);
                    all_reduce_case<std::int32_t>(world, cuda_rank, DataType::Int32, count);
                    all_reduce_case<std::int8_t>(world, cuda_rank, DataType::Int8, count);
                }
        std::cout << "[PASS] all_reduce Float32/Int32/Int8 with one CUDA rank (root and non-root), world_size 3 and 4\n";
    }

    void test_broadcast_and_all_gather()
    {
        for (std::size_t world : {std::size_t{3}, std::size_t{4}})
        {
            for (std::size_t cuda_rank : {std::size_t{0}, world - 1})
            {
                for (std::size_t root = 0; root < world; ++root)
                {
                    const std::size_t bytes = (std::size_t{1} << 20) + 5;
                    std::vector<std::uint8_t> src(bytes);
                    for (std::size_t i = 0; i < bytes; ++i) src[i] = static_cast<std::uint8_t>(i * 131 + root);
                    run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                        cudaStream_t s;
                        cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
                        const tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
                        const auto init = rank == root ? src : std::vector<std::uint8_t>(bytes, 0xEE);
                        std::vector<std::uint8_t> got(bytes);
                        if (rank == cuda_rank)
                        {
                            Dev d(bytes);
                            d.upload(init.data(), s);
                            ok(comm.broadcast(dview(d), root, ctx), "cuda broadcast");
                            d.download(got.data());
                        }
                        else
                        {
                            got = init;
                            ok(comm.broadcast(hview(got.data(), bytes), root), "host broadcast");
                        }
                        expect(got == src, "broadcast root " + std::to_string(root) + " on rank " + std::to_string(rank));
                        cudaStreamDestroy(s);
                    });
                }
                const std::size_t bytes = 100000;
                run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                    cudaStream_t s;
                    cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
                    const tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, s};
                    auto piece = [&](std::size_t r) {
                        std::vector<std::uint8_t> v(bytes);
                        for (std::size_t i = 0; i < bytes; ++i) v[i] = static_cast<std::uint8_t>(i * 7 + r * 61);
                        return v;
                    };
                    const auto mine = piece(rank);
                    std::vector<std::vector<std::uint8_t>> got(world, std::vector<std::uint8_t>(bytes));
                    if (rank == cuda_rank)
                    {
                        Dev in(bytes);
                        in.upload(mine.data(), s);
                        std::vector<std::unique_ptr<Dev>> outs;
                        std::vector<tbccl::BufferView> views;
                        for (std::size_t r = 0; r < world; ++r)
                        {
                            outs.push_back(std::make_unique<Dev>(bytes));
                            views.push_back(dview(*outs.back()));
                        }
                        ok(comm.all_gather(dview(in), views, ctx), "cuda all_gather");
                        for (std::size_t r = 0; r < world; ++r) outs[r]->download(got[r].data());
                    }
                    else
                    {
                        std::vector<tbccl::BufferView> views;
                        for (std::size_t r = 0; r < world; ++r) views.push_back(hview(got[r].data(), bytes));
                        auto in = mine;
                        ok(comm.all_gather(hview(in.data(), bytes), views), "host all_gather");
                    }
                    for (std::size_t r = 0; r < world; ++r) expect(got[r] == piece(r), "all_gather slot " + std::to_string(r) + " on rank " + std::to_string(rank));
                    cudaStreamDestroy(s);
                });
            }
        }
        std::cout << "[PASS] broadcast (every root) and all_gather with one CUDA rank, world_size 3 and 4\n";
    }

} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(240));
        std::cerr << "[FAIL] watchdog: a CUDA N-rank test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        tbccl::register_cuda_support();
        test_all_reduce();
        test_broadcast_and_all_gather();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All CUDA N-rank tests passed.\n";
    return 0;
}
