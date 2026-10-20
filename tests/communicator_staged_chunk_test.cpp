// TBCCL_STAGED_CHUNK_BYTES: a staged (non-direct) memory kind moves its payload in pipelined chunks instead of one whole-payload chunk. MemoryKind::MetalShared is
// re-registered here with a provider whose backend is NOT directly transport-accessible, so every transfer takes the staged path exactly as CUDA does, and counts its
// stage/commit calls. Run twice by ctest: without the variable (the staged transfer must remain ONE chunk, the old behaviour) and with it (the chunk count must be
// ceil(bytes / chunk), no chunk larger than the setting, and every result byte-identical).

#include "mesh_test_support.hpp"

#include <tbccl/communicator.hpp>
#include <tbccl/hetero_allreduce.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace mesh_test;

namespace
{
std::atomic<std::uint64_t> g_stage_calls{0}, g_commit_calls{0}, g_max_chunk{0};

void note_chunk(std::size_t n)
{
    std::uint64_t seen = g_max_chunk.load();
    while (seen < n && !g_max_chunk.compare_exchange_weak(seen, n)) {}
}

class StagedBackend final : public tbccl::AsyncMemoryBackend
{
public:
    StagedBackend(void *data, std::size_t bytes) : data_(static_cast<std::uint8_t *>(data)), bytes_(bytes) {}
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        ++g_stage_calls;
        note_chunk(chunk.size);
        std::memcpy(staging, data_ + chunk.offset, chunk.size);
    }
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        ++g_commit_calls;
        note_chunk(chunk.size);
        std::memcpy(data_ + chunk.offset, staging, chunk.size);
    }
    // supports_direct_transport_access() stays false: this is the staged path
    std::uint8_t *data() { return data_; }

private:
    std::uint8_t *data_;
    std::size_t bytes_;
};

class FloatSum final : public tbccl::LocalReduceBackend
{
public:
    FloatSum(std::uint8_t *local, const std::uint8_t *peer) : local_(local), peer_(peer) {}
    void reduce_sum(std::size_t count, tbccl::DataType) override
    {
        float *a = reinterpret_cast<float *>(local_);
        const float *b = reinterpret_cast<const float *>(peer_);
        for (std::size_t i = 0; i < count; ++i) a[i] += b[i];
    }

private:
    std::uint8_t *local_;
    const std::uint8_t *peer_;
};

class StagedProvider final : public tbccl::ExternalMemoryProvider
{
public:
    explicit StagedProvider(const tbccl::BufferView &b) : primary_(b.data, b.bytes), bytes_(b.bytes) {}
    tbccl::AsyncMemoryBackend &primary_backend() override { return primary_; }
    tbccl::AsyncMemoryBackend &scratch_backend() override
    {
        if (!scratch_)
        {
            storage_.assign(bytes_, 0);
            scratch_ = std::make_unique<StagedBackend>(storage_.data(), bytes_);
        }
        return *scratch_;
    }
    tbccl::LocalReduceBackend &reduce_backend() override
    {
        scratch_backend();
        reduce_ = std::make_unique<FloatSum>(primary_.data(), storage_.data());
        return *reduce_;
    }

private:
    StagedBackend primary_;
    std::size_t bytes_;
    std::vector<std::uint8_t> storage_;
    std::unique_ptr<StagedBackend> scratch_;
    std::unique_ptr<FloatSum> reduce_;
};

std::size_t configured_chunk()
{
    const char *e = std::getenv("TBCCL_STAGED_CHUNK_BYTES");
    return e && *e ? static_cast<std::size_t>(std::strtoull(e, nullptr, 10)) : 0;
}

tbccl::BufferView view(std::vector<std::uint8_t> &v) { return {tbccl::MemoryKind::MetalShared, v.data(), v.size(), -1}; }
std::uint8_t pat(std::size_t rank, std::size_t seq, std::size_t j) { return static_cast<std::uint8_t>(rank * 53 + seq * 17 + j * 11 + (j >> 10) * 5 + 3); }

std::uint64_t expected_chunks(std::size_t bytes)
{
    const std::size_t c = configured_chunk();
    return c == 0 || c >= bytes ? 1 : (bytes + c - 1) / c;
}

void reset_counters()
{
    g_stage_calls = 0;
    g_commit_calls = 0;
    g_max_chunk = 0;
}

// One send/recv of `bytes` rank 0 -> rank 1: the sender stages ceil(bytes/chunk) chunks, the receiver commits as many, contents identical.
void p2p_case(tbccl::Communicator &comm, std::size_t rank, std::size_t bytes, std::size_t seq)
{
    comm.barrier().wait();
    if (rank == 0) reset_counters();
    comm.barrier().wait();
    std::vector<std::uint8_t> buf(bytes);
    if (rank == 0)
    {
        for (std::size_t j = 0; j < bytes; ++j) buf[j] = pat(0, seq, j);
        comm.send(view(buf), bytes, tbccl::DataType::UInt8, 1).wait();
    }
    else
    {
        comm.recv(view(buf), bytes, tbccl::DataType::UInt8, 0).wait();
        for (std::size_t j = 0; j < bytes; ++j) expect(buf[j] == pat(0, seq, j), "p2p byte " + std::to_string(j) + " of " + std::to_string(bytes));
    }
    comm.barrier().wait();
    if (rank == 0)
    {
        expect(g_stage_calls == expected_chunks(bytes), "p2p " + std::to_string(bytes) + ": stage calls " + std::to_string(g_stage_calls.load()) + " expected " + std::to_string(expected_chunks(bytes)));
        expect(g_commit_calls == expected_chunks(bytes), "p2p " + std::to_string(bytes) + ": commit calls " + std::to_string(g_commit_calls.load()));
        if (configured_chunk() != 0) expect(g_max_chunk <= configured_chunk(), "a chunk larger than the configured size was staged");
    }
}

void collectives_case(tbccl::Communicator &comm, std::size_t rank)
{
    const std::size_t bytes = std::size_t{3} << 20, elems = bytes / 4;
    {   // all_reduce: float32 SUM of (rank + 1) * k
        std::vector<std::uint8_t> raw(bytes);
        float *f = reinterpret_cast<float *>(raw.data());
        for (std::size_t i = 0; i < elems; ++i) f[i] = static_cast<float>((rank + 1) * ((i % 97) + 1));
        comm.all_reduce(view(raw), view(raw), elems, tbccl::DataType::Float32, tbccl::ReduceOp::Sum).wait();
        for (std::size_t i = 0; i < elems; ++i) expect(f[i] == static_cast<float>(3 * ((i % 97) + 1)), "all_reduce element " + std::to_string(i));
    }
    for (std::size_t root = 0; root < 2; ++root)
    {   // broadcast from each root
        std::vector<std::uint8_t> buf(bytes, 0);
        if (rank == root) for (std::size_t j = 0; j < bytes; ++j) buf[j] = pat(root, 70 + root, j);
        comm.broadcast(view(buf), root).wait();
        for (std::size_t j = 0; j < bytes; ++j) expect(buf[j] == pat(root, 70 + root, j), "broadcast root " + std::to_string(root) + " byte " + std::to_string(j));
    }
    {   // all_gather
        std::vector<std::uint8_t> in(bytes), g0(bytes, 0), g1(bytes, 0);
        for (std::size_t j = 0; j < bytes; ++j) in[j] = pat(rank, 80, j);
        comm.all_gather(view(in), {view(g0), view(g1)}).wait();
        for (std::size_t j = 0; j < bytes; ++j)
            expect(g0[j] == pat(0, 80, j) && g1[j] == pat(1, 80, j), "all_gather byte " + std::to_string(j));
    }
}
} // namespace

int main()
{
    try
    {
        tbccl::register_memory_provider_factory(tbccl::MemoryKind::MetalShared, [](const tbccl::BufferView &b, const tbccl::ExecutionContext &) -> std::unique_ptr<tbccl::ExternalMemoryProvider> {
            return std::make_unique<StagedProvider>(b);
        });
        std::printf("TBCCL_STAGED_CHUNK_BYTES=%zu\n", configured_chunk());
        {
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                std::size_t seq = 0;
                for (std::size_t bytes : {std::size_t{1}, std::size_t{4096}, std::size_t{100000}, std::size_t{1} << 20, (std::size_t{3} << 20) + 12345, std::size_t{9} << 20})
                    p2p_case(comm, rank, bytes, seq++);
                collectives_case(comm, rank);
                comm.barrier().wait();
            });
            std::printf("contents and chunk counts ok\n");
        }
        {
            // a chunked staged send to a silent peer, then abort: the Work becomes terminal in bounded time and the communicator (with its staging thread) is destroyed
            World w(2);
            w.run([](std::size_t rank, tbccl::Communicator &comm) {
                if (rank == 0)
                {
                    const std::size_t bytes = std::size_t{64} << 20;
                    std::vector<std::uint8_t> buf(bytes, 7);
                    auto work = comm.send(view(buf), bytes, tbccl::DataType::UInt8, 1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    comm.abort("test abort during a staged send");
                    expect(work.wait_for(std::chrono::seconds(20)), "staged send still pending 20 s after abort");
                    expect(work.has_error(), "the aborted staged send must report an error");
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(600));
                }
            });
            std::printf("abort ok\n");
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("communicator_staged_chunk_test passed\n");
    return 0;
}
