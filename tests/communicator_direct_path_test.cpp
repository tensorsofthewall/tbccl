// Direct-access memory ignores TBCCL_STAGED_CHUNK_BYTES. MemoryKind::MetalShared is re-registered with a provider whose backend IS directly transport-accessible (like Host and
// the real MetalShared) and counts every stage/commit call: a transfer that was (wrongly) routed through the staging pool because of the chunk setting would stage or commit
// chunks, a direct one never does (all_gather is only checked on Host memory: it copies the local slot through the stage/commit calls of any provider). Run by ctest unset and with a chunk size smaller than the payloads. Built-in Host memory is exercised too (contents only).

#include "mesh_test_support.hpp"

#include <tbccl/communicator.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using namespace mesh_test;

namespace
{
std::atomic<std::uint64_t> g_stage_calls{0}, g_commit_calls{0}, g_direct_source{0}, g_direct_dest{0};

class DirectCountingBackend final : public tbccl::AsyncMemoryBackend
{
public:
    DirectCountingBackend(void *data, std::size_t bytes) : data_(static_cast<std::uint8_t *>(data)), bytes_(bytes) {}
    void stage_source_chunk(const tbccl::Chunk &, void *) override { ++g_stage_calls; }
    void commit_destination_chunk(const tbccl::Chunk &, const void *) override { ++g_commit_calls; }
    bool supports_direct_transport_access() const noexcept override { return true; }
    const void *direct_source_data() const noexcept override { ++g_direct_source; return data_; }
    void *direct_destination_data() noexcept override { ++g_direct_dest; return data_; }
    std::uint8_t *data() { return data_; }

private:
    std::uint8_t *data_;
    std::size_t bytes_;
};

std::atomic<std::uint64_t> g_primary_stage_calls{0}, g_primary_commit_calls{0};

// a staged (not directly accessible) backend, for the mixed provider below
class StagedCountingBackend final : public tbccl::AsyncMemoryBackend
{
public:
    StagedCountingBackend(void *data, std::size_t bytes) : data_(static_cast<std::uint8_t *>(data)), bytes_(bytes) {}
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        ++g_primary_stage_calls;
        std::memcpy(staging, data_ + chunk.offset, chunk.size);
    }
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        ++g_primary_commit_calls;
        std::memcpy(data_ + chunk.offset, staging, chunk.size);
    }
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

class DirectProvider final : public tbccl::ExternalMemoryProvider
{
public:
    explicit DirectProvider(const tbccl::BufferView &b) : primary_(b.data, b.bytes), bytes_(b.bytes) {}
    tbccl::AsyncMemoryBackend &primary_backend() override { return primary_; }
    tbccl::AsyncMemoryBackend &scratch_backend() override
    {
        if (!scratch_)
        {
            storage_.assign(bytes_, 0);
            scratch_ = std::make_unique<DirectCountingBackend>(storage_.data(), bytes_);
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
    DirectCountingBackend primary_;
    std::size_t bytes_;
    std::vector<std::uint8_t> storage_;
    std::unique_ptr<DirectCountingBackend> scratch_;
    std::unique_ptr<FloatSum> reduce_;
};

// Staged primary memory but a directly accessible scratch (the reverse of CUDA, whose two backends are both staged): the all_reduce root receives into the scratch, which must
// stay on the direct path even though the primary leg is chunked.
class MixedProvider final : public tbccl::ExternalMemoryProvider
{
public:
    explicit MixedProvider(const tbccl::BufferView &b) : primary_(b.data, b.bytes), bytes_(b.bytes) {}
    tbccl::AsyncMemoryBackend &primary_backend() override { return primary_; }
    tbccl::AsyncMemoryBackend &scratch_backend() override
    {
        if (!scratch_)
        {
            storage_.assign(bytes_, 0);
            scratch_ = std::make_unique<DirectCountingBackend>(storage_.data(), bytes_);
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
    StagedCountingBackend primary_;
    std::size_t bytes_;
    std::vector<std::uint8_t> storage_;
    std::unique_ptr<DirectCountingBackend> scratch_;
    std::unique_ptr<FloatSum> reduce_;
};

tbccl::BufferView view(tbccl::MemoryKind kind, std::vector<std::uint8_t> &v) { return {kind, v.data(), v.size(), -1}; }
std::uint8_t pat(std::size_t rank, std::size_t j) { return static_cast<std::uint8_t>(rank * 53 + j * 11 + (j >> 10) * 5 + 3); }

void exercise_mixed(tbccl::Communicator &comm, std::size_t rank)
{
    const std::size_t bytes = (std::size_t{3} << 20) + 12344, elems = bytes / 4;
    comm.barrier().wait();
    if (rank == 0)
    {
        g_stage_calls = 0; g_commit_calls = 0; g_primary_stage_calls = 0; g_primary_commit_calls = 0;
    }
    comm.barrier().wait();
    std::vector<std::uint8_t> raw(bytes);
    float *f = reinterpret_cast<float *>(raw.data());
    for (std::size_t i = 0; i < elems; ++i) f[i] = static_cast<float>((rank + 1) * ((i % 97) + 1));
    comm.all_reduce(view(tbccl::MemoryKind::MetalShared, raw), view(tbccl::MemoryKind::MetalShared, raw), elems, tbccl::DataType::Float32, tbccl::ReduceOp::Sum).wait();
    for (std::size_t i = 0; i < elems; ++i) expect(f[i] == static_cast<float>(3 * ((i % 97) + 1)), "mixed all_reduce element " + std::to_string(i));
    comm.barrier().wait();
    if (rank == 0)
        expect(g_stage_calls == 0 && g_commit_calls == 0,
               "the directly accessible scratch of the all_reduce root was staged: " + std::to_string(g_stage_calls.load()) + "/" + std::to_string(g_commit_calls.load()));
}

void exercise(tbccl::Communicator &comm, std::size_t rank, tbccl::MemoryKind kind, bool count_direct)
{
    const std::size_t bytes = (std::size_t{3} << 20) + 12344, elems = bytes / 4;
    comm.barrier().wait();
    if (rank == 0)
    {
        g_stage_calls = 0; g_commit_calls = 0; g_direct_source = 0; g_direct_dest = 0;
    }
    comm.barrier().wait();
    {   // point to point, both directions
        std::vector<std::uint8_t> buf(bytes);
        if (rank == 0)
        {
            for (std::size_t j = 0; j < bytes; ++j) buf[j] = pat(0, j);
            comm.send(view(kind, buf), bytes, tbccl::DataType::UInt8, 1).wait();
            comm.recv(view(kind, buf), bytes, tbccl::DataType::UInt8, 1).wait();
            for (std::size_t j = 0; j < bytes; ++j) expect(buf[j] == pat(1, j), "p2p reverse byte " + std::to_string(j));
        }
        else
        {
            comm.recv(view(kind, buf), bytes, tbccl::DataType::UInt8, 0).wait();
            for (std::size_t j = 0; j < bytes; ++j) expect(buf[j] == pat(0, j), "p2p byte " + std::to_string(j));
            for (std::size_t j = 0; j < bytes; ++j) buf[j] = pat(1, j);
            comm.send(view(kind, buf), bytes, tbccl::DataType::UInt8, 0).wait();
        }
    }
    {   // all_reduce float32 SUM
        std::vector<std::uint8_t> raw(bytes);
        float *f = reinterpret_cast<float *>(raw.data());
        for (std::size_t i = 0; i < elems; ++i) f[i] = static_cast<float>((rank + 1) * ((i % 97) + 1));
        comm.all_reduce(view(kind, raw), view(kind, raw), elems, tbccl::DataType::Float32, tbccl::ReduceOp::Sum).wait();
        for (std::size_t i = 0; i < elems; ++i) expect(f[i] == static_cast<float>(3 * ((i % 97) + 1)), "all_reduce element " + std::to_string(i));
    }
    {   // broadcast from rank 1 and all_gather
        std::vector<std::uint8_t> buf(bytes, 0);
        if (rank == 1) for (std::size_t j = 0; j < bytes; ++j) buf[j] = pat(1, j);
        comm.broadcast(view(kind, buf), 1).wait();
        for (std::size_t j = 0; j < bytes; ++j) expect(buf[j] == pat(1, j), "broadcast byte " + std::to_string(j));
        if (!count_direct)   // all_gather copies its own slot through the providers' stage/commit calls, which the counting backend would report; Host memory covers its contents
        {
            std::vector<std::uint8_t> in(bytes), g0(bytes, 0), g1(bytes, 0);
            for (std::size_t j = 0; j < bytes; ++j) in[j] = pat(rank, j);
            comm.all_gather(view(kind, in), {view(kind, g0), view(kind, g1)}).wait();
            for (std::size_t j = 0; j < bytes; ++j) expect(g0[j] == pat(0, j) && g1[j] == pat(1, j), "all_gather byte " + std::to_string(j));
        }
    }
    comm.barrier().wait();
    if (rank == 0 && count_direct)
    {
        expect(g_stage_calls == 0 && g_commit_calls == 0,
               "direct-access memory was staged: stage calls " + std::to_string(g_stage_calls.load()) + ", commit calls " + std::to_string(g_commit_calls.load()));
        expect(g_direct_source > 0 && g_direct_dest > 0, "the direct path was not used");
    }
}
} // namespace

int main()
{
    try
    {
        const char *e = std::getenv("TBCCL_STAGED_CHUNK_BYTES");
        std::printf("TBCCL_STAGED_CHUNK_BYTES=%s\n", e ? e : "(unset)");
        {
            World w(2);   // built-in Host memory
            w.run([](std::size_t rank, tbccl::Communicator &comm) { exercise(comm, rank, tbccl::MemoryKind::Host, false); });
            std::printf("host ok\n");
        }
        tbccl::register_memory_provider_factory(tbccl::MemoryKind::MetalShared, [](const tbccl::BufferView &b, const tbccl::ExecutionContext &) -> std::unique_ptr<tbccl::ExternalMemoryProvider> {
            return std::make_unique<DirectProvider>(b);
        });
        {
            World w(2);   // a directly accessible provider that counts stage/commit calls
            w.run([](std::size_t rank, tbccl::Communicator &comm) { exercise(comm, rank, tbccl::MemoryKind::MetalShared, true); });
            std::printf("direct provider ok: no stage/commit calls\n");
        }
        tbccl::register_memory_provider_factory(tbccl::MemoryKind::MetalShared, [](const tbccl::BufferView &b, const tbccl::ExecutionContext &) -> std::unique_ptr<tbccl::ExternalMemoryProvider> {
            return std::make_unique<MixedProvider>(b);
        });
        {
            World w(2);   // staged primary, direct scratch
            w.run([](std::size_t rank, tbccl::Communicator &comm) { exercise_mixed(comm, rank); });
            std::printf("mixed provider ok: the direct scratch leg was not staged\n");
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("communicator_direct_path_test passed\n");
    return 0;
}
