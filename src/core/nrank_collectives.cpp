// Reference N-rank collectives (nrank_collectives.hpp).

#include "nrank_collectives.hpp"

#include "host_pointer_backend.hpp"

#include <tbccl/communicator.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <stdexcept>

namespace tbccl::detail
{

namespace
{

std::atomic<std::uint64_t> g_transfer_id{std::uint64_t{1} << 41};

std::string peer_text(std::size_t peer) { return "peer=" + std::to_string(peer); }

const char *direction_text(TransferDirection d) { return d == TransferDirection::Send ? "send" : "recv"; }

// One traced, posted transfer.
void post(
    const CollectiveRun &run, OpGroup &group, std::size_t peer, TransferDirection direction, AsyncMemoryBackend &backend, std::size_t bytes,
    const std::string &what)
{
    if (run.trace && run.trace->on())
        run.trace->line("work=" + std::to_string(run.work_id) + " #" + std::to_string(run.sequence) + " post " + direction_text(direction) + " " + peer_text(peer) + " bytes=" + std::to_string(bytes) + " (" + what + ")");
    group.add(what + " " + direction_text(direction) + " " + peer_text(peer), post_transfer(run.mesh->channel(peer), direction, backend, bytes));
}

} // namespace

OpGroup::~OpGroup()
{
    for (auto &c : children_) c.work.wait();
}

void OpGroup::add(const std::string &what, TransferWork work) { children_.push_back({what, std::move(work)}); }

void OpGroup::wait_all()
{
    for (auto &c : children_) c.work.wait();
    for (auto &c : children_)
    {
        if (c.work.has_error()) throw std::runtime_error("transport_error: " + c.what + ": " + c.work.error());
    }
    children_.clear();
}

TransferWork post_transfer(PeerChannel &channel, TransferDirection direction, AsyncMemoryBackend &backend, std::size_t bytes)
{
    TransferRequest request;
    request.transfer_id = g_transfer_id.fetch_add(1, std::memory_order_relaxed);
    request.direction = direction;
    request.backend = &backend;
    request.transport = channel.data.get();
    request.total_bytes = bytes;
    request.chunk_hint = 0;
    return channel.worker->enqueue(request);
}

void copy_through_providers(ExternalMemoryProvider &from, ExternalMemoryProvider &to, std::size_t bytes)
{
    constexpr std::size_t kCopyChunk = std::size_t{1} << 20;
    std::vector<unsigned char> staging(std::min(bytes, kCopyChunk));
    for (std::size_t offset = 0; offset < bytes; offset += kCopyChunk)
    {
        const Chunk chunk{offset, std::min(kCopyChunk, bytes - offset)};
        from.primary_backend().stage_source_chunk(chunk, staging.data());
        to.primary_backend().commit_destination_chunk(chunk, staging.data());
    }
}

void run_descriptor_exchange(const CollectiveRun &run, const CollectiveDescriptor &mine)
{
    const auto trace = [&](const std::string &text) {
        if (run.trace && run.trace->on()) run.trace->line("work=" + std::to_string(run.work_id) + " #" + std::to_string(run.sequence) + " " + text);
    };
    trace(std::string(collective_kind_name(mine.kind)) + " begin bytes=" + std::to_string(mine.bytes) + " count=" + std::to_string(mine.count) +
          " dtype=" + datatype_label(mine.datatype) + " root=" + std::to_string(mine.root) + " memory=" + memory_kind_name(mine.memory_kind));

    CollectiveVerdict verdict;
    if (run.rank == 0)
    {
        std::vector<DescriptorWire> wire(run.world);
        std::vector<std::unique_ptr<HostPointerAsyncBackend>> backends;
        OpGroup recvs;
        for (std::size_t p = 1; p < run.world; ++p)
        {
            backends.push_back(std::make_unique<HostPointerAsyncBackend>(wire[p].data(), kDescriptorWireSize));
            post(run, recvs, p, TransferDirection::Recv, *backends.back(), kDescriptorWireSize, "descriptor");
        }
        recvs.wait_all();

        std::vector<CollectiveDescriptor> by_rank(run.world);
        by_rank[0] = mine;
        for (std::size_t p = 1; p < run.world; ++p)
        {
            by_rank[p] = decode_descriptor(wire[p]);
            by_rank[p].rank = static_cast<std::uint32_t>(p); // the channel, not the payload, says who sent it
        }
        verdict = judge_collective(by_rank);

        VerdictWire out = encode_verdict(verdict);
        std::vector<std::unique_ptr<HostPointerAsyncBackend>> verdict_backends;
        OpGroup sends;
        for (std::size_t p = 1; p < run.world; ++p)
        {
            verdict_backends.push_back(std::make_unique<HostPointerAsyncBackend>(out.data(), kVerdictWireSize));
            post(run, sends, p, TransferDirection::Send, *verdict_backends.back(), kVerdictWireSize, "verdict");
        }
        sends.wait_all();
    }
    else
    {
        DescriptorWire wire = encode_descriptor(mine);
        VerdictWire in{};
        HostPointerAsyncBackend send_backend(wire.data(), kDescriptorWireSize), recv_backend(in.data(), kVerdictWireSize);
        OpGroup group;
        post(run, group, 0, TransferDirection::Send, send_backend, kDescriptorWireSize, "descriptor");
        post(run, group, 0, TransferDirection::Recv, recv_backend, kVerdictWireSize, "verdict");
        group.wait_all();
        verdict = decode_verdict(in);
    }

    trace(std::string("verdict ") + (verdict.status == VerdictStatus::Ok ? "ok" : verdict.status == VerdictStatus::Unsupported ? "unsupported" : "mismatch") +
          (verdict.text.empty() ? "" : ": " + verdict.text));
    if (verdict.status == VerdictStatus::Unsupported) throw CollectiveRejected(verdict.text);
    if (verdict.status == VerdictStatus::Mismatch) throw CollectiveMismatch("protocol_mismatch: " + verdict.text);
}

void reference_broadcast(const CollectiveRun &run, ExternalMemoryProvider *provider, std::size_t bytes, std::size_t root)
{
    if (bytes == 0 || run.world == 1) return;
    if (run.rank == root)
    {
        for (std::size_t p = 0; p < run.world; ++p)
        {
            if (p == root) continue;
            OpGroup group;
            post(run, group, p, TransferDirection::Send, provider->primary_backend(), bytes, "broadcast");
            group.wait_all();
        }
    }
    else
    {
        OpGroup group;
        post(run, group, root, TransferDirection::Recv, provider->primary_backend(), bytes, "broadcast");
        group.wait_all();
    }
}

void reference_all_gather(
    const CollectiveRun &run, ExternalMemoryProvider *in, std::vector<std::shared_ptr<ExternalMemoryProvider>> &outputs, std::size_t bytes)
{
    if (bytes == 0) return;
    const std::size_t world = run.world;
    if (world == 1)
    {
        if (outputs[0]) copy_through_providers(*in, *outputs[0], bytes);
        return;
    }
    if (run.rank == 0)
    {
        OpGroup recvs;
        for (std::size_t p = 1; p < world; ++p) post(run, recvs, p, TransferDirection::Recv, outputs[p]->primary_backend(), bytes, "all_gather input");
        if (outputs[0]) copy_through_providers(*in, *outputs[0], bytes);
        recvs.wait_all();
        for (std::size_t p = 1; p < world; ++p)
        {
            OpGroup sends;
            for (std::size_t q = 0; q < world; ++q)
            {
                if (q == p) continue; // the peer already holds its own input
                AsyncMemoryBackend &source = (q == 0 && !outputs[0]) ? in->primary_backend() : outputs[q]->primary_backend();
                post(run, sends, p, TransferDirection::Send, source, bytes, "all_gather output " + std::to_string(q));
            }
            sends.wait_all();
        }
    }
    else
    {
        OpGroup group;
        post(run, group, 0, TransferDirection::Send, in->primary_backend(), bytes, "all_gather input");
        for (std::size_t q = 0; q < world; ++q)
        {
            if (q == run.rank) continue;
            post(run, group, 0, TransferDirection::Recv, outputs[q]->primary_backend(), bytes, "all_gather output " + std::to_string(q));
        }
        if (outputs[run.rank]) copy_through_providers(*in, *outputs[run.rank], bytes);
        group.wait_all();
    }
}

void reference_all_reduce(const CollectiveRun &run, ExternalMemoryProvider &provider, std::size_t total_bytes, std::size_t count, DataType datatype)
{
    if (run.world == 1 || total_bytes == 0) return;
    if (run.rank == 0)
    {
        AsyncMemoryBackend &scratch = provider.scratch_backend();
        LocalReduceBackend &reduce = provider.reduce_backend();
        for (std::size_t p = 1; p < run.world; ++p)
        {
            OpGroup group;
            post(run, group, p, TransferDirection::Recv, scratch, total_bytes, "all_reduce contribution");
            group.wait_all();
            reduce.reduce_sum(count, datatype);
        }
        for (std::size_t p = 1; p < run.world; ++p)
        {
            OpGroup group;
            post(run, group, p, TransferDirection::Send, provider.primary_backend(), total_bytes, "all_reduce result");
            group.wait_all();
        }
    }
    else
    {
        OpGroup send;
        post(run, send, 0, TransferDirection::Send, provider.primary_backend(), total_bytes, "all_reduce contribution");
        send.wait_all();
        OpGroup recv;
        post(run, recv, 0, TransferDirection::Recv, provider.primary_backend(), total_bytes, "all_reduce result");
        recv.wait_all();
    }
}

} // namespace tbccl::detail
