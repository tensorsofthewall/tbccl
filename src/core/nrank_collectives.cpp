// Reference N-rank collectives (nrank_collectives.hpp).

#include "nrank_collectives.hpp"

#include "collective_topology.hpp"
#include "host_pointer_backend.hpp"
#include "wire_protocol.hpp"

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

CommAlgorithm run_descriptor_exchange(const CollectiveRun &run, const CollectiveDescriptor &mine)
{
    const auto trace = [&](const std::string &text) {
        if (run.trace && run.trace->on()) run.trace->line("work=" + std::to_string(run.work_id) + " #" + std::to_string(run.sequence) + " " + text);
    };
    trace(std::string(collective_kind_name(mine.kind)) + " begin bytes=" + std::to_string(mine.bytes) + " count=" + std::to_string(mine.count) +
          " dtype=" + datatype_label(mine.datatype) + " root=" + std::to_string(mine.root) + " memory=" + memory_kind_name(mine.memory_kind));

    CollectiveVerdict verdict;
    CollectiveMailbox &mailbox = run.mesh->mailbox();
    if (run.rank == 0)
    {
        // Descriptors arrive on the control plane (one frame per rank, FIFO per peer); no data lane is touched.
        std::vector<CollectiveDescriptor> by_rank(run.world);
        by_rank[0] = mine;
        for (std::size_t p = 1; p < run.world; ++p)
        {
            const auto bytes = mailbox.wait_descriptor(p);
            DescriptorWire wire{};
            std::copy_n(bytes.begin(), std::min(bytes.size(), wire.size()), wire.begin());
            by_rank[p] = decode_descriptor(wire);
            by_rank[p].rank = static_cast<std::uint32_t>(p); // the channel, not the payload, says who sent it
        }
        verdict = judge_collective(by_rank);
        if (verdict.status == VerdictStatus::Ok)
        {
            // The agreed override (ranks that carry one agree, checked by the judge) and the plan: chosen ONCE here and carried to every rank.
            std::uint32_t forced = 0;
            for (const auto &d : by_rank)
                if (d.forced_algorithm != 0) forced = d.forced_algorithm;
            try
            {
                const PlannerThresholds defaults;
                verdict.algorithm = static_cast<std::uint32_t>(
                    plan_collective(mine.kind, run.world, static_cast<std::size_t>(mine.bytes), static_cast<CommAlgorithm>(forced), run.thresholds ? *run.thresholds : defaults).algorithm);
            }
            catch (const std::runtime_error &e)
            {
                verdict.status = VerdictStatus::Unsupported;
                verdict.text = e.what();
            }
        }
        const VerdictWire out = encode_verdict(verdict);
        const std::vector<std::uint8_t> bytes(out.begin(), out.end());
        for (std::size_t p = 1; p < run.world; ++p) run.mesh->send_collective_frame(p, /*is_verdict=*/true, bytes);
    }
    else
    {
        const DescriptorWire wire = encode_descriptor(mine);
        run.mesh->send_collective_frame(0, /*is_verdict=*/false, std::vector<std::uint8_t>(wire.begin(), wire.end()));
        const auto bytes = mailbox.wait_verdict(0);
        VerdictWire in{};
        std::copy_n(bytes.begin(), std::min(bytes.size(), in.size()), in.begin());
        verdict = decode_verdict(in);
    }

    trace(std::string("verdict ") + (verdict.status == VerdictStatus::Ok ? "ok" : verdict.status == VerdictStatus::Unsupported ? "unsupported" : "mismatch") +
          (verdict.text.empty() ? "" : ": " + verdict.text));
    if (verdict.status == VerdictStatus::Unsupported) throw CollectiveRejected(verdict.text);
    if (verdict.status == VerdictStatus::Mismatch) throw CollectiveMismatch("protocol_mismatch: " + verdict.text);
    trace(std::string("plan: ") + comm_algorithm_name(static_cast<CommAlgorithm>(verdict.algorithm)));
    return static_cast<CommAlgorithm>(verdict.algorithm);
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

// Binomial-tree broadcast (the N>2 collective-selection work): ceil(log2 N) levels. A node receives the whole payload from its parent, then sends it to its children
// (largest subtree first). Children are served concurrently when the backend can be read by several lanes at once (direct host access); otherwise one after the
// other (a staged/CUDA backend is not shared between concurrently running lanes).
void tree_broadcast(const CollectiveRun &run, ExternalMemoryProvider *provider, std::size_t bytes, std::size_t root)
{
    if (bytes == 0 || run.world == 1) return;
    AsyncMemoryBackend &backend = provider->primary_backend();
    if (run.rank != root)
    {
        OpGroup group;
        post(run, group, tree_parent(run.rank, root, run.world), TransferDirection::Recv, backend, bytes, "tree broadcast");
        group.wait_all();
    }
    const auto children = tree_children(run.rank, root, run.world);
    if (backend.supports_direct_transport_access())
    {
        OpGroup group;
        for (std::size_t child : children) post(run, group, child, TransferDirection::Send, backend, bytes, "tree broadcast");
        group.wait_all();
    }
    else
    {
        for (std::size_t child : children)
        {
            OpGroup group;
            post(run, group, child, TransferDirection::Send, backend, bytes, "tree broadcast");
            group.wait_all();
        }
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

void run_dissemination_barrier(const CollectiveRun &run, const CollectiveDescriptor &mine)
{
    const std::size_t n = run.world, r = run.rank;
    CollectiveMailbox &mailbox = run.mesh->mailbox();
    auto trace = [&](const std::string &text) {
        if (run.trace && run.trace->on()) run.trace->line("work=" + std::to_string(run.work_id) + " #" + std::to_string(run.sequence) + " " + text);
    };
    trace("barrier (dissemination) begin");

    struct Current
    {
        CollectiveMailbox &m;
        bool on;
        ~Current()
        {
            if (on) m.clear_current();
        }
    } current{mailbox, r == 0};
    if (r == 0) mailbox.set_current(mine, run.fatal ? *run.fatal : std::function<void(const std::string &)>());
    else
    {
        const DescriptorWire wire = encode_descriptor(mine);
        run.mesh->send_collective_frame(0, /*is_verdict=*/false, std::vector<std::uint8_t>(wire.begin(), wire.end()));
    }

    std::size_t round = 0;
    for (std::size_t dist = 1; dist < n; dist <<= 1, ++round)
    {
        const std::size_t to = (r + dist) % n, from = (r + n - dist) % n;
        std::uint8_t out[16], in[16] = {};
        put_u64(out, run.sequence);
        put_u32(out + 8, static_cast<std::uint32_t>(mine.kind));
        put_u32(out + 12, static_cast<std::uint32_t>(round));
        HostPointerAsyncBackend send_backend(out, sizeof(out)), recv_backend(in, sizeof(in));
        OpGroup group;
        post(run, group, to, TransferDirection::Send, send_backend, sizeof(out), "barrier token");
        post(run, group, from, TransferDirection::Recv, recv_backend, sizeof(in), "barrier token");
        group.wait_all();
        if (get_u64(in) != run.sequence || get_u32(in + 8) != static_cast<std::uint32_t>(mine.kind) || get_u32(in + 12) != static_cast<std::uint32_t>(round))
        {
            throw CollectiveMismatch(
                "protocol_mismatch: barrier token from rank " + std::to_string(from) + " in round " + std::to_string(round) + " does not match this rank's collective #" +
                std::to_string(run.sequence) + " (" + collective_kind_name(mine.kind) + "): it carries #" + std::to_string(get_u64(in)));
        }
    }

    if (r == 0)
    {
        // Consume (and judge) the descriptors every rank handed over, so the control-plane queue stays aligned with the next collective.
        std::vector<CollectiveDescriptor> by_rank(n);
        by_rank[0] = mine;
        for (std::size_t p = 1; p < n; ++p)
        {
            const auto bytes = mailbox.wait_descriptor(p);
            DescriptorWire wire{};
            std::copy_n(bytes.begin(), std::min(bytes.size(), wire.size()), wire.begin());
            by_rank[p] = decode_descriptor(wire);
            by_rank[p].rank = static_cast<std::uint32_t>(p);
        }
        const auto verdict = judge_collective(by_rank);
        if (verdict.status == VerdictStatus::Mismatch) throw CollectiveMismatch("protocol_mismatch: " + verdict.text);
    }
    trace("barrier (dissemination) done");
}

namespace
{
[[noreturn]] void not_implemented(const char *kind, CommAlgorithm a)
{
    throw std::runtime_error(std::string("internal_error: ") + kind + " algorithm '" + comm_algorithm_name(a) + "' is not implemented");
}
} // namespace

void run_barrier(const CollectiveRun &, CommAlgorithm algorithm)
{
    switch (algorithm)
    {
    case CommAlgorithm::N2FastPath: // barrier has no specialised N=2 engine: it is the same exchange
    case CommAlgorithm::Reference: return; // the descriptor exchange (gather at rank 0, verdict back) is the barrier
    case CommAlgorithm::Dissemination: throw std::runtime_error("internal_error: the dissemination barrier does not run after a verdict exchange");
    default: not_implemented("barrier", algorithm);
    }
}

void run_broadcast(const CollectiveRun &run, CommAlgorithm algorithm, ExternalMemoryProvider *provider, std::size_t bytes, std::size_t root)
{
    switch (algorithm)
    {
    case CommAlgorithm::Reference: reference_broadcast(run, provider, bytes, root); return;
    case CommAlgorithm::BinomialTree: tree_broadcast(run, provider, bytes, root); return;
    default: not_implemented("broadcast", algorithm);
    }
}

void run_all_gather(
    const CollectiveRun &run, CommAlgorithm algorithm, ExternalMemoryProvider *in, std::vector<std::shared_ptr<ExternalMemoryProvider>> &outputs, std::size_t bytes)
{
    switch (algorithm)
    {
    case CommAlgorithm::Reference: reference_all_gather(run, in, outputs, bytes); return;
    default: not_implemented("all_gather", algorithm);
    }
}

void run_all_reduce(
    const CollectiveRun &run, CommAlgorithm algorithm, ExternalMemoryProvider &provider, std::size_t total_bytes, std::size_t count, DataType datatype)
{
    switch (algorithm)
    {
    case CommAlgorithm::Reference: reference_all_reduce(run, provider, total_bytes, count, datatype); return;
    default: not_implemented("all_reduce", algorithm);
    }
}

} // namespace tbccl::detail
