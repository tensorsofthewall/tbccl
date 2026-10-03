// Phase 50: collective descriptor / verdict encoding and the coordinator's judgement (collective_protocol.hpp).

#include "collective_protocol.hpp"

#include "wire_protocol.hpp"

#include <cstring>

namespace tbccl::detail
{

const char *collective_kind_name(CollectiveKind kind) noexcept
{
    switch (kind)
    {
    case CollectiveKind::Barrier: return "barrier";
    case CollectiveKind::Broadcast: return "broadcast";
    case CollectiveKind::AllGather: return "all_gather";
    case CollectiveKind::AllReduce: return "all_reduce";
    }
    return "unknown";
}

DescriptorWire encode_descriptor(const CollectiveDescriptor &d)
{
    DescriptorWire wire{};
    std::uint8_t *out = wire.data();
    put_u64(out + 0, d.sequence);
    put_u32(out + 8, static_cast<std::uint32_t>(d.kind));
    put_u32(out + 12, d.rank);
    put_u32(out + 16, d.root);
    put_u32(out + 20, static_cast<std::uint32_t>(d.datatype));
    put_u32(out + 24, static_cast<std::uint32_t>(d.reduce_op));
    put_u32(out + 28, static_cast<std::uint32_t>(d.memory_kind));
    put_u64(out + 32, d.count);
    put_u64(out + 40, d.bytes);
    put_u32(out + 48, d.local_status);
    put_text(out + 52, kDescriptorNoteBytes, d.note);
    put_u32(out + 124, d.forced_algorithm);
    return wire;
}

CollectiveDescriptor decode_descriptor(const DescriptorWire &wire)
{
    const std::uint8_t *in = wire.data();
    CollectiveDescriptor d;
    d.sequence = get_u64(in + 0);
    d.kind = static_cast<CollectiveKind>(get_u32(in + 8));
    d.rank = get_u32(in + 12);
    d.root = get_u32(in + 16);
    d.datatype = static_cast<DataType>(get_u32(in + 20));
    d.reduce_op = static_cast<ReduceOp>(get_u32(in + 24));
    d.memory_kind = static_cast<MemoryKind>(get_u32(in + 28));
    d.count = get_u64(in + 32);
    d.bytes = get_u64(in + 40);
    d.local_status = get_u32(in + 48);
    d.note = get_text(in + 52, kDescriptorNoteBytes);
    d.forced_algorithm = get_u32(in + 124);
    return d;
}

VerdictWire encode_verdict(const CollectiveVerdict &v)
{
    VerdictWire wire{};
    std::uint8_t *out = wire.data();
    put_u32(out + 0, static_cast<std::uint32_t>(v.status));
    put_u32(out + 4, v.algorithm);
    put_u64(out + 8, v.sequence);
    put_text(out + 16, kVerdictTextBytes, v.text);
    return wire;
}

CollectiveVerdict decode_verdict(const VerdictWire &wire)
{
    const std::uint8_t *in = wire.data();
    CollectiveVerdict v;
    v.status = static_cast<VerdictStatus>(get_u32(in + 0));
    v.algorithm = get_u32(in + 4);
    v.sequence = get_u64(in + 8);
    v.text = get_text(in + 16, kVerdictTextBytes);
    return v;
}

namespace
{

std::string rank_text(std::uint32_t r) { return "rank " + std::to_string(r); }

} // namespace

CollectiveVerdict judge_collective(const std::vector<CollectiveDescriptor> &by_rank)
{
    CollectiveVerdict verdict;
    const CollectiveDescriptor &ref = by_rank.at(0);
    verdict.sequence = ref.sequence;

    auto mismatch = [&](const std::string &text) {
        verdict.status = VerdictStatus::Mismatch;
        verdict.text = "collective mismatch at #" + std::to_string(ref.sequence) + ": " + text;
        return verdict;
    };

    for (std::size_t r = 1; r < by_rank.size(); ++r)
    {
        const auto &d = by_rank[r];
        const std::string who = rank_text(d.rank);
        if (d.sequence != ref.sequence)
            return mismatch(who + " is at collective #" + std::to_string(d.sequence) + " (" + collective_kind_name(d.kind) + ") but rank 0 is at #" + std::to_string(ref.sequence) + " (" + collective_kind_name(ref.kind) + ")");
        if (d.kind != ref.kind)
            return mismatch(std::string("rank 0 called ") + collective_kind_name(ref.kind) + " but " + who + " called " + collective_kind_name(d.kind));
        if (ref.kind == CollectiveKind::Broadcast && d.root != ref.root)
            return mismatch("broadcast root differs: rank 0 says root=" + std::to_string(ref.root) + ", " + who + " says root=" + std::to_string(d.root));
        if (ref.kind == CollectiveKind::AllReduce)
        {
            if (d.datatype != ref.datatype)
                return mismatch(std::string("all_reduce dtype differs: rank 0 uses ") + datatype_label(ref.datatype) + ", " + who + " uses " + datatype_label(d.datatype));
            if (d.reduce_op != ref.reduce_op) return mismatch("all_reduce reduce op differs between rank 0 and " + who);
            if (d.count != ref.count)
                return mismatch("all_reduce element count differs: rank 0 has " + std::to_string(ref.count) + ", " + who + " has " + std::to_string(d.count));
        }
        if (d.forced_algorithm != 0 && ref.forced_algorithm != 0 && d.forced_algorithm != ref.forced_algorithm)
            return mismatch("ranks force different algorithms: rank 0 forces " + std::to_string(ref.forced_algorithm) + ", " + who + " forces " + std::to_string(d.forced_algorithm));
        if (ref.kind != CollectiveKind::Barrier && d.bytes != ref.bytes)
            return mismatch(std::string(collective_kind_name(ref.kind)) + " byte count differs: rank 0 has " + std::to_string(ref.bytes) + ", " + who + " has " + std::to_string(d.bytes));
    }

    for (const auto &d : by_rank)
    {
        if (d.local_status != 0)
        {
            verdict.status = VerdictStatus::Unsupported;
            verdict.text = std::string("unsupported: ") + collective_kind_name(ref.kind) + " cannot run: " + rank_text(d.rank) + " (" + memory_kind_name(d.memory_kind) + "): " + d.note;
            return verdict;
        }
    }
    return verdict;
}

} // namespace tbccl::detail
