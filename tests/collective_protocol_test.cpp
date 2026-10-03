// Collective descriptor / verdict encoding and the coordinator's judgement. Pure: no sockets.

#include "collective_protocol.hpp"

#include "test_utils.hpp"

#include <iostream>

using tbccl_test::expect;
using namespace tbccl;
using namespace tbccl::detail;

namespace
{

    CollectiveDescriptor make(std::size_t rank, CollectiveKind kind)
    {
        CollectiveDescriptor d;
        d.sequence = 17;
        d.kind = kind;
        d.rank = static_cast<std::uint32_t>(rank);
        d.root = 1;
        d.datatype = DataType::Float32;
        d.count = 1000;
        d.bytes = 4000;
        return d;
    }

    std::vector<CollectiveDescriptor> world(std::size_t n, CollectiveKind kind)
    {
        std::vector<CollectiveDescriptor> v;
        for (std::size_t r = 0; r < n; ++r) v.push_back(make(r, kind));
        return v;
    }

    bool has(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

} // namespace

int main()
{
    // Encoding round trips
    {
        CollectiveDescriptor d = make(3, CollectiveKind::AllReduce);
        d.local_status = 1;
        d.note = "metal-shared cannot reduce int8";
        d.memory_kind = MemoryKind::MetalShared;
        d.datatype = DataType::Int8;
        d.sequence = 0xFFFFFFFFFFFFFFF0ull;
        const auto e = decode_descriptor(encode_descriptor(d));
        expect(e.sequence == d.sequence && e.kind == d.kind && e.rank == 3 && e.root == 1 && e.datatype == DataType::Int8 && e.count == 1000 && e.bytes == 4000 &&
                   e.memory_kind == MemoryKind::MetalShared && e.local_status == 1 && e.note == d.note,
               "descriptor round trip");
        CollectiveVerdict v;
        v.status = VerdictStatus::Mismatch;
        v.sequence = 9;
        v.text = std::string(400, 'x');
        const auto w = decode_verdict(encode_verdict(v));
        expect(w.status == VerdictStatus::Mismatch && w.sequence == 9 && w.text.size() == kVerdictTextBytes - 1, "verdict round trip, long text truncated");
    }

    // Agreement
    for (auto kind : {CollectiveKind::Barrier, CollectiveKind::Broadcast, CollectiveKind::AllGather, CollectiveKind::AllReduce})
    {
        const auto v = judge_collective(world(4, kind));
        expect(v.status == VerdictStatus::Ok, std::string("agreeing ranks pass: ") + collective_kind_name(kind));
    }

    // Each way to disagree
    {
        auto d = world(3, CollectiveKind::AllReduce);
        d[2].kind = CollectiveKind::Broadcast;
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Mismatch && has(v.text, "rank 0 called all_reduce but rank 2 called broadcast") && has(v.text, "#17"), "wrong collective kind: " + v.text);
    }
    {
        auto d = world(3, CollectiveKind::AllReduce);
        d[1].sequence = 18;
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Mismatch && has(v.text, "rank 1 is at collective #18") && has(v.text, "rank 0 is at #17"), "wrong sequence: " + v.text);
    }
    {
        auto d = world(4, CollectiveKind::Broadcast);
        d[3].root = 2;
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Mismatch && has(v.text, "root differs") && has(v.text, "rank 3"), "wrong root: " + v.text);
    }
    {
        auto d = world(3, CollectiveKind::AllReduce);
        d[1].count = 999;
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Mismatch && has(v.text, "element count differs"), "wrong count: " + v.text);
    }
    {
        auto d = world(3, CollectiveKind::AllReduce);
        d[2].datatype = DataType::Int32;
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Mismatch && has(v.text, "dtype differs") && has(v.text, "float32") , "wrong dtype: " + v.text);
    }
    {
        auto d = world(3, CollectiveKind::AllGather);
        d[2].bytes = 8;
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Mismatch && has(v.text, "byte count differs"), "wrong byte count: " + v.text);
    }
    {
        auto d = world(3, CollectiveKind::Barrier);
        d[1].bytes = 99; // irrelevant to a barrier
        d[1].root = 3;
        expect(judge_collective(d).status == VerdictStatus::Ok, "a barrier compares only sequence and kind");
    }

    // Capability: Unsupported only when everything else agrees; Mismatch beats it.
    {
        auto d = world(3, CollectiveKind::AllReduce);
        d[2].local_status = 1;
        d[2].memory_kind = MemoryKind::MetalShared;
        d[2].note = "dtype=int8 is not available for this memory kind";
        auto v = judge_collective(d);
        expect(v.status == VerdictStatus::Unsupported && has(v.text, "rank 2") && has(v.text, "metal-shared") && has(v.text, "int8"), "unsupported names the rank and kind: " + v.text);
        d[1].count = 1;
        expect(judge_collective(d).status == VerdictStatus::Mismatch, "a mismatch outranks an unsupported rank");
    }

    std::cout << "collective_protocol_test passed\n";
    return 0;
}
