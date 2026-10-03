#pragma once

// Collective sequencing and descriptor validation, private to libtbccl.
//
// Every Communicator keeps a collective sequence number (0, 1, 2, ... in the order collectives run) and, for the N-rank
// reference algorithms, begins every collective with a descriptor exchange: each non-coordinator rank sends a fixed-size
// CollectiveDescriptor to rank 0 on its data lane, rank 0 judges all descriptors with judge_collective(), and answers every rank
// with a fixed-size CollectiveVerdict. Only an Ok verdict lets any payload move. This turns "rank 0 called all_reduce #17 while
// rank 1 called broadcast #17" (a silent hang or stream corruption without it) into a clear error on every rank:
//
//   Ok           proceed
//   Unsupported  a rank-local capability is missing (an unregistered memory kind, a dtype its memory kind cannot reduce); every
//                rank consumed the same descriptor and verdict and no payload moved, so the communicator stays usable
//   Mismatch     the ranks disagree on sequence, kind, root, element count, dtype, reduce op or byte count; the collective sequence
//                can no longer be trusted, so the communicator is poisoned and aborted everywhere
//
// Wire format: fixed-size, big-endian (wire_protocol.hpp helpers). The N=2 specialised paths (all_reduce / broadcast / all_gather at
// world_size 2) deliberately skip the exchange: its extra round trip would regress them. Their sequence counter still advances.

#include <tbccl/reduction.hpp>
#include <tbccl/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tbccl::detail
{

enum class CollectiveKind : std::uint32_t
{
    Barrier = 1,
    Broadcast = 2,
    AllGather = 3,
    AllReduce = 4,
};

const char *collective_kind_name(CollectiveKind kind) noexcept;

struct CollectiveDescriptor
{
    std::uint64_t sequence = 0;
    CollectiveKind kind = CollectiveKind::Barrier;
    std::uint32_t rank = 0;
    std::uint32_t root = 0;       // Broadcast only
    DataType datatype = DataType::UInt8;   // AllReduce only
    ReduceOp reduce_op = ReduceOp::Sum;    // AllReduce only
    MemoryKind memory_kind = MemoryKind::Host;
    std::uint64_t count = 0;      // AllReduce only
    std::uint64_t bytes = 0;      // Broadcast / AllGather (per rank) / AllReduce
    // 0 = this rank can run the collective; 1 = it cannot, and `note` says why.
    std::uint32_t local_status = 0;
    std::string note;
    // The algorithm this rank's debug override forces (a CommAlgorithm value; 0 = no override). Ranks that force different algorithms are a mismatch.
    std::uint32_t forced_algorithm = 0;
};

constexpr std::size_t kDescriptorWireSize = 128;
using DescriptorWire = std::array<std::uint8_t, kDescriptorWireSize>;
constexpr std::size_t kDescriptorNoteBytes = 72;

DescriptorWire encode_descriptor(const CollectiveDescriptor &d);
CollectiveDescriptor decode_descriptor(const DescriptorWire &in);

enum class VerdictStatus : std::uint32_t
{
    Ok = 0,
    Unsupported = 1,
    Mismatch = 2,
};

struct CollectiveVerdict
{
    VerdictStatus status = VerdictStatus::Ok;
    std::uint64_t sequence = 0;
    std::string text;
    // The algorithm every rank must execute (a CommAlgorithm value), chosen once by rank 0. Meaningful when status == Ok.
    std::uint32_t algorithm = 0;
};

constexpr std::size_t kVerdictWireSize = 256;
using VerdictWire = std::array<std::uint8_t, kVerdictWireSize>;
constexpr std::size_t kVerdictTextBytes = 232;

VerdictWire encode_verdict(const CollectiveVerdict &v);
CollectiveVerdict decode_verdict(const VerdictWire &in);

// The coordinator's judgement of one collective. `by_rank[r]` is rank r's descriptor (rank 0's own at index 0). Mismatch beats
// Unsupported (a disagreement means the rest of the information is not about the same collective). Pure.
CollectiveVerdict judge_collective(const std::vector<CollectiveDescriptor> &by_rank);

// Thrown on every rank after an Unsupported verdict: the Work fails, the communicator is NOT poisoned.
class CollectiveRejected : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Thrown after a Mismatch verdict: fatal, the executor poisons and aborts the communicator.
class CollectiveMismatch : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace tbccl::detail
