#pragma once

// Phase 50: the conservative REFERENCE collectives of the N-rank Communicator, private to libtbccl. They exist for correctness and for the
// runtime structure (peer channels, collective sequencing, abort, capabilities), not for speed: Phase 51 owns optimized algorithms.
//
//   barrier     descriptor exchange only (every rank -> rank 0, verdict back)
//   broadcast   root -> every other rank, sequential fan-out
//   all_gather  every rank -> rank 0, rank 0 assembles outputs in rank order and fans out; sequential
//   all_reduce  every rank -> rank 0 in rank order 1..N-1 (a fixed reduction order, so a result is reproducible), rank 0 reduces
//               into its buffer through the memory provider's reduce backend, then sends the result to every rank; sequential
//
// All of them run on the Communicator's single collective-executor thread (the collective ordering domain) and post their transfers to
// PeerChannel lanes. A child failure never leaves another child running against the caller's buffer: OpGroup waits for every child.

#include "collective_plan.hpp"
#include "collective_protocol.hpp"
#include "connection_manager.hpp"
#include "trace.hpp"

#include <tbccl/communicator.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tbccl::detail
{

struct CollectiveRun
{
    std::size_t rank = 0;
    std::size_t world = 0;
    ConnectionManager *mesh = nullptr;
    std::uint64_t sequence = 0;
    std::uint64_t work_id = 0;
    const Trace *trace = nullptr;
    const PlannerThresholds *thresholds = nullptr; // used by rank 0 only
    const std::function<void(const std::string &)> *fatal = nullptr; // aborts the communicator (used by rank 0's descriptor validation)
};

// The children of one collective. wait_all() waits for EVERY child (even after the first failure) and then throws the first error in
// posting order, so a failing collective never returns while a transport thread can still touch the caller's buffer. The destructor
// waits too, for the same reason on an unwinding path.
class OpGroup
{
public:
    OpGroup() = default;
    ~OpGroup();
    OpGroup(const OpGroup &) = delete;
    OpGroup &operator=(const OpGroup &) = delete;

    void add(const std::string &what, TransferWork work);
    void wait_all();

private:
    struct Child
    {
        std::string what;
        TransferWork work;
    };
    std::vector<Child> children_;
};

// Posts one unframed transfer of `bytes` bytes through `backend` on the matching lane of `peer`'s channel.
TransferWork post_transfer(PeerChannel &channel, TransferDirection direction, AsyncMemoryBackend &backend, std::size_t bytes);

// Copies `bytes` bytes from `from` to `to` through the providers' staging interface (no device code here).
void copy_through_providers(ExternalMemoryProvider &from, ExternalMemoryProvider &to, std::size_t bytes);

// Descriptor exchange with rank 0 and the verdict. Returns the algorithm rank 0 chose (the same on every rank) on Ok; throws CollectiveRejected (Unsupported) or
// CollectiveMismatch.
CommAlgorithm run_descriptor_exchange(const CollectiveRun &run, const CollectiveDescriptor &mine);

void reference_broadcast(const CollectiveRun &run, ExternalMemoryProvider *provider, std::size_t bytes, std::size_t root);

// Phase 51: execute the data phase of a collective with the algorithm rank 0 chose (the same on every rank). world_size > 2 only; an algorithm that is not
// implemented for the collective is an internal_error (the planner never returns one).
void tree_broadcast(const CollectiveRun &run, ExternalMemoryProvider *provider, std::size_t bytes, std::size_t root);
void ring_all_gather(
    const CollectiveRun &run, ExternalMemoryProvider *in, std::vector<std::shared_ptr<ExternalMemoryProvider>> &outputs, std::size_t bytes);
void tree_all_reduce(const CollectiveRun &run, ExternalMemoryProvider &provider, std::size_t total_bytes, std::size_t count, DataType datatype);
void run_barrier(const CollectiveRun &run, CommAlgorithm algorithm);

// The dissemination barrier (Phase 51): ceil(log2 N) rounds, each rank sends a token to (rank + 2^k) mod N and receives one from (rank - 2^k) mod N. There is no
// coordinator round trip: every rank hands its descriptor to rank 0 on the control plane without waiting for a verdict, rank 0 validates them as they arrive, and
// tokens carry (sequence, kind) which each receiver checks. Throws CollectiveMismatch on any disagreement.
void run_dissemination_barrier(const CollectiveRun &run, const CollectiveDescriptor &mine);
void run_broadcast(const CollectiveRun &run, CommAlgorithm algorithm, ExternalMemoryProvider *provider, std::size_t bytes, std::size_t root);
void run_all_gather(
    const CollectiveRun &run, CommAlgorithm algorithm, ExternalMemoryProvider *in, std::vector<std::shared_ptr<ExternalMemoryProvider>> &outputs, std::size_t bytes);
void run_all_reduce(
    const CollectiveRun &run, CommAlgorithm algorithm, ExternalMemoryProvider &provider, std::size_t total_bytes, std::size_t count, DataType datatype);

// `outputs[r]` receives rank r's input on every rank; `in` and the outputs are never null when bytes > 0.
void reference_all_gather(
    const CollectiveRun &run, ExternalMemoryProvider *in, std::vector<std::shared_ptr<ExternalMemoryProvider>> &outputs, std::size_t bytes);

void reference_all_reduce(const CollectiveRun &run, ExternalMemoryProvider &provider, std::size_t total_bytes, std::size_t count, DataType datatype);

} // namespace tbccl::detail
