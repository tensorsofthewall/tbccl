#include "../transport/latency_trace.hpp"
#include <tbccl/communicator.hpp>
#include <tbccl/error.hpp>

#include <tbccl/collectives.hpp>
#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/tcp_world.hpp>
#include <tbccl/transport.hpp>

#include "bootstrap_config.hpp"
#include "connection_manager.hpp"
#include "collective_protocol.hpp"
#include "communicator_debug.hpp"
#include "host_pointer_backend.hpp"
#include "nrank_collectives.hpp"
#include "trace.hpp"
#include "reduction_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <new>
#include <condition_variable>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace tbccl
{

namespace
{

// ---------------------------------------------------------------------
// Built-in Host / MetalShared provider.
//
// A deliberately minimal, internal duplicate of
// benchmarks/tensor/host_async_backend.hpp's and host_reduce_backend.hpp's
// logic, rather than including those benchmark-only headers from core
// tbccl (public installed headers -- and, by the same
// reasoning, core's own internal implementation -- must not depend on
// benchmarks/, which is structurally built ON TOP of core, never the
// reverse). Both pieces are small and proven; this is a deliberate,
// documented duplication, not a design gap.
// ---------------------------------------------------------------------

class HostPointerReduceBackend final : public LocalReduceBackend
{
public:
    HostPointerReduceBackend(void *local_and_output, const void *peer)
        : local_and_output_(local_and_output), peer_(peer)
    {
    }

    void reduce_sum(std::size_t count, DataType datatype) override
    {
        detail::visit_reduction_type(datatype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            detail::apply_reduction(static_cast<T *>(local_and_output_), static_cast<const T *>(peer_), count, ReduceOp::Sum);
        });
    }

private:
    void *local_and_output_;
    const void *peer_;
};

class HostMemoryProvider final : public ExternalMemoryProvider
{
public:
    HostMemoryProvider(const BufferView &buffer, const ExecutionContext & /*context*/)
        : primary_(buffer.data, buffer.bytes), bytes_(buffer.bytes)
    {
    }

    AsyncMemoryBackend &primary_backend() override { return primary_; }

    AsyncMemoryBackend &scratch_backend() override
    {
        if (!scratch_)
        {
            scratch_storage_.assign(bytes_, std::byte{0});
            scratch_ = std::make_unique<detail::HostPointerAsyncBackend>(scratch_storage_.data(), bytes_);
        }
        return *scratch_;
    }

    LocalReduceBackend &reduce_backend() override
    {
        // Force scratch_ to exist first so reduce_ can safely point at it.
        scratch_backend();
        if (!reduce_)
        {
            reduce_ = std::make_unique<HostPointerReduceBackend>(primary_.data(), scratch_->data());
        }
        return *reduce_;
    }

    LocalReduceBackend &reduce_backend_range(std::size_t byte_offset) override
    {
        scratch_backend();
        range_reduce_ = std::make_unique<HostPointerReduceBackend>(
            static_cast<std::byte *>(primary_.data()) + byte_offset, static_cast<const std::byte *>(scratch_->data()) + byte_offset);
        return *range_reduce_;
    }

private:
    detail::HostPointerAsyncBackend primary_;
    std::size_t bytes_;
    std::vector<std::byte> scratch_storage_;
    std::unique_ptr<detail::HostPointerAsyncBackend> scratch_;
    std::unique_ptr<HostPointerReduceBackend> reduce_;
    std::unique_ptr<HostPointerReduceBackend> range_reduce_;
};

// ---------------------------------------------------------------------
// Provider factory registry (extension point).
// ---------------------------------------------------------------------

std::mutex &registry_mutex()
{
    static std::mutex m;
    return m;
}

std::map<MemoryKind, MemoryProviderFactoryEx> &registry()
{
    static std::map<MemoryKind, MemoryProviderFactoryEx> r = {
        {MemoryKind::Host, [](const BufferView &b, const ExecutionContext &c, ProviderResourceSlot &) -> std::unique_ptr<ExternalMemoryProvider> {
             return std::make_unique<HostMemoryProvider>(b, c);
         }},
        {MemoryKind::MetalShared, [](const BufferView &b, const ExecutionContext &c, ProviderResourceSlot &) -> std::unique_ptr<ExternalMemoryProvider> {
             return std::make_unique<HostMemoryProvider>(b, c);
         }},
    };
    return r;
}

} // namespace

void register_memory_provider_factory(MemoryKind kind, MemoryProviderFactory factory)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[kind] = [factory = std::move(factory)](
                           const BufferView &b, const ExecutionContext &c, ProviderResourceSlot &) { return factory(b, c); };
}

void register_memory_provider_factory_ex(MemoryKind kind, MemoryProviderFactoryEx factory)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[kind] = std::move(factory);
}

bool memory_kind_registered(MemoryKind kind)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    return registry().count(kind) != 0;
}

namespace
{

std::unique_ptr<ExternalMemoryProvider> make_provider(
    const BufferView &buffer, const ExecutionContext &context, std::map<MemoryKind, ProviderResourceSlot> &slots)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = registry().find(buffer.memory_kind);
    if (it == registry().end())
    {
        throw Error(ErrorCode::Unsupported, 
            "unsupported: no memory provider registered for kind=" + memory_kind_name(buffer.memory_kind) +
            " (call tbccl::register_memory_provider_factory() first)");
    }
    return it->second(buffer, context, slots[buffer.memory_kind]);
}

// ---------------------------------------------------------------------
// Collective executor: a generalized, promoted BucketAllReduceWorker
// (the bucketed all-reduce overlap work) -- one persistent thread
// driving n2_all_reduce_tensor() jobs one at a time (matching that
// function's own "at most one AllReduce-related transfer in flight"
// contract), returning a Work per job immediately. Unlike
// BucketAllReduceWorker, a job failure does NOT poison all subsequently
// queued jobs -- each job's failure is isolated to its own Work (Part
// Q: communicator-owned Work semantics should not silently wedge
// unrelated future operations). A communicator-level transport/protocol
// failure is still tracked separately (failed_) and does deliberately
// fail all future operations.
// ---------------------------------------------------------------------

struct CollectiveJob
{
    std::function<void()> run; // throws on failure
    // Holds the public Work handle itself (copyable, shared state)
    // rather than naming the private TransferWork::State type directly
    // -- only detail::TransferWorkAccess (a friend of TransferWork) may
    // name that type; non-friend code like this file can only obtain it
    // via state_of() used inline as a function-call argument (see run()
    // below), never stored in a variable with an explicit type.
    Work work;
    std::uint64_t lat_id = 0; // the cold-progress work trace id of the collective's Work (0 = untraced)
};

class CollectiveExecutor
{
public:
    CollectiveExecutor() : thread_([this] { run(); }) {}

    ~CollectiveExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    CollectiveExecutor(const CollectiveExecutor &) = delete;
    CollectiveExecutor &operator=(const CollectiveExecutor &) = delete;

    // Terminal and idempotent. Queued jobs never started, so they fail without touching user memory; the active
    // job (if any) unwinds when the Transport is interrupted and its Work fails after it returns.
    void abort(const std::string &reason)
    {
        std::deque<CollectiveJob> doomed;
        std::string message;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (aborted_) return;
            aborted_ = true;
            message = "aborted: communicator aborted" + (reason.empty() ? "" : " (" + reason + ")");
            abort_message_ = message;
            doomed.swap(queue_);
        }
        cv_.notify_all();
        // wait_idle() sleeps until the queue is empty and nothing runs. Emptying the queue just now can make that true for a caller that started waiting while a
        // submitted job was still queued; the worker has nothing left to wake it for, so without this the caller would sleep forever (concurrent abort() calls).
        idle_cv_.notify_all();
        for (auto &job : doomed)
        {
            unfinished_.fetch_sub(1, std::memory_order_acq_rel);
            detail::TransferWorkAccess::complete_error(detail::TransferWorkAccess::state_of(job.work), ErrorCode::Aborted, message);
        }
    }

    // True while a submitted job has not finished. Unlike running_/queue_ (cleared after a Work completes) the counter drops BEFORE the job's
    // Work becomes terminal, so a caller that has waited for every Work never sees the executor busy (the destructor treats "busy" as "abort").
    bool busy() const { return unfinished_.load(std::memory_order_acquire) > 0; }

    void set_fatal_handler(std::function<void(const std::string &)> handler) { on_fatal_ = std::move(handler); }

    void wait_idle()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_cv_.wait(lock, [&] { return !running_ && queue_.empty(); });
    }

    Work submit(std::function<void()> run)
    {
        std::uint64_t lat_id = 0;
        if (detail::lat_on())
        {
            lat_id = detail::LatencyTrace::get().new_id();
            detail::lat_event(detail::kLatCollSubmit, lat_id);
            detail::tl_lat_next_work_id = lat_id;
        }
        Work work = detail::TransferWorkAccess::make();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (aborted_) throw Error(ErrorCode::Aborted, abort_message_);
            queue_.push_back(CollectiveJob{std::move(run), work, lat_id});
            unfinished_.fetch_add(1, std::memory_order_acq_rel);
        }
        cv_.notify_all();
        detail::lat_event(detail::kLatCollNotified, lat_id);
        return work;
    }

private:
    void run()
    {
        for (;;)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            CollectiveJob job = std::move(queue_.front());
            queue_.pop_front();
            running_ = true;
            lock.unlock();
            detail::tl_lat_parent_id = job.lat_id;
            detail::lat_event(detail::kLatCollExecWake, job.lat_id);
            try
            {
                job.run();
                unfinished_.fetch_sub(1, std::memory_order_acq_rel);
                detail::TransferWorkAccess::complete_ok(detail::TransferWorkAccess::state_of(job.work));
            }
            catch (const detail::CollectiveRejected &e)
            {
                unfinished_.fetch_sub(1, std::memory_order_acq_rel);
                // Every rank consumed the same descriptor and verdict and no payload moved: the Work fails, the communicator stays usable.
                detail::TransferWorkAccess::complete_error(detail::TransferWorkAccess::state_of(job.work), e.code(), e.what());
            }
            catch (const std::exception &e)
            {
                // A job fails only after protocol participation began (arguments were validated before
                // submission): the collective sequence is no longer trustworthy, so poison the communicator.
                unfinished_.fetch_sub(1, std::memory_order_acq_rel);
                std::string aborted_text;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (aborted_) aborted_text = abort_message_; // already aborted before this job failed: a plain transport error is then a consequence of the abort (peer closed first); a semantic failure (e.g. ProtocolMismatch) keeps its own code
                }
                if (on_fatal_) on_fatal_(e.what());
                if (!aborted_text.empty() && error_code_of(e) == ErrorCode::TransportError) detail::TransferWorkAccess::complete_error(detail::TransferWorkAccess::state_of(job.work), ErrorCode::Aborted, aborted_text);
                else detail::TransferWorkAccess::complete_error(detail::TransferWorkAccess::state_of(job.work), error_code_of(e), e.what());
            }
            catch (...)
            {
                unfinished_.fetch_sub(1, std::memory_order_acq_rel);
                if (on_fatal_) on_fatal_("unknown error in collective executor");
                detail::TransferWorkAccess::complete_error(
                    detail::TransferWorkAccess::state_of(job.work), ErrorCode::InternalError, "internal_error: unknown error in collective executor");
            }
            detail::tl_lat_parent_id = 0;
            lock.lock();
            running_ = false;
            lock.unlock();
            idle_cv_.notify_all();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    std::deque<CollectiveJob> queue_;
    bool stop_ = false;
    bool aborted_ = false;
    bool running_ = false;
    std::atomic<int> unfinished_{0};
    std::string abort_message_;
    std::function<void(const std::string &)> on_fatal_;
    std::thread thread_;
};

} // namespace

// ---------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------

bool Capabilities::supports_memory_kind(MemoryKind kind) const noexcept
{
    const auto remote_name = [&]() -> MemoryBackendKind {
        switch (kind)
        {
        case MemoryKind::Host: return MemoryBackendKind::Host;
        case MemoryKind::Cuda: return MemoryBackendKind::CudaPinned;
        case MemoryKind::MetalShared: return MemoryBackendKind::MetalShared;
        }
        return MemoryBackendKind::Host;
    }();
    for (auto k : negotiation_.common_memory_backends)
    {
        if (k == remote_name) return true;
    }
    // Host is always valid locally even if not explicitly negotiated
    // (every peer always advertises it -- peer_capabilities.hpp).
    return kind == MemoryKind::Host;
}

bool Capabilities::supports_collective_all_reduce(MemoryKind kind, DataType datatype, ReduceOp op) const noexcept
{
    // The N=2 engine only implements Sum. Host reduces every type reduction_supported() allows. A MetalShared reduce runs the
    // benchmark-side host loop (original four element types only); a Cuda reduce runs the device kernels, which cover every type.
    if (op != ReduceOp::Sum || !negotiation_.ok || !reduction_supported(datatype, op)) return false;
    switch (kind)
    {
    case MemoryKind::MetalShared:
        switch (datatype)
        {
        case DataType::Int32:
        case DataType::Int64:
        case DataType::Float32:
        case DataType::Float64:
            return true;
        default:
            return false; // the Metal-shared host loop only has the original four element types
        }
    case MemoryKind::Cuda:
        switch (datatype)
        {
        case DataType::Int32:
        case DataType::Int64:
        case DataType::Float32:
        case DataType::Float64:
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Float16:
        case DataType::BFloat16:
            return true;
        }
        return false;
    default:
        return true;
    }
}

bool Capabilities::supports_collective_broadcast(MemoryKind kind) const noexcept
{
    return negotiation_.ok && supports_memory_kind(kind);
}

bool Capabilities::supports_collective_all_gather(MemoryKind kind) const noexcept
{
    return negotiation_.ok && supports_memory_kind(kind);
}

// ---------------------------------------------------------------------
// Communicator::Impl
// ---------------------------------------------------------------------

struct Communicator::Impl
{
    // Declared first so it is destroyed last: provider resources (e.g. pinned staging) must outlive the worker
    // threads that use them. Guarded by registry_mutex() (make_provider holds it).
    std::map<MemoryKind, ProviderResourceSlot> provider_slots;

    std::size_t rank = 0;
    std::size_t world_size = 0;

    Capabilities caps;

    // Collective sequence: assigned by the executor thread, in the order collectives actually run (read/written only there).
    std::uint64_t next_sequence = 0;
    std::atomic<std::uint64_t> next_work_id{0};
    detail::Trace trace;
    detail::PlannerOverrides overrides;     // this rank's debug overrides (carried in descriptors)
    detail::PlannerThresholds thresholds;   // used by rank 0's planner
    std::function<void(const std::string &)> fatal_cb; // aborts this communicator (handed to the collectives' validators)

    // The N=2 specialised paths (all_reduce / broadcast / all_gather) talk to the one remote rank.
    detail::PeerChannel &sole_channel()
    {
        for (const auto &c : mesh->channels())
            if (c) return *c;
        throw Error(ErrorCode::InternalError, "internal_error: sole_channel() on a communicator without peers");
    }

    // Running -> AbortRequested -> Aborted. The first transition's reason wins. Lock order: nothing is held
    // across calls out of request_abort() (it only takes each component's own short mutex, one at a time).
    enum class State : int { Running = 0, AbortRequested = 1, Aborted = 2 };
    std::atomic<int> state{0};
    std::mutex reason_mutex;
    std::string reason;
    std::atomic<bool> failed{false};

    // Non-blocking, idempotent, safe from any thread including workers (never joins). `origin` is the rank that first aborted (this rank
    // unless the abort arrived from a peer); `raw` is its reason. The first caller wins and tells every peer before interrupting
    // anything local, so a rank blocked on an unrelated peer learns about the failure too (communicator-wide abort).
    void request_abort(const std::string &why, std::size_t origin, const std::string &raw)
    {
        int expected = static_cast<int>(State::Running);
        if (!state.compare_exchange_strong(expected, static_cast<int>(State::AbortRequested))) return;
        {
            std::lock_guard<std::mutex> lock(reason_mutex);
            reason = why;
        }
        failed.store(true, std::memory_order_release);
        trace.line("abort: " + why);
        // Close the collective queue before any I/O. Once the active job fails the worker takes the next queued job; broadcast_abort() below lets a peer react and close
        // its sockets, which on a busy or single CPU can fail the active job before this thread drains the queue, so a job that never started would run on a dead transport
        // and fail with a transport error instead of the abort. CollectiveExecutor::abort() does not wait for the active job.
        if (collective_executor) collective_executor->abort(why);
        if (mesh) mesh->mark_aborting(why);
        if (mesh) mesh->broadcast_abort(origin, raw);
        if (mesh) mesh->abort_transfers(why);
    }

    void request_abort(const std::string &why) { request_abort(why, rank, why); }

    std::string abort_message()
    {
        std::lock_guard<std::mutex> lock(reason_mutex);
        return "aborted: communicator aborted" + (reason.empty() ? "" : " (" + reason + ")");
    }

    void mark_failed(const std::string &why) { request_abort(why); }

    // P2P send()/recv() enqueue onto TensorCommWorker, which is
    // asynchronous: enqueue() returns immediately, and the actual
    // backend->stage_source_chunk()/commit_destination_chunk() (or
    // direct_source_data()/direct_destination_data()) calls happen later
    // on TensorCommWorker's own internal threads. The per-call
    // ExternalMemoryProvider (which owns the AsyncMemoryBackend those
    // calls run against) must therefore outlive completion, not just the
    // call to send()/recv() itself -- tracked here and pruned lazily
    // (amortized, no per-call thread, matching) rather than spawning a
    // dedicated thread per operation just to keep it alive.
    std::atomic<int> debug_fail_admission{0}; // test hook: the next N P2P admissions fail as if out of memory
    std::mutex outstanding_mutex;
    std::vector<std::pair<Work, std::shared_ptr<ExternalMemoryProvider>>> outstanding;

    // Pruning is amortized: completed entries are swept only when the list has doubled since the last sweep, so a burst of N submissions costs O(N)
    // in total (a sweep per call made it quadratic).
    std::size_t prune_threshold = 64;

    Work track(Work work, std::shared_ptr<ExternalMemoryProvider> provider)
    {
        std::lock_guard<std::mutex> lock(outstanding_mutex);
        if (outstanding.size() >= prune_threshold)
        {
            outstanding.erase(
                std::remove_if(
                    outstanding.begin(), outstanding.end(),
                    [](const auto &entry) { return entry.first.is_completed(); }),
                outstanding.end());
            prune_threshold = std::max<std::size_t>(64, outstanding.size() * 2);
        }
        outstanding.emplace_back(work, std::move(provider));
        return work;
    }

    // Declared last so they are destroyed first: the watcher and worker threads they own call back into the state above, so those
    // threads must be gone before any of it is destroyed. One PeerChannel (control + data connection, duplex worker) per remote
    // rank; no channel for world_size == 1.
    std::unique_ptr<detail::ConnectionManager> mesh;
    std::unique_ptr<CollectiveExecutor> collective_executor;
};

Communicator::Communicator() : impl_(std::make_unique<Impl>()) {}
Communicator::~Communicator()
{
    auto &impl = *impl_;
    const bool busy = (impl.mesh && impl.mesh->busy()) || (impl.collective_executor && impl.collective_executor->busy());
    if (busy)
    {
        // Outstanding operations on a (possibly silent) peer: abandon them instead of waiting for it forever.
        impl.request_abort("communicator destroyed with outstanding operations");
    }
    else
    {
        // Idle: nothing can fail later, but make late fatal-handler calls (during member destruction) no-ops, and tell the peers this is a
        // clean departure (their watchers must not treat the sockets closing as a failure).
        int expected = static_cast<int>(Impl::State::Running);
        if (impl.state.compare_exchange_strong(expected, static_cast<int>(Impl::State::Aborted)) && impl.mesh) impl.mesh->goodbye();
    }
    // Watcher threads call back into the state below (including the executor): join them before anything is destroyed.
    if (impl.mesh) impl.mesh->stop_watchers();
    // Members are destroyed executor -> mesh (watchers, workers, transports): queues are empty/failed, so each join is prompt.
}

void Communicator::abort(const std::string &reason)
{
    auto &impl = *impl_;
    impl.request_abort(reason.empty() ? "abort() called" : reason);
    // Wait (event-driven) until no local TBCCL thread can touch user buffers any more.
    impl.collective_executor->wait_idle();
    if (impl.mesh) impl.mesh->wait_idle();
    impl.state.store(static_cast<int>(Impl::State::Aborted));
}

bool Communicator::aborted() const noexcept
{
    return impl_->state.load() != static_cast<int>(Impl::State::Running);
}

std::string Communicator::abort_reason() const
{
    std::lock_guard<std::mutex> lock(impl_->reason_mutex);
    return impl_->reason;
}

// TBCCL_STAGED_CHUNK_BYTES is validated once, when the first Communicator is created (like the algorithm overrides), so a typo is an error rather than a silently unchunked run.
// Accepted: unset, empty or "0" (one chunk per transfer), or a decimal byte count of at least kMinStagedChunkBytes. Anything else is invalid_argument.
namespace
{
constexpr unsigned long long kMinStagedChunkBytes = 4096; // smaller chunks only multiply the per-chunk bookkeeping

std::size_t parse_staged_chunk_bytes()
{
    const char *text = std::getenv("TBCCL_STAGED_CHUNK_BYTES");
    if (!text || !*text) return 0;
    const std::string v(text);
    auto invalid = [&](const char *why) {
        return Error(ErrorCode::InvalidArgument, std::string("invalid_argument: TBCCL_STAGED_CHUNK_BYTES='") + v + "' " + why);
    };
    if (v.find_first_not_of("0123456789") != std::string::npos) throw invalid("is not a non-negative decimal byte count");
    errno = 0;
    const unsigned long long n = std::strtoull(v.c_str(), nullptr, 10);
    if (errno == ERANGE || n > std::numeric_limits<std::size_t>::max()) throw invalid("is too large");
    if (n != 0 && n < kMinStagedChunkBytes) throw invalid("is below the 4096 byte minimum (use 0 for one chunk per transfer)");
    return static_cast<std::size_t>(n);
}

std::size_t staged_chunk_setting()
{
    static const std::size_t configured = parse_staged_chunk_bytes(); // a failed initialisation is retried, so every creation reports the same error
    return configured;
}
} // namespace

std::unique_ptr<Communicator> Communicator::create(const CommunicatorOptions &options)
{
    const detail::ResolvedBootstrap boot = detail::resolve_bootstrap(options);

    auto comm = std::unique_ptr<Communicator>(new Communicator());
    auto &impl = *comm->impl_;
    impl.rank = boot.rank;
    impl.world_size = boot.world_size;

    impl.caps.local_ = local_capabilities();
    impl.mesh = detail::ConnectionManager::establish(
        boot, options.bootstrap_timeout, options.listeners.get(), impl.caps.local_, impl.caps.negotiation_);
    impl.caps.rank_capabilities_.resize(boot.world_size);
    for (std::size_t r = 0; r < boot.world_size; ++r)
    {
        const auto &channel = impl.mesh->channels()[r];
        impl.caps.rank_capabilities_[r] = channel ? channel->capabilities : impl.caps.local_;
    }
    impl.caps.remote_ = impl.caps.rank_capabilities_[boot.world_size == 1 ? 0 : (boot.rank == 0 ? 1 : 0)];

    impl.trace = detail::Trace(boot.communicator_id.prefix(), boot.rank, boot.world_size);
    impl.overrides = detail::planner_overrides_from_environment();
    staged_chunk_setting(); // reject an invalid TBCCL_STAGED_CHUNK_BYTES at creation
    impl.fatal_cb = [&impl](const std::string &why) { impl.request_abort(why); };
    impl.collective_executor = std::make_unique<CollectiveExecutor>();
    impl.mesh->set_fatal_handler([&impl](const std::string &m) { impl.request_abort(m); });
    impl.collective_executor->set_fatal_handler([&impl](const std::string &m) { impl.request_abort(m); });
    impl.mesh->start_watchers([&impl](std::size_t peer, std::size_t origin, const std::string &reason, bool peer_lost) {
        if (peer_lost)
            impl.request_abort("lost the control connection to rank " + std::to_string(peer) + " (" + reason + ")", impl.rank, "lost the control connection to rank " + std::to_string(peer));
        else
            impl.request_abort("rank " + std::to_string(origin) + " aborted the communicator: " + reason, origin, reason);
    });

    return comm;
}

std::size_t Communicator::rank() const noexcept { return impl_->rank; }
std::size_t Communicator::world_size() const noexcept { return impl_->world_size; }
const Capabilities &Communicator::capabilities() const noexcept { return impl_->caps; }
bool Communicator::failed() const noexcept { return impl_->failed.load(std::memory_order_relaxed); }

ProviderResourceSlot Communicator::provider_resources(MemoryKind kind) const
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = impl_->provider_slots.find(kind);
    return it == impl_->provider_slots.end() ? nullptr : it->second;
}

namespace
{
void require_not_failed(const Communicator &comm)
{
    if (comm.failed())
    {
        const std::string reason = comm.abort_reason();
        throw Error(ErrorCode::Aborted,
            "peer_failure: communicator is in a failed state: aborted" +
            (reason.empty() ? std::string() : " (" + reason + ")"));
    }
}
} // namespace

namespace
{

// Staged transfers (a backend that cannot be read or written by the transport directly, such as CUDA device memory) move their payload through host staging buffers.
// With the default (one chunk for the whole payload) the device-to-host copy, the network transfer and the host-to-device copy run one after the other.
// TBCCL_STAGED_CHUNK_BYTES=<n> (opt-in, unset or 0 = unchanged) splits a staged point-to-point transfer or an N=2 all_reduce leg into chunks of n bytes that pipeline across
// those three stages. Payloads of at most n bytes are still one chunk. Direct-access backends (Host, MetalShared) ignore it. The bytes on the wire are identical either way
// (the frame header travels once, then the raw payload stream), so the two ranks need not use the same value. Read once per process.
std::size_t staged_chunk_hint(const AsyncMemoryBackend &backend)
{
    return backend.supports_direct_transport_access() ? 0 : staged_chunk_setting();
}

// One hint serves both legs of the two-rank all_reduce (receive into `a`, send from `b`). A leg whose memory is directly accessible must stay on the direct path,
// so the setting applies only when neither backend is.
std::size_t staged_chunk_hint(const AsyncMemoryBackend &a, const AsyncMemoryBackend &b)
{
    return a.supports_direct_transport_access() || b.supports_direct_transport_access() ? 0 : staged_chunk_setting();
}

// P2P admission. Nothing here waits for transport progress, a peer, a lane slot or staging: the request is a lightweight descriptor that the peer's
// persistent progress thread picks up in FIFO order. An allocation failure while admitting is reported at once as ResourceExhausted; nothing was
// accepted, so the communicator stays usable. Any other failure after admission began poisons the communicator, as before. (A template only because
// Communicator::Impl is private to the class.)
template <typename ImplT>
Work post_p2p(
    ImplT &impl, TransferDirection direction, const BufferView &buffer, std::size_t count, DataType datatype, std::size_t peer,
    const ExecutionContext &context)
{
    if (detail::lat_on()) detail::tl_lat_submit_ns = detail::lat_now_ns();
    detail::PeerChannel &channel = impl.mesh->channel(peer); // throws invalid_argument for self / out of range
    validate_buffer_view(buffer, count, datatype);
    const auto exhausted = [&] {
        return Error(ErrorCode::ResourceExhausted, std::string("resource_exhausted: could not admit the ") + (direction == TransferDirection::Send ? "send" : "recv") + " (out of memory)");
    };
    std::shared_ptr<ExternalMemoryProvider> provider;
    TransferRequest request;
    try
    {
        provider = make_provider(buffer, context, impl.provider_slots); // a rejected kind throws here, before anything is accepted: no poisoning
        request.direction = direction;
        request.backend = &provider->primary_backend();
        request.transport = channel.data.get();
        request.total_bytes = buffer.bytes;
        request.chunk_hint = staged_chunk_hint(*request.backend);
        request.framed = true; // a receive of a different size fails with protocol_mismatch instead of hanging
        if (impl.debug_fail_admission.load(std::memory_order_relaxed) > 0 && impl.debug_fail_admission.fetch_sub(1) > 0) throw std::bad_alloc();
    }
    catch (const std::bad_alloc &)
    {
        throw exhausted();
    }
    try
    {
        Work work = channel.worker->enqueue(request);
        return impl.track(work, std::move(provider));
    }
    catch (const std::bad_alloc &)
    {
        throw exhausted();
    }
    catch (...)
    {
        impl.mark_failed(direction == TransferDirection::Send ? "send enqueue failed" : "recv enqueue failed");
        throw;
    }
}

} // namespace

Work Communicator::send(
    const BufferView &buffer, std::size_t count, DataType datatype, std::size_t peer, const ExecutionContext &context)
{
    require_not_failed(*this);
    return post_p2p(*impl_, TransferDirection::Send, buffer, count, datatype, peer, context);
}

Work Communicator::recv(
    const BufferView &buffer, std::size_t count, DataType datatype, std::size_t peer, const ExecutionContext &context)
{
    require_not_failed(*this);
    return post_p2p(*impl_, TransferDirection::Recv, buffer, count, datatype, peer, context);
}

namespace
{

// A provider for one buffer, or the reason it could not be made. For world_size > 2 a rank-local failure (an unregistered memory
// kind, say) must not become a local throw that strands the other ranks waiting for this one: it is reported in the collective
// descriptor and every rank fails the collective together. For world_size <= 2 it throws at the call, as before.
struct LocalProvider
{
    std::shared_ptr<ExternalMemoryProvider> provider;
    std::string error;
};

LocalProvider make_local_provider(
    const BufferView &buffer, const ExecutionContext &context, std::map<MemoryKind, ProviderResourceSlot> &slots, bool defer_errors)
{
    LocalProvider out;
    try
    {
        out.provider = make_provider(buffer, context, slots);
    }
    catch (const std::exception &e)
    {
        if (!defer_errors) throw;
        out.error = e.what();
    }
    return out;
}

detail::CollectiveDescriptor describe(
    detail::CollectiveKind kind, std::size_t rank, std::size_t root, MemoryKind memory, std::size_t count, DataType datatype, ReduceOp op,
    std::size_t bytes, const std::string &local_error, detail::CommAlgorithm forced = detail::CommAlgorithm::Unspecified)
{
    detail::CollectiveDescriptor d;
    d.forced_algorithm = static_cast<std::uint32_t>(forced);
    d.kind = kind;
    d.rank = static_cast<std::uint32_t>(rank);
    d.root = static_cast<std::uint32_t>(root);
    d.memory_kind = memory;
    d.count = count;
    d.datatype = datatype;
    d.reduce_op = op;
    d.bytes = bytes;
    if (!local_error.empty())
    {
        d.local_status = 1;
        d.note = local_error;
    }
    return d;
}

std::atomic<std::uint64_t> g_transfer_id{std::uint64_t{1} << 40};

void run_transfer(
    TensorCommWorker &worker,
    Transport &transport,
    AsyncMemoryBackend &backend,
    TransferDirection direction,
    std::size_t bytes,
    const char *what)
{
    TransferRequest request;
    request.transfer_id = g_transfer_id.fetch_add(1, std::memory_order_relaxed);
    request.direction = direction;
    request.backend = &backend;
    request.transport = &transport;
    request.total_bytes = bytes;
    request.chunk_hint = staged_chunk_hint(backend);
    request.shared_lane = true; // N=2 specialised path: strictly sequential, single FIFO as before duplex lanes
    TransferWork work = worker.enqueue(request);
    const std::uint64_t child = detail::tl_lat_last_enqueued_id;
    detail::lat_event(detail::kLatCollChildPosted, detail::tl_lat_parent_id, static_cast<std::uint32_t>(child));
    work.wait();
    detail::lat_event(detail::kLatCollChildObserved, detail::tl_lat_parent_id, static_cast<std::uint32_t>(child));
    if (work.has_error()) throw Error(work.error_code(), std::string("transport_error: ") + what + ": " + work.error());
}

// The N=2 all_gather exchange: both directions are in flight at once, each on its own duplex lane of the collective worker (shared_lane stays false, so a receive
// that is waiting for the peer never sits in front of the send). The rank-dependent send-then-receive order this replaces made every rank's second leg wait for its
// first. Admission never waits (the collective worker's queues are unbounded), so posting both before waiting for either cannot deadlock for any payload size; the
// peer's matching transfers run on its own independent lanes. Each leg keeps the staged-chunk hint of its own backend, exactly as run_transfer does.
//
// Failure policy: the job never returns while a child can still touch the providers' memory.
//  * If the second enqueue throws, the first is already in flight and the peer will wait for the missing leg, so the communicator is aborted (`fatal`, idempotent) which
//    interrupts the admitted child; it is waited for, then the admission error is rethrown.
//  * If a child fails, `fatal` is invoked at once (the failed exchange has desynchronised the collective sequence anyway) so the other child is interrupted instead of
//    waiting for a peer that may never answer; then both are waited for. The send is polled while the receive blocks, so a send failure is noticed without waiting for the receive.
//  * The reported error is the first cause observed; an Aborted that is only the consequence of the abort we triggered never replaces it.
void run_exchange(
    TensorCommWorker &worker,
    Transport &transport,
    AsyncMemoryBackend &send_backend,
    AsyncMemoryBackend &recv_backend,
    std::size_t bytes,
    const char *what,
    const std::function<void(const std::string &)> &fatal)
{
    const auto make_request = [&](TransferDirection direction, AsyncMemoryBackend &backend) {
        TransferRequest request;
        request.transfer_id = g_transfer_id.fetch_add(1, std::memory_order_relaxed);
        request.direction = direction;
        request.backend = &backend;
        request.transport = &transport;
        request.total_bytes = bytes;
        request.chunk_hint = staged_chunk_hint(backend);
        request.shared_lane = false; // duplex: the send and the receive must not share a lane
        return request;
    };
    TransferWork send_work = worker.enqueue(make_request(TransferDirection::Send, send_backend));
    const std::uint64_t send_child = detail::tl_lat_last_enqueued_id;
    std::optional<TransferWork> recv_work;
    std::uint64_t recv_child = 0;
    try
    {
        if (detail::g_n2_exchange_fail_second_enqueue.load(std::memory_order_relaxed) > 0 && detail::g_n2_exchange_fail_second_enqueue.fetch_sub(1) > 0) throw std::bad_alloc(); // test hook
        recv_work.emplace(worker.enqueue(make_request(TransferDirection::Recv, recv_backend)));
        recv_child = detail::tl_lat_last_enqueued_id;
    }
    catch (...)
    {
        fatal(std::string(what) + ": could not post the receive");
        send_work.wait(); // interrupted by the abort above; the providers must outlive it
        throw;
    }
    detail::lat_event(detail::kLatCollChildPosted, detail::tl_lat_parent_id, static_cast<std::uint32_t>(send_child));
    detail::lat_event(detail::kLatCollChildPosted, detail::tl_lat_parent_id, static_cast<std::uint32_t>(recv_child));

    bool aborted_by_us = false;
    const auto trigger = [&](const char *which, const TransferWork &failed) {
        if (aborted_by_us) return;
        aborted_by_us = true;
        fatal(std::string("transport_error: ") + what + " " + which + ": " + failed.error()); // the same text the failed job would have reported to the abort
    };
    // Wait for the receive, noticing a send failure meanwhile (wait_for returns as soon as the receive completes: no added latency).
    while (!recv_work->wait_for(std::chrono::milliseconds(20)))
    {
        if (send_work.is_completed() && send_work.has_error())
        {
            trigger("send", send_work);
            break;
        }
    }
    if (recv_work->is_completed() && recv_work->has_error()) trigger("recv", *recv_work);
    recv_work->wait();
    send_work.wait();
    detail::lat_event(detail::kLatCollChildObserved, detail::tl_lat_parent_id, static_cast<std::uint32_t>(send_child));
    detail::lat_event(detail::kLatCollChildObserved, detail::tl_lat_parent_id, static_cast<std::uint32_t>(recv_child));

    const bool send_failed = send_work.has_error(), recv_failed = recv_work->has_error();
    if (!send_failed && !recv_failed) return;
    // Cause over consequence: a failure that is not Aborted outranks one that is; between equals the one we saw first (the receive: a lost peer shows there).
    const bool use_send = send_failed && (!recv_failed || (recv_work->error_code() == ErrorCode::Aborted && send_work.error_code() != ErrorCode::Aborted));
    if (use_send) throw Error(send_work.error_code(), std::string("transport_error: ") + what + " send: " + send_work.error());
    throw Error(recv_work->error_code(), std::string("transport_error: ") + what + " recv: " + recv_work->error());
}

} // namespace

namespace detail
{
void run_n2_exchange(
    TensorCommWorker &worker, Transport &transport, AsyncMemoryBackend &send_backend, AsyncMemoryBackend &recv_backend, std::size_t bytes, const char *what,
    const std::function<void(const std::string &)> &fatal)
{
    run_exchange(worker, transport, send_backend, recv_backend, bytes, what, fatal);
}
} // namespace detail

Work Communicator::all_reduce(
    const BufferView &send_buf,
    const BufferView &recv_buf,
    std::size_t count,
    DataType datatype,
    ReduceOp op,
    const ExecutionContext &context)
{
    require_not_failed(*this);

    const std::size_t world = impl_->world_size;
    if (op != ReduceOp::Sum)
    {
        throw Error(ErrorCode::Unsupported, "unsupported: all_reduce only supports ReduceOp::Sum");
    }
    validate_reduction(datatype, op);
    if (world > 2 && (datatype == DataType::Float16 || datatype == DataType::BFloat16))
    {
        throw Error(ErrorCode::Unsupported, 
            std::string("unsupported: ") + datatype_label(datatype) + " SUM across " + std::to_string(world) +
            " ranks: Float16/BFloat16 N>2 reduction semantics are not defined (only the two-operand case is specified)");
    }
    const bool defer = world > 2; // see LocalProvider
    if (!defer && !impl_->caps.supports_collective_all_reduce(recv_buf.memory_kind, datatype, op))
    {
        throw Error(ErrorCode::Unsupported, 
            std::string("unsupported: all_reduce of dtype=") + datatype_label(datatype) + " is not available for this memory kind");
    }
    validate_buffer_view(send_buf, count, datatype);
    validate_buffer_view(recv_buf, count, datatype);

    const std::size_t rank = impl_->rank;
    const std::size_t total_bytes = recv_buf.bytes;

    // Keep the provider alive for the duration of the collective by capturing it (shared_ptr) into the executor job's lambda.
    LocalProvider local = make_local_provider(recv_buf, context, impl_->provider_slots, defer);
    std::string local_error = local.error;
    if (local_error.empty() && defer && !impl_->caps.supports_collective_all_reduce(recv_buf.memory_kind, datatype, op))
    {
        local_error = std::string("dtype=") + datatype_label(datatype) + " is not available for this memory kind";
    }

    if (send_buf.data != recv_buf.data && count > 0 && local_error.empty())
    {
        // Out-of-place: stage send_buf's content into recv_buf's location before the collective, so both root and non-root logic can
        // uniformly treat recv_buf as "holds the local input, ends up holding the result". Host-only memcpy is correct here because
        // MemoryKind::MetalShared's data is CPU-visible by contract and MemoryKind::Cuda providers are expected to perform
        // device-side copies before returning from their own factory if they need this -- the framework-independent runtime work
        // does not implement an out-of-place external-CUDA AllReduce; call with send_buf.data == recv_buf.data (in-place) for Cuda
        // buffers.
        if (recv_buf.memory_kind == MemoryKind::Cuda)
        {
            throw Error(ErrorCode::Unsupported, 
                "unsupported: out-of-place all_reduce (send_buf.data != recv_buf.data) is not supported for MemoryKind::Cuda; pass the same BufferView for send_buf and recv_buf");
        }
        std::memcpy(recv_buf.data, send_buf.data, recv_buf.bytes);
    }

    Impl *impl = impl_.get();
    const std::uint64_t work_id = impl->next_work_id.fetch_add(1);
    const MemoryKind memory = recv_buf.memory_kind;
    auto provider = local.provider;

    auto run = [impl, provider, local_error, rank, world, total_bytes, count, datatype, op, work_id, memory]() {
        const std::uint64_t sequence = impl->next_sequence++;
        const detail::CollectiveRun r{rank, world, impl->mesh.get(), sequence, work_id, &impl->trace, &impl->thresholds, &impl->fatal_cb};
        if (world == 2)
        {
            // The N=2 fast path: no descriptor exchange, the specialised heterogeneous engine unchanged (Part 55).
            constexpr std::size_t kRoot = 0; // Fixed internal policy, never exposed.
            detail::PeerChannel &sole = impl->sole_channel();
            AsyncMemoryBackend &primary = provider->primary_backend();
            if (impl->trace.on()) impl->trace.line("work=" + std::to_string(work_id) + " #" + std::to_string(sequence) + " all_reduce n2 fast path bytes=" + std::to_string(total_bytes) +
                             " dtype=" + datatype_label(datatype));
            if (rank == kRoot)
            {
                AsyncMemoryBackend &scratch = provider->scratch_backend();
                LocalReduceBackend &reduce = provider->reduce_backend();
                n2_all_reduce_tensor(*sole.coll_data, *sole.coll_worker, scratch, primary, &reduce, rank, kRoot, total_bytes, staged_chunk_hint(scratch, primary), count, datatype);
            }
            else
            {
                n2_all_reduce_tensor(*sole.coll_data, *sole.coll_worker, primary, primary, nullptr, rank, kRoot, total_bytes, staged_chunk_hint(primary), count, datatype);
            }
            return;
        }
        if (world > 1)
        {
            const auto algorithm = detail::run_descriptor_exchange(
                r, describe(detail::CollectiveKind::AllReduce, rank, 0, memory, count, datatype, op, total_bytes, local_error, impl->overrides.all_reduce));
            detail::run_all_reduce(r, algorithm, *provider, total_bytes, count, datatype);
        }
        // world_size 1: the (already staged) local input is the result.
    };

    try
    {
        return impl_->collective_executor->submit(std::move(run));
    }
    catch (...)
    {
        impl_->mark_failed("all_reduce submit failed");
        throw;
    }
}

Work Communicator::broadcast(const BufferView &buffer, std::size_t root, const ExecutionContext &context)
{
    require_not_failed(*this);
    const std::size_t world = impl_->world_size;
    if (root >= world) throw Error(ErrorCode::InvalidArgument, "invalid_argument: broadcast root out of range");
    if (buffer.bytes > 0 && buffer.data == nullptr)
        throw Error(ErrorCode::InvalidArgument, "invalid_argument: broadcast buffer.data is null for a non-empty buffer");

    const std::size_t rank = impl_->rank;
    const std::size_t bytes = buffer.bytes;
    LocalProvider local;
    if (bytes > 0) local = make_local_provider(buffer, context, impl_->provider_slots, /*defer_errors=*/world > 2);

    Impl *impl = impl_.get();
    const std::uint64_t work_id = impl->next_work_id.fetch_add(1);
    const MemoryKind memory = buffer.memory_kind;
    auto provider = local.provider;
    const std::string local_error = local.error;
    auto run = [impl, provider, local_error, rank, world, root, bytes, work_id, memory]() {
        const std::uint64_t sequence = impl->next_sequence++;
        const detail::CollectiveRun r{rank, world, impl->mesh.get(), sequence, work_id, &impl->trace, &impl->thresholds, &impl->fatal_cb};
        if (world == 2)
        {
            // N=2 fast path: one transfer, no descriptor exchange.
            if (bytes == 0) return;
            detail::PeerChannel &sole = impl->sole_channel();
            if (impl->trace.on()) impl->trace.line("work=" + std::to_string(work_id) + " #" + std::to_string(sequence) + " broadcast n2 fast path bytes=" + std::to_string(bytes));
            run_transfer(*sole.coll_worker, *sole.coll_data, provider->primary_backend(), rank == root ? TransferDirection::Send : TransferDirection::Recv, bytes, "broadcast");
            return;
        }
        if (world > 1)
        {
            const auto algorithm = detail::run_descriptor_exchange(
                r, describe(detail::CollectiveKind::Broadcast, rank, root, memory, 0, DataType::UInt8, ReduceOp::Sum, bytes, local_error, impl->overrides.broadcast));
            detail::run_broadcast(r, algorithm, provider.get(), bytes, root);
        }
    };
    try
    {
        return impl_->collective_executor->submit(std::move(run));
    }
    catch (...)
    {
        impl_->mark_failed("broadcast submit failed");
        throw;
    }
}

Work Communicator::all_gather(
    const BufferView &input, const std::vector<BufferView> &outputs, const ExecutionContext &context)
{
    require_not_failed(*this);
    const std::size_t world = impl_->world_size;
    if (outputs.size() != world)
        throw Error(ErrorCode::InvalidArgument, "invalid_argument: all_gather needs exactly world_size output buffers");
    const std::size_t bytes = input.bytes;
    for (const auto &out : outputs)
    {
        if (out.bytes != bytes) throw Error(ErrorCode::InvalidArgument, "invalid_argument: all_gather output size differs from input size");
    }
    if (bytes > 0 && input.data == nullptr) throw Error(ErrorCode::InvalidArgument, "invalid_argument: all_gather input.data is null");
    for (const auto &out : outputs)
    {
        if (bytes > 0 && out.data == nullptr) throw Error(ErrorCode::InvalidArgument, "invalid_argument: all_gather output.data is null");
    }

    const std::size_t rank = impl_->rank;
    const bool defer = world > 2;
    std::shared_ptr<ExternalMemoryProvider> in_provider;
    std::vector<std::shared_ptr<ExternalMemoryProvider>> out_providers(world);
    std::string local_error;
    if (bytes > 0)
    {
        auto take = [&](const BufferView &b, std::shared_ptr<ExternalMemoryProvider> &slot) {
            LocalProvider p = make_local_provider(b, context, impl_->provider_slots, defer);
            if (local_error.empty()) local_error = p.error;
            slot = p.provider;
        };
        take(input, in_provider);
        for (std::size_t r = 0; r < world; ++r)
        {
            // outputs[rank] <- input is a local copy through the providers; skipped if they alias.
            if (r == rank && outputs[r].data == input.data) continue;
            take(outputs[r], out_providers[r]);
        }
    }

    Impl *impl = impl_.get();
    const std::uint64_t work_id = impl->next_work_id.fetch_add(1);
    const MemoryKind memory = input.memory_kind;
    auto run = [impl, in_provider, out_providers, local_error, rank, world, bytes, work_id, memory]() mutable {
        const std::uint64_t sequence = impl->next_sequence++;
        const detail::CollectiveRun r{rank, world, impl->mesh.get(), sequence, work_id, &impl->trace, &impl->thresholds, &impl->fatal_cb};
        if (world == 2)
        {
            // N=2 fast path: no descriptor exchange; the send and the receive are exchanged concurrently (see run_exchange).
            if (bytes == 0) return;
            detail::PeerChannel &sole = impl->sole_channel();
            const std::size_t peer = sole.peer_rank;
            if (out_providers[rank]) detail::copy_through_providers(*in_provider, *out_providers[rank], bytes);
            if (impl->trace.on()) impl->trace.line("work=" + std::to_string(work_id) + " #" + std::to_string(sequence) + " all_gather n2 fast path bytes=" + std::to_string(bytes));
            run_exchange(*sole.coll_worker, *sole.coll_data, in_provider->primary_backend(), out_providers[peer]->primary_backend(), bytes, "all_gather", impl->fatal_cb);
            return;
        }
        if (world > 1)
        {
            const auto algorithm = detail::run_descriptor_exchange(
                r, describe(detail::CollectiveKind::AllGather, rank, 0, memory, 0, DataType::UInt8, ReduceOp::Sum, bytes, local_error, impl->overrides.all_gather));
            detail::run_all_gather(r, algorithm, in_provider.get(), out_providers, bytes);
        }
        else
        {
            detail::reference_all_gather(r, in_provider.get(), out_providers, bytes); // world_size 1: the local copy
        }
    };
    try
    {
        return impl_->collective_executor->submit(std::move(run));
    }
    catch (...)
    {
        impl_->mark_failed("all_gather submit failed");
        throw;
    }
}

Work Communicator::barrier()
{
    require_not_failed(*this);
    Impl *impl = impl_.get();
    const std::size_t rank = impl->rank, world = impl->world_size;
    const std::uint64_t work_id = impl->next_work_id.fetch_add(1);
    auto run = [impl, rank, world, work_id]() {
        const std::uint64_t sequence = impl->next_sequence++;
        const detail::CollectiveRun r{rank, world, impl->mesh.get(), sequence, work_id, &impl->trace, &impl->thresholds, &impl->fatal_cb};
        if (world > 1)
        {
            // The barrier's algorithm is decided locally (it has no payload and needs no verdict): the agreed-upon default, or this rank's override. Ranks whose
            // overrides differ are caught by the descriptor validation (the override is part of the descriptor).
            const detail::CommAlgorithm forced = impl->overrides.barrier;
            detail::CommAlgorithm algorithm;
            try
            {
                algorithm = detail::plan_collective(detail::CollectiveKind::Barrier, world, 0, forced, impl->thresholds).algorithm;
            }
            catch (const std::runtime_error &e)
            {
                throw detail::CollectiveRejected(e.what()); // a bad override: this Work fails, nothing was communicated
            }
            const auto mine = describe(detail::CollectiveKind::Barrier, rank, 0, MemoryKind::Host, 0, DataType::UInt8, ReduceOp::Sum, 0, std::string(), forced);
            if (algorithm == detail::CommAlgorithm::Dissemination)
            {
                detail::run_dissemination_barrier(r, mine);
            }
            else
            {
                algorithm = detail::run_descriptor_exchange(r, mine);
                detail::run_barrier(r, algorithm);
            }
        }
    };
    try
    {
        return impl_->collective_executor->submit(std::move(run));
    }
    catch (...)
    {
        impl_->mark_failed("barrier submit failed");
        throw;
    }
}

namespace detail
{
struct CommunicatorAccess
{
    static std::vector<std::size_t> connected_data_peers(const Communicator &comm) { return comm.impl_->mesh->connected_data_peers(); }
    static void fail_next_admissions(const Communicator &comm, int n) { comm.impl_->debug_fail_admission.store(n); }
    static void pause_progress(const Communicator &comm, bool paused) { comm.impl_->mesh->set_progress_paused(paused); }
};

std::vector<std::size_t> debug_connected_data_peers(const Communicator &comm) { return CommunicatorAccess::connected_data_peers(comm); }
void debug_fail_next_admissions(const Communicator &comm, int count) { CommunicatorAccess::fail_next_admissions(comm, count); }
void debug_set_progress_paused(const Communicator &comm, bool paused) { CommunicatorAccess::pause_progress(comm, paused); }
} // namespace detail

} // namespace tbccl
