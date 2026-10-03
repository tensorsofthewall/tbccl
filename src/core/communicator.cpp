#include <tbccl/communicator.hpp>

#include <tbccl/collectives.hpp>
#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/tcp_world.hpp>
#include <tbccl/transport.hpp>

#include "reduction_internal.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
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
// tbccl (Part AE: public installed headers -- and, by the same
// reasoning, core's own internal implementation -- must not depend on
// benchmarks/, which is structurally built ON TOP of core, never the
// reverse). Both pieces are small and proven; this is a deliberate,
// documented duplication, not a design gap.
// ---------------------------------------------------------------------

class HostPointerAsyncBackend final : public AsyncMemoryBackend
{
public:
    HostPointerAsyncBackend(void *buffer, std::size_t capacity)
        : buffer_(static_cast<std::byte *>(buffer)), capacity_(capacity)
    {
    }

    void stage_source_chunk(const Chunk &chunk, void *staging) override
    {
        std::memcpy(staging, buffer_ + chunk.offset, chunk.size);
    }

    void commit_destination_chunk(const Chunk &chunk, const void *staging) override
    {
        std::memcpy(buffer_ + chunk.offset, staging, chunk.size);
    }

    bool supports_direct_transport_access() const noexcept override { return true; }
    const void *direct_source_data() const noexcept override { return buffer_; }
    void *direct_destination_data() noexcept override { return buffer_; }

    void *data() const noexcept { return buffer_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    std::byte *buffer_;
    std::size_t capacity_;
};

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
            scratch_ = std::make_unique<HostPointerAsyncBackend>(scratch_storage_.data(), bytes_);
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

private:
    HostPointerAsyncBackend primary_;
    std::size_t bytes_;
    std::vector<std::byte> scratch_storage_;
    std::unique_ptr<HostPointerAsyncBackend> scratch_;
    std::unique_ptr<HostPointerReduceBackend> reduce_;
};

// ---------------------------------------------------------------------
// Provider factory registry (Part P/N extension point).
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
        throw std::runtime_error(
            "unsupported: no memory provider registered for kind=" + memory_kind_name(buffer.memory_kind) +
            " (call tbccl::register_memory_provider_factory() first)");
    }
    return it->second(buffer, context, slots[buffer.memory_kind]);
}

// ---------------------------------------------------------------------
// Collective executor: a generalized, promoted BucketAllReduceWorker
// (Phase 39) -- one persistent thread driving n2_all_reduce_tensor()
// jobs one at a time (matching that function's own "at most one
// AllReduce-related transfer in flight" contract), returning a Work per
// job immediately. Unlike BucketAllReduceWorker, a job failure does NOT
// poison all subsequently queued jobs -- each job's failure is isolated
// to its own Work (Part Q: communicator-owned Work semantics should not
// silently wedge unrelated future operations). A communicator-level
// transport/protocol failure is still tracked separately (failed_) and
// does deliberately fail all future operations (Part BJ).
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

    // Phase 45: terminal and idempotent. Queued jobs never started, so they fail without touching user memory; the
    // active job (if any) unwinds when the Transport is interrupted and its Work fails after it returns.
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
        for (auto &job : doomed)
        {
            detail::TransferWorkAccess::complete_error(detail::TransferWorkAccess::state_of(job.work), message);
        }
    }

    bool busy() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return running_ || !queue_.empty();
    }

    void set_fatal_handler(std::function<void(const std::string &)> handler) { on_fatal_ = std::move(handler); }

    void wait_idle()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_cv_.wait(lock, [&] { return !running_ && queue_.empty(); });
    }

    Work submit(std::function<void()> run)
    {
        Work work = detail::TransferWorkAccess::make();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (aborted_) throw std::runtime_error(abort_message_);
            queue_.push_back(CollectiveJob{std::move(run), work});
        }
        cv_.notify_all();
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
            try
            {
                job.run();
                detail::TransferWorkAccess::complete_ok(detail::TransferWorkAccess::state_of(job.work));
            }
            catch (const std::exception &e)
            {
                // A job fails only after protocol participation began (arguments were validated before
                // submission): the collective sequence is no longer trustworthy, so poison the communicator.
                if (on_fatal_) on_fatal_(e.what());
                detail::TransferWorkAccess::complete_error(detail::TransferWorkAccess::state_of(job.work), e.what());
            }
            catch (...)
            {
                if (on_fatal_) on_fatal_("unknown error in collective executor");
                detail::TransferWorkAccess::complete_error(
                    detail::TransferWorkAccess::state_of(job.work), "unknown error in collective executor");
            }
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
    // benchmark-side host loop and a Cuda reduce runs the device kernels; each has only the element types listed here.
    if (op != ReduceOp::Sum || !negotiation_.ok || !reduction_supported(datatype, op)) return false;
    switch (kind)
    {
    case MemoryKind::MetalShared:
    case MemoryKind::Cuda:
        switch (datatype)
        {
        case DataType::Int32:
        case DataType::Int64:
        case DataType::Float32:
        case DataType::Float64:
            return true;
        default:
            return false; // device/Metal arithmetic for the newer types is added with their backends
        }
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
    std::size_t other_peer = 0; // the only valid P2P peer index (world_size==2)

    std::unique_ptr<TcpTransport> transport;
    std::unique_ptr<TensorCommWorker> comm_worker;
    std::unique_ptr<CollectiveExecutor> collective_executor;
    Capabilities caps;

    // Phase 45: Running -> AbortRequested -> Aborted. The first transition's reason wins. Lock order: nothing is
    // held across calls out of request_abort() (it only takes each component's own short mutex, one at a time).
    enum class State : int { Running = 0, AbortRequested = 1, Aborted = 2 };
    std::atomic<int> state{0};
    std::mutex reason_mutex;
    std::string reason;
    std::atomic<bool> failed{false};

    // Non-blocking, idempotent, safe from any thread including workers (never joins).
    void request_abort(const std::string &why)
    {
        int expected = static_cast<int>(State::Running);
        if (!state.compare_exchange_strong(expected, static_cast<int>(State::AbortRequested))) return;
        {
            std::lock_guard<std::mutex> lock(reason_mutex);
            reason = why;
        }
        failed.store(true, std::memory_order_release);
        // Interrupt first so the active transfer unwinds; then reject/drain queues.
        if (transport) transport->abort(why);
        if (comm_worker) comm_worker->abort(why);
        if (collective_executor) collective_executor->abort(why);
    }

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
    // (amortized, no per-call thread, matching Part BB/BC) rather than
    // spawning a dedicated thread per operation just to keep it alive.
    std::mutex outstanding_mutex;
    std::vector<std::pair<Work, std::shared_ptr<ExternalMemoryProvider>>> outstanding;

    Work track(Work work, std::shared_ptr<ExternalMemoryProvider> provider)
    {
        std::lock_guard<std::mutex> lock(outstanding_mutex);
        outstanding.erase(
            std::remove_if(
                outstanding.begin(), outstanding.end(),
                [](const auto &entry) { return entry.first.is_completed(); }),
            outstanding.end());
        outstanding.emplace_back(work, std::move(provider));
        return work;
    }
};

Communicator::Communicator() : impl_(std::make_unique<Impl>()) {}
Communicator::~Communicator()
{
    auto &impl = *impl_;
    const bool busy = (impl.comm_worker && impl.comm_worker->busy()) ||
                      (impl.collective_executor && impl.collective_executor->busy());
    if (busy)
    {
        // Outstanding operations on a (possibly silent) peer: abandon them instead of waiting for it forever.
        impl.request_abort("communicator destroyed with outstanding operations");
    }
    else
    {
        // Idle: nothing can fail later, but make late fatal-handler calls (during member destruction) no-ops.
        int expected = static_cast<int>(Impl::State::Running);
        impl.state.compare_exchange_strong(expected, static_cast<int>(Impl::State::Aborted));
    }
    // Members are destroyed executor -> worker -> transport: queues are empty/failed, so each join is prompt.
}

void Communicator::abort(const std::string &reason)
{
    auto &impl = *impl_;
    impl.request_abort(reason.empty() ? "abort() called" : reason);
    // Wait (event-driven) until no local TBCCL thread can touch user buffers any more.
    impl.collective_executor->wait_idle();
    impl.comm_worker->wait_idle();
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

std::unique_ptr<Communicator> Communicator::create(const CommunicatorOptions &options)
{
    if (options.peers.size() != 2)
    {
        throw std::runtime_error(
            "unsupported: Communicator::create() requires exactly 2 peers this phase (world_size=" +
            std::to_string(options.peers.size()) + "); see docs/framework_integration_architecture.md");
    }
    if (options.rank >= options.peers.size())
    {
        throw std::runtime_error("invalid_argument: options.rank out of range for peers.size()");
    }

    auto comm = std::unique_ptr<Communicator>(new Communicator());
    auto &impl = *comm->impl_;
    impl.rank = options.rank;
    impl.world_size = options.peers.size();
    impl.other_peer = 1 - options.rank;

    // Readiness barrier via TcpWorld, then a separate data-path
    // connection -- the exact pattern established in Phase 38 to avoid
    // the sleep-based startup race, reused unchanged for every benchmark
    // since (tbccl_hetero_allreduce_bench.cpp, tbccl_bucketed_allreduce_bench.cpp).
    TcpWorldOptions world_opts;
    world_opts.rank = options.rank;
    world_opts.bootstrap_timeout = options.bootstrap_timeout;
    for (const auto &p : options.peers) world_opts.peers.push_back({p.host, p.port});
    auto world = create_tcp_world(world_opts);

    std::unique_ptr<Connection> connection;
    std::unique_ptr<Listener> listener;
    const auto data_port = static_cast<std::uint16_t>(options.peers[0].port + 1000);
    if (options.rank == 0) listener = tcp_listen(options.peers[0].host, data_port, {});
    barrier(*world);
    if (options.rank == 0) connection = listener->accept();
    else connection = tcp_connect(options.peers[0].host, data_port, {});

    impl.caps.local_ = local_capabilities();
    impl.caps.remote_ = exchange_capabilities(*connection, impl.caps.local_);
    impl.caps.negotiation_ = negotiate(impl.caps.local_, impl.caps.remote_);
    if (!impl.caps.negotiation_.ok)
    {
        throw std::runtime_error("transport_error: capability negotiation failed: " + impl.caps.negotiation_.failure_reason);
    }

    impl.transport = std::make_unique<TcpTransport>(std::move(connection));
    impl.comm_worker = std::make_unique<TensorCommWorker>(/*pipeline_depth=*/2, /*queue_depth=*/8);
    impl.collective_executor = std::make_unique<CollectiveExecutor>();
    impl.comm_worker->set_fatal_handler([&impl](const std::string &m) { impl.request_abort(m); });
    impl.collective_executor->set_fatal_handler([&impl](const std::string &m) { impl.request_abort(m); });

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
        throw std::runtime_error(
            "peer_failure: communicator is in a failed state: aborted" +
            (reason.empty() ? std::string() : " (" + reason + ")"));
    }
}
} // namespace

Work Communicator::send(
    const BufferView &buffer, std::size_t count, DataType datatype, std::size_t peer, const ExecutionContext &context)
{
    require_not_failed(*this);
    if (peer != impl_->other_peer)
    {
        throw std::runtime_error("invalid_argument: peer must be this communicator's single other rank");
    }
    validate_buffer_view(buffer, count, datatype);

    std::shared_ptr<ExternalMemoryProvider> provider = make_provider(buffer, context, impl_->provider_slots);
    auto &backend = provider->primary_backend();

    TransferRequest request;
    request.direction = TransferDirection::Send;
    request.backend = &backend;
    request.transport = impl_->transport.get();
    request.total_bytes = buffer.bytes;
    request.chunk_hint = 0;

    try
    {
        Work work = impl_->comm_worker->enqueue(request);
        return impl_->track(work, std::move(provider));
    }
    catch (...)
    {
        impl_->mark_failed("send enqueue failed");
        throw;
    }
}

Work Communicator::recv(
    const BufferView &buffer, std::size_t count, DataType datatype, std::size_t peer, const ExecutionContext &context)
{
    require_not_failed(*this);
    if (peer != impl_->other_peer)
    {
        throw std::runtime_error("invalid_argument: peer must be this communicator's single other rank");
    }
    validate_buffer_view(buffer, count, datatype);

    std::shared_ptr<ExternalMemoryProvider> provider = make_provider(buffer, context, impl_->provider_slots);
    auto &backend = provider->primary_backend();

    TransferRequest request;
    request.direction = TransferDirection::Recv;
    request.backend = &backend;
    request.transport = impl_->transport.get();
    request.total_bytes = buffer.bytes;
    request.chunk_hint = 0;

    try
    {
        Work work = impl_->comm_worker->enqueue(request);
        return impl_->track(work, std::move(provider));
    }
    catch (...)
    {
        impl_->mark_failed("recv enqueue failed");
        throw;
    }
}

Work Communicator::all_reduce(
    const BufferView &send_buf,
    const BufferView &recv_buf,
    std::size_t count,
    DataType datatype,
    ReduceOp op,
    const ExecutionContext &context)
{
    require_not_failed(*this);

    if (impl_->world_size != 2)
    {
        throw std::runtime_error("unsupported: all_reduce requires world_size==2 (heterogeneous N=2 engine)");
    }
    if (op != ReduceOp::Sum)
    {
        throw std::runtime_error("unsupported: all_reduce only supports ReduceOp::Sum this phase");
    }
    validate_reduction(datatype, op);
    if (!impl_->caps.supports_collective_all_reduce(recv_buf.memory_kind, datatype, op))
    {
        throw std::runtime_error(
            std::string("unsupported: all_reduce of dtype=") + datatype_label(datatype) + " is not available for this memory kind");
    }
    validate_buffer_view(send_buf, count, datatype);
    validate_buffer_view(recv_buf, count, datatype);

    constexpr std::size_t kRoot = 0; // Part X: fixed internal policy, never exposed.
    const std::size_t rank = impl_->rank;
    const std::size_t total_bytes = recv_buf.bytes;

    // Keep the provider alive for the duration of the collective by
    // capturing it (shared_ptr) into the executor job's lambda.
    auto provider = std::shared_ptr<ExternalMemoryProvider>(make_provider(recv_buf, context, impl_->provider_slots));

    if (send_buf.data != recv_buf.data && count > 0)
    {
        // Out-of-place: stage send_buf's content into recv_buf's location
        // before the collective, so both root and non-root logic can
        // uniformly treat recv_buf as "holds the local input, ends up
        // holding the result" (Part AB). Host-only memcpy is correct
        // here because MemoryKind::MetalShared's data is CPU-visible by
        // contract and MemoryKind::Cuda providers are expected to
        // perform device-side copies before returning from their own
        // factory if they need this -- Phase 41 does not implement an
        // out-of-place external-CUDA AllReduce; call with send_buf.data
        // == recv_buf.data (in-place) for Cuda buffers.
        if (recv_buf.memory_kind == MemoryKind::Cuda)
        {
            throw std::runtime_error(
                "unsupported: out-of-place all_reduce (send_buf.data != recv_buf.data) is not supported for MemoryKind::Cuda; pass the same BufferView for send_buf and recv_buf");
        }
        std::memcpy(recv_buf.data, send_buf.data, recv_buf.bytes);
    }

    Transport *transport = impl_->transport.get();
    TensorCommWorker *worker = impl_->comm_worker.get();

    auto run = [transport, worker, provider, rank, total_bytes, count, datatype]() {
        AsyncMemoryBackend &primary = provider->primary_backend();
        if (rank == kRoot)
        {
            AsyncMemoryBackend &scratch = provider->scratch_backend();
            LocalReduceBackend &reduce = provider->reduce_backend();
            n2_all_reduce_tensor(
                *transport, *worker, scratch, primary, &reduce,
                rank, kRoot, total_bytes, /*chunk_hint=*/0, count, datatype);
        }
        else
        {
            n2_all_reduce_tensor(
                *transport, *worker, primary, primary, nullptr,
                rank, kRoot, total_bytes, /*chunk_hint=*/0, count, datatype);
        }
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

namespace
{

std::atomic<std::uint64_t> g_phase43_transfer_id{std::uint64_t{1} << 40};

void run_transfer(
    TensorCommWorker &worker,
    Transport &transport,
    AsyncMemoryBackend &backend,
    TransferDirection direction,
    std::size_t bytes,
    const char *what)
{
    TransferRequest request;
    request.transfer_id = g_phase43_transfer_id.fetch_add(1, std::memory_order_relaxed);
    request.direction = direction;
    request.backend = &backend;
    request.transport = &transport;
    request.total_bytes = bytes;
    request.chunk_hint = 0;
    TransferWork work = worker.enqueue(request);
    work.wait();
    if (work.has_error()) throw std::runtime_error(std::string("transport_error: ") + what + ": " + work.error());
}

} // namespace

Work Communicator::broadcast(const BufferView &buffer, std::size_t root, const ExecutionContext &context)
{
    require_not_failed(*this);
    if (impl_->world_size != 2) throw std::runtime_error("unsupported: broadcast requires world_size==2");
    if (root >= impl_->world_size) throw std::runtime_error("invalid_argument: broadcast root out of range");
    if (buffer.bytes > 0 && buffer.data == nullptr)
        throw std::runtime_error("invalid_argument: broadcast buffer.data is null for a non-empty buffer");

    const std::size_t rank = impl_->rank;
    const std::size_t bytes = buffer.bytes;
    std::shared_ptr<ExternalMemoryProvider> provider;
    if (bytes > 0) provider = make_provider(buffer, context, impl_->provider_slots);

    Transport *transport = impl_->transport.get();
    TensorCommWorker *worker = impl_->comm_worker.get();
    auto run = [transport, worker, provider, rank, root, bytes]() {
        if (bytes == 0) return;
        run_transfer(
            *worker, *transport, provider->primary_backend(),
            rank == root ? TransferDirection::Send : TransferDirection::Recv, bytes, "broadcast");
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
    if (impl_->world_size != 2) throw std::runtime_error("unsupported: all_gather requires world_size==2");
    if (outputs.size() != impl_->world_size)
        throw std::runtime_error("invalid_argument: all_gather needs exactly world_size output buffers");
    const std::size_t bytes = input.bytes;
    for (const auto &out : outputs)
    {
        if (out.bytes != bytes) throw std::runtime_error("invalid_argument: all_gather output size differs from input size");
    }
    if (bytes > 0 && input.data == nullptr) throw std::runtime_error("invalid_argument: all_gather input.data is null");
    for (const auto &out : outputs)
    {
        if (bytes > 0 && out.data == nullptr) throw std::runtime_error("invalid_argument: all_gather output.data is null");
    }

    const std::size_t rank = impl_->rank;
    const std::size_t peer = impl_->other_peer;
    std::shared_ptr<ExternalMemoryProvider> in_provider, local_out_provider, peer_out_provider;
    if (bytes > 0)
    {
        in_provider = make_provider(input, context, impl_->provider_slots);
        peer_out_provider = make_provider(outputs[peer], context, impl_->provider_slots);
        if (outputs[rank].data != input.data) local_out_provider = make_provider(outputs[rank], context, impl_->provider_slots);
    }

    Transport *transport = impl_->transport.get();
    TensorCommWorker *worker = impl_->comm_worker.get();
    auto run = [transport, worker, in_provider, local_out_provider, peer_out_provider, rank, bytes]() {
        if (bytes == 0) return;
        if (local_out_provider)
        {
            // outputs[rank] <- input through the providers' own staging interface (no device code here).
            constexpr std::size_t kCopyChunk = std::size_t{1} << 20;
            std::vector<unsigned char> staging(std::min(bytes, kCopyChunk));
            for (std::size_t offset = 0; offset < bytes; offset += kCopyChunk)
            {
                const Chunk chunk{offset, std::min(kCopyChunk, bytes - offset)};
                in_provider->primary_backend().stage_source_chunk(chunk, staging.data());
                local_out_provider->primary_backend().commit_destination_chunk(chunk, staging.data());
            }
        }
        // Deterministic, deadlock-safe order: lower rank sends first.
        if (rank == 0)
        {
            run_transfer(*worker, *transport, in_provider->primary_backend(), TransferDirection::Send, bytes, "all_gather send");
            run_transfer(*worker, *transport, peer_out_provider->primary_backend(), TransferDirection::Recv, bytes, "all_gather recv");
        }
        else
        {
            run_transfer(*worker, *transport, peer_out_provider->primary_backend(), TransferDirection::Recv, bytes, "all_gather recv");
            run_transfer(*worker, *transport, in_provider->primary_backend(), TransferDirection::Send, bytes, "all_gather send");
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

} // namespace tbccl
