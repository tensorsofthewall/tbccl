// full-mesh establishment and PeerChannel ownership (connection_manager.hpp).

#include "connection_manager.hpp"

#include "wire_protocol.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace tbccl
{

struct CommunicatorListeners::Impl
{
    std::string host;
    std::unique_ptr<Listener> control;
    std::unique_ptr<Listener> data;
};

CommunicatorListeners::CommunicatorListeners() : impl_(std::make_unique<Impl>()) {}
CommunicatorListeners::~CommunicatorListeners() = default;

std::unique_ptr<CommunicatorListeners> CommunicatorListeners::bind(const std::string &host, std::uint16_t control_port, std::uint16_t data_port)
{
    std::unique_ptr<CommunicatorListeners> out(new CommunicatorListeners());
    out->impl_->host = host;
    out->impl_->control = tcp_listen(host, control_port, {});
    out->impl_->data = tcp_listen(host, data_port, {});
    return out;
}

Endpoint CommunicatorListeners::control() const { return {impl_->host, impl_->control ? impl_->control->local_port() : std::uint16_t{0}}; }
Endpoint CommunicatorListeners::data() const { return {impl_->host, impl_->data ? impl_->data->local_port() : std::uint16_t{0}}; }

namespace detail
{

std::unique_ptr<Listener> ListenersAccess::take_control(CommunicatorListeners &l) { return std::move(l.impl_->control); }
std::unique_ptr<Listener> ListenersAccess::take_data(CommunicatorListeners &l) { return std::move(l.impl_->data); }

// A data transport that connects on first use. The HIGHER rank of a pair dials the lower rank's data listener (the N-rank runtime rule); the lower rank waits for
// its acceptor thread to install the incoming connection. Both happen on the lane worker thread, never on the caller's thread, so send()/recv() on the Communicator
// stay non-blocking. Waiting for the peer to dial is unbounded (like any receive) and is ended by abort(). Exactly one connection can ever exist: only one side
// dials, and the acceptor rejects a second connection for the same rank.
struct DialSpec
{
    bool enabled = false; // false on the accepting (lower-rank) side
    Endpoint endpoint;
    Hello hello;
    std::size_t peer = 0;
    std::chrono::milliseconds timeout{10000};
};

std::unique_ptr<Connection> dial_data_peer(const DialSpec &spec, const std::function<bool()> &cancelled, const std::function<void(Connection *)> &registered);

class LazyDataTransport final : public Transport
{
public:
    explicit LazyDataTransport(DialSpec dial) : dial_(std::move(dial)) {}

    void install(std::unique_ptr<Connection> connection)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!inner_ && !aborted_) inner_ = std::make_unique<TcpTransport>(std::move(connection));
        cv_.notify_all();
    }

    bool connected() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return inner_ != nullptr;
    }

    void send(const void *data, std::size_t bytes) override { ensure().send(data, bytes); }
    void recv(void *data, std::size_t bytes) override { ensure().recv(data, bytes); }
    void send_framed(const void *header, std::size_t header_bytes, const void *data, std::size_t bytes) override
    {
        ensure().send_framed(header, header_bytes, data, bytes);
    }
    void recv_framed(void *header, std::size_t header_bytes, void *data, std::size_t bytes, const std::function<void(const void *)> &validate) override
    {
        ensure().recv_framed(header, header_bytes, data, bytes, validate);
    }
    TransportCapabilities capabilities() const noexcept override { return TcpTransport(std::unique_ptr<Connection>(new NullConnection())).capabilities(); }
    std::string peer_name() const override { return "lazy-data-peer"; }

    void abort(const std::string &reason) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aborted_) return;
        aborted_ = true;
        reason_ = reason;
        if (inner_) inner_->abort(reason);
        if (dialing_conn_) dialing_conn_->abort(reason); // a handshake in progress is interrupted too
        cv_.notify_all();
    }

private:
    struct NullConnection final : Connection
    {
        void send(const void *, std::size_t) override {}
        void recv(void *, std::size_t) override {}
        std::string peer_name() const override { return {}; }
    };

    [[noreturn]] void throw_aborted()
    {
        throw std::runtime_error("aborted: communicator aborted" + (reason_.empty() ? std::string() : " (" + reason_ + ")"));
    }

    Transport &ensure()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;)
        {
            if (aborted_) throw_aborted();
            if (inner_) return *inner_;
            if (dial_.enabled && !dialing_)
            {
                dialing_ = true;
                lock.unlock();
                std::unique_ptr<Connection> connection;
                try
                {
                    // The dial is interruptible by abort(): the retry loop checks the flag, and the connection (once it exists) is registered so abort() can shut it down.
                    connection = dial_data_peer(
                        dial_, [this] { std::lock_guard<std::mutex> l(mutex_); return aborted_; },
                        [this](Connection *c) { std::lock_guard<std::mutex> l(mutex_); dialing_conn_ = c; if (aborted_ && c) c->abort(reason_); });
                }
                catch (const std::exception &e)
                {
                    lock.lock();
                    dialing_ = false;
                    dialing_conn_ = nullptr;
                    if (aborted_) throw_aborted();
                    throw std::runtime_error(std::string("transport_error: lazy data connection failed: ") + e.what());
                }
                lock.lock();
                dialing_conn_ = nullptr;
                if (aborted_) throw_aborted();
                inner_ = std::make_unique<TcpTransport>(std::move(connection));
                cv_.notify_all();
                continue;
            }
            cv_.wait(lock);
        }
    }

    DialSpec dial_;
    Connection *dialing_conn_ = nullptr; // the connection of a dial in progress (guarded by mutex_), so abort() can interrupt its handshake
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unique_ptr<TcpTransport> inner_;
    bool dialing_ = false;
    bool aborted_ = false;
    std::string reason_;
};

namespace
{

using Clock = std::chrono::steady_clock;

std::chrono::milliseconds remaining(Clock::time_point deadline)
{
    return std::max(std::chrono::milliseconds(0), std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()));
}

bool starts_with(const std::string &text, const std::string &prefix) { return text.rfind(prefix, 0) == 0; }

std::unique_ptr<Connection> connect_with_retry(const Endpoint &endpoint, Clock::time_point deadline, const std::string &what)
{
    std::string last;
    for (;;)
    {
        try
        {
            return tcp_connect(endpoint.host, endpoint.port, {});
        }
        catch (const std::exception &e)
        {
            last = e.what();
        }
        const auto left = remaining(deadline);
        if (left.count() <= 0)
            throw std::runtime_error("timeout: bootstrap timed out connecting to " + what + " at " + endpoint.host + ":" + std::to_string(endpoint.port) + " (" + last + ")");
        std::this_thread::sleep_for(std::min(left, std::chrono::milliseconds(20)));
    }
}

std::unique_ptr<Connection> connect_with_retry_cancellable(const Endpoint &endpoint, Clock::time_point deadline, const std::string &what, const std::function<bool()> &cancelled)
{
    std::string last;
    for (;;)
    {
        if (cancelled()) throw std::runtime_error("aborted: communicator aborted");
        try
        {
            return tcp_connect(endpoint.host, endpoint.port, {});
        }
        catch (const std::exception &e)
        {
            last = e.what();
        }
        const auto left = remaining(deadline);
        if (left.count() <= 0)
            throw std::runtime_error("timeout: lazy data connection timed out connecting to " + what + " at " + endpoint.host + ":" + std::to_string(endpoint.port) + " (" + last + ")");
        std::this_thread::sleep_for(std::min(left, std::chrono::milliseconds(10)));
    }
}

std::string join_ranks(const std::vector<std::size_t> &ranks)
{
    std::string out;
    for (auto r : ranks) out += (out.empty() ? "" : ", ") + std::to_string(r);
    return out;
}

// Dials every lower rank for `role`, returning the connections indexed by peer rank.
void dial_lower_ranks(
    const ResolvedBootstrap &boot, ConnectionRole role, Clock::time_point deadline, std::vector<std::unique_ptr<Connection>> &slots)
{
    for (std::size_t peer = 0; peer < boot.rank; ++peer)
    {
        const auto &entry = boot.directory.entries[peer];
        const Endpoint &endpoint = role == ConnectionRole::Control ? entry.control : entry.data;
        auto connection = connect_with_retry(endpoint, deadline, std::string("rank ") + std::to_string(peer) + " " + connection_role_name(role) + " endpoint");
        connection->set_io_timeout(std::max(std::chrono::milliseconds(1), remaining(deadline)));
        Hello hello;
        hello.communicator_id = boot.communicator_id;
        hello.rank = static_cast<std::uint32_t>(boot.rank);
        hello.world_size = static_cast<std::uint32_t>(boot.world_size);
        hello.role = role;
        dial_handshake(*connection, hello, peer);
        slots[peer] = std::move(connection);
    }
}

// Accepts one connection of `role` from every higher rank. A stranger that never completes a hello is dropped; a
// connection that speaks the protocol but is rejected (wrong id, duplicate rank, ...) fails the bootstrap immediately.
void accept_higher_ranks(
    const ResolvedBootstrap &boot, ConnectionRole role, Listener &listener, Clock::time_point deadline,
    std::vector<std::unique_ptr<Connection>> &slots)
{
    const std::size_t expected = boot.world_size - 1 - boot.rank;
    std::vector<bool> have(boot.world_size, false);
    std::size_t accepted = 0;
    while (accepted < expected)
    {
        const auto left = remaining(deadline);
        if (left.count() <= 0)
        {
            std::vector<std::size_t> missing;
            for (std::size_t r = boot.rank + 1; r < boot.world_size; ++r)
                if (!have[r]) missing.push_back(r);
            throw std::runtime_error(
                "timeout: rank " + std::to_string(boot.rank) + " timed out waiting for the " + connection_role_name(role) + " connection from rank(s) " + join_ranks(missing));
        }
        auto connection = listener.accept_for(left);
        if (!connection) continue;
        connection->set_io_timeout(std::min(std::max(std::chrono::milliseconds(1), left), std::chrono::milliseconds(2000)));
        AcceptExpectation expect;
        expect.communicator_id = boot.communicator_id;
        expect.local_rank = boot.rank;
        expect.world_size = boot.world_size;
        expect.role = role;
        expect.already_connected = &have;
        Hello hello;
        try
        {
            hello = accept_handshake(*connection, expect);
        }
        catch (const std::runtime_error &e)
        {
            if (starts_with(e.what(), "protocol_mismatch: not a TBCCL peer")) continue;
            throw;
        }
        have[hello.rank] = true;
        slots[hello.rank] = std::move(connection);
        ++accepted;
    }
}

NegotiationResult fold_negotiation(const std::vector<NegotiationResult> &results, const std::vector<std::size_t> &ranks)
{
    NegotiationResult out;
    out.ok = true;
    bool first = true;
    for (std::size_t i = 0; i < results.size(); ++i)
    {
        const auto &r = results[i];
        if (!r.ok)
        {
            out.ok = false;
            out.failure_reason = "rank " + std::to_string(ranks[i]) + ": " + r.failure_reason;
            return out;
        }
        if (first)
        {
            out = r;
            first = false;
            continue;
        }
        std::vector<MemoryBackendKind> common;
        for (auto k : out.common_memory_backends)
            if (std::find(r.common_memory_backends.begin(), r.common_memory_backends.end(), k) != r.common_memory_backends.end()) common.push_back(k);
        out.common_memory_backends = std::move(common);
        if (out.effective_max_chunk == 0) out.effective_max_chunk = r.effective_max_chunk;
        else if (r.effective_max_chunk != 0) out.effective_max_chunk = std::min(out.effective_max_chunk, r.effective_max_chunk);
        out.effective_alignment = std::max(out.effective_alignment, r.effective_alignment);
    }
    return out;
}

} // namespace

std::unique_ptr<Connection> dial_data_peer(const DialSpec &spec, const std::function<bool()> &cancelled, const std::function<void(Connection *)> &registered)
{
    const auto deadline = Clock::now() + spec.timeout;
    auto connection = connect_with_retry_cancellable(spec.endpoint, deadline, "rank " + std::to_string(spec.peer) + " data endpoint", cancelled);
    registered(connection.get());
    try
    {
        connection->set_io_timeout(std::max(std::chrono::milliseconds(1), remaining(deadline)));
        dial_handshake(*connection, spec.hello, spec.peer);
        connection->set_io_timeout(std::chrono::milliseconds(0));
    }
    catch (...)
    {
        registered(nullptr); // unregister (under the caller's lock) before the connection is destroyed, so a concurrent abort never touches it
        throw;
    }
    return connection;
}

std::unique_ptr<ConnectionManager> ConnectionManager::establish(
    const ResolvedBootstrap &boot, std::chrono::milliseconds timeout, CommunicatorListeners *prebound, const PeerCapabilities &local,
    NegotiationResult &aggregate_negotiation)
{
    std::unique_ptr<ConnectionManager> mgr(new ConnectionManager(boot.world_size));
    mgr->rank_ = boot.rank;
    mgr->boot_ = boot;
    mgr->dial_timeout_ = timeout;
    const bool lazy_data = boot.world_size > 2;
    mgr->channels_.resize(boot.world_size);
    if (boot.world_size == 1)
    {
        aggregate_negotiation = negotiate(local, local);
        return mgr; // no sockets at all
    }

    const auto deadline = Clock::now() + timeout;

    std::unique_ptr<Listener> control_listener, data_listener;
    if (rank_accepts_connections(boot.rank, boot.world_size))
    {
        if (prebound != nullptr)
        {
            control_listener = ListenersAccess::take_control(*prebound);
            data_listener = ListenersAccess::take_data(*prebound);
            if (!control_listener || !data_listener) throw std::runtime_error("invalid_argument: CommunicatorListeners were already used by another communicator");
        }
        else
        {
            const auto &entry = boot.directory.entries[boot.rank];
            control_listener = tcp_listen(entry.control.host, entry.control.port, {});
            data_listener = tcp_listen(entry.data.host, entry.data.port, {});
        }
    }

    std::vector<std::unique_ptr<Connection>> control(boot.world_size), data(boot.world_size);
    dial_lower_ranks(boot, ConnectionRole::Control, deadline, control);
    if (control_listener) accept_higher_ranks(boot, ConnectionRole::Control, *control_listener, deadline, control);
    if (!lazy_data)
    {
        // world_size 2: the single data connection is made at bootstrap (the specialised fast path must not pay a first-use dial).
        dial_lower_ranks(boot, ConnectionRole::Data, deadline, data);
        if (data_listener) accept_higher_ranks(boot, ConnectionRole::Data, *data_listener, deadline, data);
        data_listener.reset();
    }
    control_listener.reset();

    std::vector<NegotiationResult> negotiations;
    std::vector<std::size_t> negotiated_ranks;
    for (std::size_t peer = 0; peer < boot.world_size; ++peer)
    {
        if (peer == boot.rank) continue;
        auto channel = std::make_unique<PeerChannel>();
        channel->peer_rank = peer;
        control[peer]->set_io_timeout(std::max(std::chrono::milliseconds(1), remaining(deadline)));
        try
        {
            channel->capabilities = exchange_capabilities(*control[peer], local);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("transport_error: capability exchange with rank " + std::to_string(peer) + " failed: " + e.what());
        }
        negotiations.push_back(negotiate(local, channel->capabilities));
        negotiated_ranks.push_back(peer);
        if (!negotiations.back().ok)
            throw std::runtime_error("transport_error: capability negotiation with rank " + std::to_string(peer) + " failed: " + negotiations.back().failure_reason);
        control[peer]->set_io_timeout(std::chrono::milliseconds(0));
        channel->control = std::move(control[peer]);
        if (!lazy_data)
        {
            data[peer]->set_io_timeout(std::chrono::milliseconds(0));
            channel->data = std::make_unique<TcpTransport>(std::move(data[peer]));
        }
        else
        {
            DialSpec dial;
            if (peer < boot.rank) // this rank is the higher one: it dials the lower rank's data listener on first use
            {
                dial.enabled = true;
                dial.endpoint = boot.directory.entries[peer].data;
                dial.hello = Hello{kWireProtocolVersion, boot.communicator_id, static_cast<std::uint32_t>(boot.rank), static_cast<std::uint32_t>(boot.world_size), ConnectionRole::Data};
                dial.peer = peer;
                dial.timeout = timeout;
            }
            channel->data = std::make_unique<LazyDataTransport>(std::move(dial));
        }
        channel->worker = std::make_unique<TensorCommWorker>(/*pipeline_depth=*/2, /*queue_depth=*/8, /*duplex=*/true);
        mgr->channels_[peer] = std::move(channel);
    }
    aggregate_negotiation = fold_negotiation(negotiations, negotiated_ranks);
    if (lazy_data && data_listener)
    {
        mgr->data_listener_endpoint_ = boot.directory.entries[boot.rank].data;
        mgr->data_listener_ = std::move(data_listener);
        mgr->data_installed_.assign(boot.world_size, false);
        mgr->start_data_acceptor();
    }
    return mgr;
}

void ConnectionManager::start_data_acceptor()
{
    data_acceptor_ = std::thread([this] {
        while (!closing_.load())
        {
            std::unique_ptr<Connection> connection;
            try
            {
                connection = data_listener_->accept_for(std::chrono::milliseconds(500)); // fallback poll: shutdown must not depend on the wake-up connection below succeeding
            }
            catch (const std::exception &)
            {
                return; // the listener is gone
            }
            if (!connection || closing_.load()) continue;
            connection->set_io_timeout(std::chrono::milliseconds(2000));
            AcceptExpectation expect;
            expect.communicator_id = boot_.communicator_id;
            expect.local_rank = boot_.rank;
            expect.world_size = boot_.world_size;
            expect.role = ConnectionRole::Data;
            std::vector<bool> have;
            {
                std::lock_guard<std::mutex> lock(data_install_mutex_);
                have = data_installed_;
            }
            expect.already_connected = &have;
            Hello hello;
            try
            {
                hello = accept_handshake(*connection, expect);
            }
            catch (const std::exception &)
            {
                continue; // a stranger, or a rejected (duplicate / foreign) dial: the reply was sent, the connection is dropped
            }
            connection->set_io_timeout(std::chrono::milliseconds(0));
            {
                std::lock_guard<std::mutex> lock(data_install_mutex_);
                data_installed_[hello.rank] = true;
            }
            static_cast<LazyDataTransport &>(*channels_[hello.rank]->data).install(std::move(connection));
        }
    });
}

ConnectionManager::~ConnectionManager()
{
    stop_watchers();
}

void ConnectionManager::stop_watchers()
{
    closing_.store(true);
    if (data_acceptor_.joinable())
    {
        try
        {
            // fast path: wake the acceptor blocked in accept_for() with a throwaway connection to our own listener (dropped by the handshake); if it fails the poll above ends the wait
            auto wake = tcp_connect(data_listener_endpoint_.host, data_listener_->local_port(), {});
        }
        catch (const std::exception &)
        {
        }
        data_acceptor_.join();
    }
    for (const auto &c : channels_)
    {
        if (c && c->control) c->control->abort("communicator closing"); // shutdown(): wakes the watcher blocked in recv
    }
    for (const auto &c : channels_)
    {
        if (c && c->watcher.joinable()) c->watcher.join();
    }
}

void ConnectionManager::start_watchers(PeerEventHandler handler)
{
    for (const auto &c : channels_)
    {
        if (!c) continue;
        PeerChannel *channel = c.get();
        channel->watcher = std::thread([this, channel, handler] {
            try
            {
                for (;;)
                {
                    const ControlFrame frame = recv_control_frame(*channel->control);
                    if (frame.type == ControlFrameType::Goodbye)
                    {
                        channel->departed.store(true);
                        return;
                    }
                    if (frame.type == ControlFrameType::Abort)
                    {
                        handler(channel->peer_rank, frame.origin_rank, frame.reason, /*peer_lost=*/false);
                        return;
                    }
                    if (frame.type == ControlFrameType::CollectiveDescriptor)
                    {
                        mailbox_.post_descriptor(channel->peer_rank, frame.payload);
                        continue;
                    }
                    if (frame.type == ControlFrameType::CollectiveVerdict)
                    {
                        mailbox_.post_verdict(channel->peer_rank, frame.payload);
                        continue;
                    }
                    // Unknown frame types are ignored: a newer peer may add some.
                }
            }
            catch (const std::exception &e)
            {
                if (closing_.load() || channel->departed.load()) return;
                handler(channel->peer_rank, channel->peer_rank, e.what(), /*peer_lost=*/true);
            }
        });
    }
}

void ConnectionManager::broadcast_abort(std::size_t origin, const std::string &reason)
{
    for (const auto &c : channels_)
    {
        if (!c) continue;
        try
        {
            std::lock_guard<std::mutex> lock(c->control_send_mutex);
            ControlFrame frame;
            frame.type = ControlFrameType::Abort;
            frame.origin_rank = static_cast<std::uint32_t>(origin);
            frame.reason = reason;
            send_control_frame(*c->control, frame);
        }
        catch (const std::exception &)
        {
            // The peer is gone or unreachable: nothing more to tell it.
        }
    }
}

void ConnectionManager::goodbye()
{
    for (const auto &c : channels_)
    {
        if (!c) continue;
        try
        {
            std::lock_guard<std::mutex> lock(c->control_send_mutex);
            ControlFrame frame;
            frame.type = ControlFrameType::Goodbye;
            frame.origin_rank = static_cast<std::uint32_t>(rank_);
            send_control_frame(*c->control, frame);
        }
        catch (const std::exception &)
        {
        }
    }
}

PeerChannel &ConnectionManager::channel(std::size_t peer)
{
    if (peer >= channels_.size()) throw std::runtime_error("invalid_argument: rank " + std::to_string(peer) + " is outside the world of " + std::to_string(channels_.size()));
    if (!channels_[peer]) throw std::runtime_error("invalid_argument: rank " + std::to_string(peer) + " is this rank; there is no channel to itself");
    return *channels_[peer];
}

bool ConnectionManager::busy() const
{
    for (const auto &c : channels_)
        if (c && c->worker->busy()) return true;
    return false;
}

void ConnectionManager::wait_idle()
{
    for (const auto &c : channels_)
        if (c) c->worker->wait_idle();
}

void ConnectionManager::set_fatal_handler(const std::function<void(const std::string &)> &handler)
{
    for (const auto &c : channels_)
        if (c) c->worker->set_fatal_handler(handler);
}

void ConnectionManager::abort_transfers(const std::string &reason)
{
    mailbox_.abort(reason);
    for (const auto &c : channels_)
    {
        if (!c) continue;
        c->data->abort(reason);
        c->worker->abort(reason);
    }
}

} // namespace detail
} // namespace tbccl

namespace tbccl::detail
{

namespace
{
// Compares an arriving descriptor with rank 0's current one. Returns the mismatch text, or an empty string if they agree or are about different sequences.
std::string mismatch_text(const CollectiveDescriptor &current, const std::vector<std::uint8_t> &bytes, std::size_t peer)
{
    DescriptorWire wire{};
    std::copy_n(bytes.begin(), std::min(bytes.size(), wire.size()), wire.begin());
    CollectiveDescriptor d = decode_descriptor(wire);
    d.rank = static_cast<std::uint32_t>(peer);
    if (d.sequence != current.sequence) return {};
    const auto verdict = judge_collective({current, d});
    return verdict.status == VerdictStatus::Mismatch ? verdict.text : std::string();
}
} // namespace

void CollectiveMailbox::post_descriptor(std::size_t peer, const std::vector<std::uint8_t> &bytes)
{
    std::function<void(const std::string &)> fatal;
    std::string text;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        descriptors_.at(peer).push_back(bytes);
        if (has_current_)
        {
            text = mismatch_text(current_, bytes, peer);
            if (!text.empty()) fatal = current_fatal_;
        }
    }
    cv_.notify_all();
    if (fatal) fatal("protocol_mismatch: " + text);
}

void CollectiveMailbox::set_current(const CollectiveDescriptor &mine, std::function<void(const std::string &)> fatal)
{
    std::string text;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        has_current_ = true;
        current_ = mine;
        current_fatal_ = fatal;
        for (std::size_t p = 0; p < descriptors_.size() && text.empty(); ++p)
            if (!descriptors_[p].empty()) text = mismatch_text(mine, descriptors_[p].front(), p);
    }
    if (!text.empty() && fatal) fatal("protocol_mismatch: " + text);
}

void CollectiveMailbox::clear_current()
{
    std::lock_guard<std::mutex> lock(mutex_);
    has_current_ = false;
    current_fatal_ = nullptr;
}

void CollectiveMailbox::post_verdict(std::size_t peer, const std::vector<std::uint8_t> &bytes)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        verdicts_.at(peer).push_back(bytes);
    }
    cv_.notify_all();
}

std::vector<std::uint8_t> CollectiveMailbox::wait_on(std::vector<std::deque<std::vector<std::uint8_t>>> &queues, std::size_t peer)
{
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;)
    {
        if (aborted_) throw std::runtime_error("aborted: communicator aborted" + (reason_.empty() ? std::string() : " (" + reason_ + ")"));
        auto &q = queues.at(peer);
        if (!q.empty())
        {
            auto bytes = std::move(q.front());
            q.pop_front();
            return bytes;
        }
        cv_.wait(lock);
    }
}

std::vector<std::uint8_t> CollectiveMailbox::wait_descriptor(std::size_t peer) { return wait_on(descriptors_, peer); }
std::vector<std::uint8_t> CollectiveMailbox::wait_verdict(std::size_t peer) { return wait_on(verdicts_, peer); }

void CollectiveMailbox::abort(const std::string &reason)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aborted_) return;
        aborted_ = true;
        reason_ = reason;
    }
    cv_.notify_all();
}

void ConnectionManager::send_collective_frame(std::size_t peer, bool is_verdict, const std::vector<std::uint8_t> &bytes)
{
    PeerChannel &c = channel(peer);
    ControlFrame frame;
    frame.type = is_verdict ? ControlFrameType::CollectiveVerdict : ControlFrameType::CollectiveDescriptor;
    frame.origin_rank = static_cast<std::uint32_t>(rank_);
    frame.payload = bytes;
    std::lock_guard<std::mutex> lock(c.control_send_mutex);
    send_control_frame(*c.control, frame);
}

bool ConnectionManager::data_connected(std::size_t peer) const
{
    const auto &c = channels_.at(peer);
    if (!c) return false;
    if (const auto *lazy = dynamic_cast<const LazyDataTransport *>(c->data.get())) return lazy->connected();
    return true;
}

std::vector<std::size_t> ConnectionManager::connected_data_peers() const
{
    std::vector<std::size_t> out;
    for (std::size_t p = 0; p < channels_.size(); ++p)
        if (data_connected(p)) out.push_back(p);
    return out;
}

} // namespace tbccl::detail
