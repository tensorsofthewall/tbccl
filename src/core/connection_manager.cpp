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

std::unique_ptr<ConnectionManager> ConnectionManager::establish(
    const ResolvedBootstrap &boot, std::chrono::milliseconds timeout, CommunicatorListeners *prebound, const PeerCapabilities &local,
    NegotiationResult &aggregate_negotiation)
{
    std::unique_ptr<ConnectionManager> mgr(new ConnectionManager());
    mgr->rank_ = boot.rank;
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
    dial_lower_ranks(boot, ConnectionRole::Data, deadline, data);
    if (data_listener) accept_higher_ranks(boot, ConnectionRole::Data, *data_listener, deadline, data);
    control_listener.reset();
    data_listener.reset();

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
        data[peer]->set_io_timeout(std::chrono::milliseconds(0));
        channel->control = std::move(control[peer]);
        channel->data = std::make_unique<TcpTransport>(std::move(data[peer]));
        channel->worker = std::make_unique<TensorCommWorker>(/*pipeline_depth=*/2, /*queue_depth=*/8, /*duplex=*/true);
        mgr->channels_[peer] = std::move(channel);
    }
    aggregate_negotiation = fold_negotiation(negotiations, negotiated_ranks);
    return mgr;
}

ConnectionManager::~ConnectionManager() = default;

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
    for (const auto &c : channels_)
    {
        if (!c) continue;
        c->data->abort(reason);
        c->worker->abort(reason);
    }
}

} // namespace detail
} // namespace tbccl
