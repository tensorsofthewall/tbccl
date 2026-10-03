#pragma once

// Phase 50: the multi-peer connection runtime behind Communicator, private to libtbccl.
//
// A ConnectionManager owns one PeerChannel per remote rank. All Communicator traffic goes through a PeerChannel; nothing
// above this class knows how the channels were made (full mesh today; Phase 51 may make them lazy or sparse without
// touching Communicator). For world_size <= kMaxFullMeshWorldSize it eagerly builds a full mesh with this rule: for every rank
// pair the LOWER rank dials and the HIGHER rank accepts, so exactly two sockets exist per pair (control, data) and there are
// no connect/connect races. Phases: all control connections, then all data connections, then a capability exchange over the
// control connections. Every connection starts with the wire-protocol handshake (wire_protocol.hpp).
//
// PeerChannel:
//   control   Hello, capability record, and (after bootstrap) Abort / Goodbye frames; one persistent watcher thread per
//             channel is added with the abort-propagation work
//   data      payload bytes and in-band collective descriptors
//   worker    a duplex TensorCommWorker: independent send and receive lanes, so one direction never delays the other and
//             different peers make independent progress

#include "bootstrap_config.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/communicator.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <memory>
#include <string>
#include <vector>

namespace tbccl::detail
{

struct PeerChannel
{
    std::size_t peer_rank = 0;
    // Destruction order (reverse of declaration): worker threads are joined before the transport they use is closed.
    std::unique_ptr<Connection> control;
    std::unique_ptr<TcpTransport> data;
    std::unique_ptr<TensorCommWorker> worker;
    PeerCapabilities capabilities;

    // Control-channel state. The watcher thread is the only reader of `control`; writers (Abort / Goodbye frames) take the mutex.
    std::mutex control_send_mutex;
    std::thread watcher;
    std::atomic<bool> departed{false}; // the peer said Goodbye: a later EOF on its sockets is not a failure
};

// What a control-channel watcher reports to the Communicator. `origin` is the rank that first aborted (the peer itself, or a rank
// the peer is relaying for); `peer_lost` means the control connection broke without a Goodbye or an Abort frame.
using PeerEventHandler = std::function<void(std::size_t peer, std::size_t origin, const std::string &reason, bool peer_lost)>;

class ConnectionManager
{
public:
    // Builds the mesh described by `boot`, bounded by `timeout`. If `prebound` is non-null its listeners are used (and
    // consumed) instead of binding the rank's own directory entry. `aggregate_negotiation` receives the fold of every
    // per-peer negotiation (common memory backends intersected, smallest non-zero chunk, largest alignment); for
    // world_size 1 it is the local capabilities negotiated with themselves. Throws on any failure with every socket closed.
    static std::unique_ptr<ConnectionManager> establish(
        const ResolvedBootstrap &boot,
        std::chrono::milliseconds timeout,
        CommunicatorListeners *prebound,
        const PeerCapabilities &local_capabilities,
        NegotiationResult &aggregate_negotiation);

    ~ConnectionManager();
    ConnectionManager(const ConnectionManager &) = delete;
    ConnectionManager &operator=(const ConnectionManager &) = delete;

    std::size_t rank() const noexcept { return rank_; }
    std::size_t world_size() const noexcept { return channels_.size(); }

    // The channel to `peer`; throws "invalid_argument: ..." for this rank itself or a rank outside the world.
    PeerChannel &channel(std::size_t peer);
    const std::vector<std::unique_ptr<PeerChannel>> &channels() const noexcept { return channels_; } // null at own rank

    // True while any lane of any peer has queued or active work.
    bool busy() const;
    // Blocks (event-driven) until every lane of every peer is idle.
    void wait_idle();
    void set_fatal_handler(const std::function<void(const std::string &)> &handler);
    // Interrupts every data transport (a blocked send/recv unwinds) and fails every queued request. Idempotent.
    void abort_transfers(const std::string &reason);

    // One persistent thread per peer reads that peer's control connection. An Abort frame or a control connection that breaks
    // without a Goodbye is reported through `handler` (from the watcher thread; it must not block). Call once, after bootstrap.
    void start_watchers(PeerEventHandler handler);
    // Stops and joins every watcher thread (idempotent). The Communicator calls it before any of its own state is destroyed, because
    // a watcher calls back into that state.
    void stop_watchers();
    // Best effort, never blocks on a dead peer: tells every peer this rank is aborting (`origin` = the rank that first aborted).
    void broadcast_abort(std::size_t origin, const std::string &reason);
    // Best effort: tells every peer this rank is leaving cleanly, so the sockets closing afterwards is not reported as a failure.
    void goodbye();

private:
    ConnectionManager() = default;
    std::size_t rank_ = 0;
    std::atomic<bool> closing_{false};
    std::vector<std::unique_ptr<PeerChannel>> channels_;
};

// Out-of-line access to CommunicatorListeners' sockets (the class is public but its sockets are private).
struct ListenersAccess
{
    static std::unique_ptr<Listener> take_control(CommunicatorListeners &l);
    static std::unique_ptr<Listener> take_data(CommunicatorListeners &l);
};

} // namespace tbccl::detail
