#pragma once

// control-plane capability description and
// negotiation, kept deliberately separate from data-plane transport
// (transport.hpp) and memory-backend code (which stays under
// benchmarks/tensor/ -- this header has no CUDA/Metal awareness, only
// enum labels for what a peer reports supporting). See
// docs/concepts/transports.md for the intended future extension
// points (RDMA, native transports, Windows).
//
// This is intentionally small: enough to represent
// today's real OS/transport/memory-backend/async-capability set, not a
// general-purpose extensible schema. New values are added to the enums
// below as new backends/transports are actually implemented, not
// speculatively.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tbccl
{

class Connection;

// Bumped whenever the wire encoding of PeerCapabilities (or the
// negotiation rules applied to it) changes incompatibly. A peer
// advertising a different version than expected fails negotiation
// explicitly (see negotiate()) rather than silently misinterpreting a
// mismatched wire layout.
constexpr std::uint32_t kProtocolVersion = 1;

enum class OsKind : std::uint8_t
{
    Unknown = 0,
    Linux = 1,
    MacOS = 2,
    // Exists as a capability value so a future Windows peer can
    // advertise itself through the same wire format without a protocol
    // version bump -- no Windows code exists.
    Windows = 3,
};

std::string os_kind_name(OsKind kind);

// Transport identifiers a peer can advertise it supports. The async
// tensor-transfer work advertises only "tcp" -- the others exist so
// this enum doesn't need another protocol-version bump the day a real
// second transport exists, but negotiate() only ever picks a
// transport both peers report AND that this build actually implements
// (never a pretend capability).
enum class TransportKind : std::uint8_t
{
    Tcp = 0,
    Rdma = 1,
    NativeThunderbolt = 2,
};

std::string transport_kind_name(TransportKind kind);

// Memory-backend identifiers, matching (by name, not by direct
// dependency) benchmarks/tensor/tensor_backend.hpp's BackendKind. Kept
// as a separate enum here rather than including that benchmark-only
// header, preserving the existing boundary that the core library has
// no CUDA/Metal awareness (tensor_backend.hpp's own docstring).
enum class MemoryBackendKind : std::uint8_t
{
    Host = 0,
    CudaPageable = 1,
    CudaPinned = 2,
    MetalShared = 3,
    MetalPrivateStaged = 4,
};

std::string memory_backend_kind_name(MemoryBackendKind kind);

// What kind of device->host readiness signal a peer's async memory
// backend can use instead of a blocking device synchronize.
enum class AsyncCapability : std::uint8_t
{
    CudaEvents = 0,
    MetalEvents = 1,
};

std::string async_capability_name(AsyncCapability capability);

struct PeerCapabilities
{
    std::uint32_t protocol_version = kProtocolVersion;
    OsKind os = OsKind::Unknown;
    std::vector<TransportKind> transports;
    std::vector<MemoryBackendKind> memory_backends;
    std::vector<AsyncCapability> async_capabilities;

    // Largest chunk this peer is willing to stage/transfer in one
    // piece; 0 means "no specific limit reported".
    std::size_t max_chunk = 0;
    std::size_t preferred_alignment = 1;
};

// This process's own capabilities, detected from how this binary was
// built (TBCCL_ENABLE_CUDA/TBCCL_ENABLE_METAL) and the host OS. Always
// includes TransportKind::Tcp and MemoryBackendKind::Host.
PeerCapabilities local_capabilities();

// Sends `local` over `connection` and blocks for the peer's own
// PeerCapabilities in return, in that order on both sides (both callers
// must agree on this send-then-recv order -- see
// docs/concepts/transports.md). Throws on any I/O error (propagated
// from Connection::send/recv) or malformed wire data.
PeerCapabilities exchange_capabilities(
    Connection &connection,
    const PeerCapabilities &local);

struct NegotiationResult
{
    bool ok = false;
    std::string failure_reason;

    TransportKind transport = TransportKind::Tcp;
    // Memory backends both peers reported support for, intersected --
    // NOT a selection (a transfer's actual source/destination backend
    // is the caller's choice per-transfer, this is only which choices
    // are valid).
    std::vector<MemoryBackendKind> common_memory_backends;
    std::size_t effective_max_chunk = 0;
    std::size_t effective_alignment = 1;
};

// Applies the negotiation rules:
// - protocol_version must match exactly, else fail with a clear reason
//   (never silently proceed on a version mismatch).
// - transport: intersect both sides' advertised transports with what
//   this build actually implements (currently {Tcp} only -- never advertises or selects a capability this build
//   cannot back). Fails if the intersection is empty.
// - memory backends / max_chunk / alignment: informational
//   intersection/min, never causes failure on their own (an empty
//   common_memory_backends list is legal -- callers decide whether
//   that blocks a specific transfer they wanted to run, not
//   negotiate() itself, since negotiate() doesn't know what transfer
//   is intended).
NegotiationResult negotiate(
    const PeerCapabilities &local,
    const PeerCapabilities &remote);

} // namespace tbccl
