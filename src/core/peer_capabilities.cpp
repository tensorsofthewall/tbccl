#include <tbccl/peer_capabilities.hpp>

#include <tbccl/transport.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tbccl
{

std::string os_kind_name(OsKind kind)
{
    switch (kind)
    {
        case OsKind::Linux:
            return "linux";
        case OsKind::MacOS:
            return "macos";
        case OsKind::Windows:
            return "windows";
        case OsKind::Unknown:
        default:
            return "unknown";
    }
}

std::string transport_kind_name(TransportKind kind)
{
    switch (kind)
    {
        case TransportKind::Tcp:
            return "tcp";
        case TransportKind::Rdma:
            return "rdma";
        case TransportKind::NativeThunderbolt:
            return "native_thunderbolt";
        default:
            return "unknown";
    }
}

std::string memory_backend_kind_name(MemoryBackendKind kind)
{
    switch (kind)
    {
        case MemoryBackendKind::Host:
            return "host";
        case MemoryBackendKind::CudaPageable:
            return "cuda_pageable";
        case MemoryBackendKind::CudaPinned:
            return "cuda_pinned";
        case MemoryBackendKind::MetalShared:
            return "metal_shared";
        case MemoryBackendKind::MetalPrivateStaged:
            return "metal_private_staged";
        default:
            return "unknown";
    }
}

std::string async_capability_name(AsyncCapability capability)
{
    switch (capability)
    {
        case AsyncCapability::CudaEvents:
            return "cuda_events";
        case AsyncCapability::MetalEvents:
            return "metal_events";
        default:
            return "unknown";
    }
}

PeerCapabilities local_capabilities()
{
    PeerCapabilities caps;
    caps.protocol_version = kProtocolVersion;

#if defined(__linux__)
    caps.os = OsKind::Linux;
#elif defined(__APPLE__)
    caps.os = OsKind::MacOS;
#elif defined(_WIN32)
    caps.os = OsKind::Windows;
#else
    caps.os = OsKind::Unknown;
#endif

    // Deliberately CUDA/Metal-unaware, matching the rest of this
    // library (see benchmarks/tensor/tensor_backend.hpp's own
    // docstring: the core tbccl library has no CUDA/Metal awareness).
    // A caller that has compiled in a GPU tensor backend (currently
    // only benchmarks/tensor_transfer_bench.cpp) extends this result
    // with the extra backends/async capabilities its own build
    // actually has -- see
    // benchmarks/tensor/tensor_backend_capabilities.hpp.
    caps.transports = {TransportKind::Tcp};
    caps.memory_backends = {MemoryBackendKind::Host};

    // No specific chunk limit is imposed by this process itself; the
    // async substrate's own ChunkPlan decides real chunk sizes.
    caps.max_chunk = 0;
    caps.preferred_alignment = 1;

    return caps;
}

namespace
{

// Minimal, explicit little-endian wire encoding -- not meant to be a
// general serialization framework, just enough to move a
// PeerCapabilities across one Connection. Every multi-byte field is
// written/read as a fixed-width type via memcpy, so this is portable
// across the two peers' compilers regardless of struct padding, and
// deliberately does not depend on Transport (this exchange happens
// once, at connection setup, directly over Connection -- before any
// Transport-level capability negotiation could even be meaningful).

void write_u32(std::vector<std::uint8_t> &buffer, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        buffer.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

void write_u64(std::vector<std::uint8_t> &buffer, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        buffer.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

void write_u8(std::vector<std::uint8_t> &buffer, std::uint8_t value)
{
    buffer.push_back(value);
}

std::vector<std::uint8_t> encode(const PeerCapabilities &caps)
{
    std::vector<std::uint8_t> buffer;
    write_u32(buffer, caps.protocol_version);
    write_u8(buffer, static_cast<std::uint8_t>(caps.os));

    write_u8(buffer, static_cast<std::uint8_t>(caps.transports.size()));
    for (auto t : caps.transports)
    {
        write_u8(buffer, static_cast<std::uint8_t>(t));
    }

    write_u8(buffer, static_cast<std::uint8_t>(caps.memory_backends.size()));
    for (auto m : caps.memory_backends)
    {
        write_u8(buffer, static_cast<std::uint8_t>(m));
    }

    write_u8(buffer, static_cast<std::uint8_t>(caps.async_capabilities.size()));
    for (auto a : caps.async_capabilities)
    {
        write_u8(buffer, static_cast<std::uint8_t>(a));
    }

    write_u64(buffer, static_cast<std::uint64_t>(caps.max_chunk));
    write_u64(buffer, static_cast<std::uint64_t>(caps.preferred_alignment));

    return buffer;
}

class ByteReader
{
public:
    explicit ByteReader(const std::vector<std::uint8_t> &data) : data_(data) {}

    std::uint8_t read_u8()
    {
        require(1);
        return data_[offset_++];
    }

    std::uint32_t read_u32()
    {
        require(4);
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
        {
            value |= static_cast<std::uint32_t>(data_[offset_ + i]) << (8 * i);
        }
        offset_ += 4;
        return value;
    }

    std::uint64_t read_u64()
    {
        require(8);
        std::uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
        {
            value |= static_cast<std::uint64_t>(data_[offset_ + i]) << (8 * i);
        }
        offset_ += 8;
        return value;
    }

private:
    void require(std::size_t n) const
    {
        if (offset_ + n > data_.size())
        {
            throw std::runtime_error(
                "malformed PeerCapabilities wire data (truncated)");
        }
    }

    const std::vector<std::uint8_t> &data_;
    std::size_t offset_ = 0;
};

PeerCapabilities decode(const std::vector<std::uint8_t> &data)
{
    ByteReader reader(data);

    PeerCapabilities caps;
    caps.protocol_version = reader.read_u32();
    caps.os = static_cast<OsKind>(reader.read_u8());

    const std::uint8_t transport_count = reader.read_u8();
    caps.transports.reserve(transport_count);
    for (std::uint8_t i = 0; i < transport_count; ++i)
    {
        caps.transports.push_back(
            static_cast<TransportKind>(reader.read_u8()));
    }

    const std::uint8_t backend_count = reader.read_u8();
    caps.memory_backends.reserve(backend_count);
    for (std::uint8_t i = 0; i < backend_count; ++i)
    {
        caps.memory_backends.push_back(
            static_cast<MemoryBackendKind>(reader.read_u8()));
    }

    const std::uint8_t async_count = reader.read_u8();
    caps.async_capabilities.reserve(async_count);
    for (std::uint8_t i = 0; i < async_count; ++i)
    {
        caps.async_capabilities.push_back(
            static_cast<AsyncCapability>(reader.read_u8()));
    }

    caps.max_chunk = static_cast<std::size_t>(reader.read_u64());
    caps.preferred_alignment = static_cast<std::size_t>(reader.read_u64());

    return caps;
}

} // namespace

PeerCapabilities exchange_capabilities(
    Connection &connection,
    const PeerCapabilities &local)
{
    const std::vector<std::uint8_t> encoded = encode(local);
    const std::uint32_t encoded_size = static_cast<std::uint32_t>(encoded.size());

    connection.send(&encoded_size, sizeof(encoded_size));
    connection.send(encoded.data(), encoded.size());

    std::uint32_t remote_size = 0;
    connection.recv(&remote_size, sizeof(remote_size));

    // Bounded to a sane maximum so a corrupt/hostile peer cannot force
    // an unbounded allocation here -- real encode() output is always a
    // few dozen bytes.
    constexpr std::uint32_t kMaxReasonableSize = 4096;
    if (remote_size > kMaxReasonableSize)
    {
        throw std::runtime_error(
            "PeerCapabilities exchange: peer reported an implausible "
            "payload size");
    }

    std::vector<std::uint8_t> remote_encoded(remote_size);
    if (remote_size > 0)
    {
        connection.recv(remote_encoded.data(), remote_encoded.size());
    }

    return decode(remote_encoded);
}

NegotiationResult negotiate(
    const PeerCapabilities &local,
    const PeerCapabilities &remote)
{
    NegotiationResult result;

    if (local.protocol_version != remote.protocol_version)
    {
        result.ok = false;
        result.failure_reason =
            "protocol version mismatch: local=" +
            std::to_string(local.protocol_version) +
            " remote=" + std::to_string(remote.protocol_version);
        return result;
    }

    // Only ever select a transport this build actually implements
    // (Phase 32 Part Z item 99/100) -- never a capability either peer
    // merely claims to support in the wire format's forward-looking
    // enum.
    const std::vector<TransportKind> implemented = {TransportKind::Tcp};

    bool selected = false;
    for (auto candidate : implemented)
    {
        const bool local_has =
            std::find(local.transports.begin(), local.transports.end(),
                      candidate) != local.transports.end();
        const bool remote_has =
            std::find(remote.transports.begin(), remote.transports.end(),
                      candidate) != remote.transports.end();

        if (local_has && remote_has)
        {
            result.transport = candidate;
            selected = true;
            break;
        }
    }

    if (!selected)
    {
        result.ok = false;
        result.failure_reason =
            "no common, implemented transport between peers";
        return result;
    }

    for (auto backend : local.memory_backends)
    {
        if (std::find(remote.memory_backends.begin(),
                       remote.memory_backends.end(),
                       backend) != remote.memory_backends.end())
        {
            result.common_memory_backends.push_back(backend);
        }
    }

    const auto min_nonzero = [](std::size_t a, std::size_t b) -> std::size_t
    {
        if (a == 0) return b;
        if (b == 0) return a;
        return std::min(a, b);
    };

    result.effective_max_chunk = min_nonzero(local.max_chunk, remote.max_chunk);
    result.effective_alignment =
        std::max<std::size_t>(1, std::max(local.preferred_alignment,
                                           remote.preferred_alignment));

    result.ok = true;
    return result;
}

} // namespace tbccl
