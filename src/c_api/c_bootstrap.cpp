// TBCCL C ABI v1: caller-exchanged endpoint bootstrap (docs/c_api_bootstrap.md).
//
//   tbcclGetUniqueId -> (app distributes the id) -> tbcclBootstrapBegin (binds this rank's listeners on kernel-chosen ports and owns them)
//   -> tbcclBootstrapGetEndpoint -> (app all-gathers the opaque blobs) -> tbcclBootstrapComplete (hands the listeners to the Communicator) -> tbcclBootstrapDestroy
//
// There is no hidden rendezvous server and no process-global registry: the only shared state is what the application passes around. A bootstrap handle is
// single-threaded. The blob payload is serialized explicitly (little-endian), never memcpy'd from a struct, and carries its own format version.

#include "c_internal.hpp"

#include <tbccl/rank_directory.hpp>

#include <arpa/inet.h>

#include <chrono>
#include <cstdint>
#include <vector>

using namespace tbccl;
using namespace tbccl::capi;

namespace
{
constexpr std::uint32_t kBlobFormatVersion = 1;
constexpr std::uint32_t kBlobMagic = 0x50455442u; // "BTEP"
constexpr std::size_t kPayloadBytes = 240;
constexpr std::size_t kHeaderBytes = 33;          // magic, rank, world, id[16], control port, data port, host length
constexpr std::size_t kMaxHostBytes = 63;

void put32(std::uint8_t *p, std::uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i)); }
void put16(std::uint8_t *p, std::uint16_t v) { p[0] = static_cast<std::uint8_t>(v); p[1] = static_cast<std::uint8_t>(v >> 8); }
std::uint32_t get32(const std::uint8_t *p) { return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24); }
std::uint16_t get16(const std::uint8_t *p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

struct EndpointRecord
{
    std::uint32_t rank = 0, world = 0;
    CommunicatorId id;
    std::uint16_t control_port = 0, data_port = 0;
    std::string host;
};

bool is_ipv4(const std::string &s)
{
    in_addr a;
    return !s.empty() && inet_pton(AF_INET, s.c_str(), &a) == 1;
}

bool is_wildcard(const std::string &s) { return s == "0.0.0.0"; }

void serialize(const EndpointRecord &r, tbcclEndpointBlob &blob)
{
    std::memset(&blob, 0, sizeof(blob));
    blob.struct_size = TBCCL_ENDPOINT_BLOB_SIZE;
    blob.format_version = kBlobFormatVersion;
    std::uint8_t *p = blob.payload;
    put32(p, kBlobMagic);
    put32(p + 4, r.rank);
    put32(p + 8, r.world);
    std::memcpy(p + 12, r.id.bytes().data(), CommunicatorId::kBytes);
    put16(p + 28, r.control_port);
    put16(p + 30, r.data_port);
    p[32] = static_cast<std::uint8_t>(r.host.size());
    std::memcpy(p + kHeaderBytes, r.host.data(), r.host.size());
    blob.used_bytes = static_cast<std::uint32_t>(kHeaderBytes + r.host.size());
}

// false for anything malformed or from another format version; the caller maps that to INVALID_ARGUMENT.
bool parse(const tbcclEndpointBlob &blob, EndpointRecord &r)
{
    if (blob.struct_size != TBCCL_ENDPOINT_BLOB_SIZE || blob.format_version != kBlobFormatVersion || blob.reserved != 0) return false;
    if (blob.used_bytes < kHeaderBytes || blob.used_bytes > kPayloadBytes) return false;
    const std::uint8_t *p = blob.payload;
    if (get32(p) != kBlobMagic) return false;
    const std::size_t host_len = p[32];
    if (host_len == 0 || host_len > kMaxHostBytes || kHeaderBytes + host_len != blob.used_bytes) return false;
    r.rank = get32(p + 4);
    r.world = get32(p + 8);
    std::array<std::uint8_t, CommunicatorId::kBytes> id;
    std::memcpy(id.data(), p + 12, id.size());
    r.id = CommunicatorId(id);
    r.control_port = get16(p + 28);
    r.data_port = get16(p + 30);
    r.host.assign(reinterpret_cast<const char *>(p + kHeaderBytes), host_len);
    return is_ipv4(r.host);
}
} // namespace

struct tbcclBootstrap_st
{
    std::uint32_t rank = 0, world = 0;
    CommunicatorId id;
    std::chrono::milliseconds timeout{10000};
    std::shared_ptr<CommunicatorListeners> listeners; // only for a rank that accepts connections; handed to the Communicator by Complete
    EndpointRecord mine;
    bool used = false; // Complete was called (successfully or not): the listeners are no longer ours
};

extern "C"
{

tbcclResult_t TBCCL_CALL tbcclGetUniqueId(tbcclUniqueId *id)
{
    return guard([&]() -> tbcclResult_t {
        if (id == nullptr) return TBCCL_INVALID_ARGUMENT;
        std::memset(id, 0, sizeof(*id));
        std::memcpy(id->bytes, CommunicatorId::generate().bytes().data(), CommunicatorId::kBytes);
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclBootstrapBegin(
    uint32_t rank, uint32_t world_size, const tbcclUniqueId *id, const tbcclBootstrapOptions *options, tbcclBootstrap_t *bootstrap)
{
    return guard([&]() -> tbcclResult_t {
        if (bootstrap == nullptr) return TBCCL_INVALID_ARGUMENT;
        *bootstrap = nullptr;
        if (id == nullptr || world_size == 0 || world_size > kMaxFullMeshWorldSize || rank >= world_size) return TBCCL_INVALID_ARGUMENT;

        std::string bind_host = "127.0.0.1", advertise_host;
        std::uint32_t timeout_ms = 10000;
        bool advertise_given = false;
        if (options != nullptr)
        {
            if (!size_ok(options, sizeof(tbcclBootstrapOptions))) return TBCCL_INVALID_ARGUMENT;
            if (options->reserved0 != 0 || options->reserved1 != 0 || options->reserved2[0] != 0 || options->reserved2[1] != 0 || options->reserved2[2] != 0)
                return TBCCL_INVALID_ARGUMENT;
            if (options->bind_host != nullptr) bind_host = options->bind_host; // copied now; the caller's strings are not retained
            if (options->advertise_host != nullptr)
            {
                advertise_host = options->advertise_host;
                advertise_given = true;
            }
            if (options->timeout_ms != 0) timeout_ms = options->timeout_ms;
        }
        if (!is_ipv4(bind_host)) return TBCCL_INVALID_ARGUMENT;
        if (!advertise_given) advertise_host = bind_host;
        // Binding a wildcard says nothing about which address the peers should dial: never serialize one.
        if (!is_ipv4(advertise_host) || is_wildcard(advertise_host) || advertise_host.size() > kMaxHostBytes) return TBCCL_INVALID_ARGUMENT;

        std::unique_ptr<tbcclBootstrap_st> b(new tbcclBootstrap_st());
        b->rank = rank;
        b->world = world_size;
        std::array<std::uint8_t, CommunicatorId::kBytes> raw;
        std::memcpy(raw.data(), id->bytes, raw.size());
        b->id = CommunicatorId(raw);
        b->timeout = std::chrono::milliseconds(timeout_ms);
        b->mine.rank = rank;
        b->mine.world = world_size;
        b->mine.id = b->id;
        b->mine.host = advertise_host;
        if (world_size > 1 && rank_accepts_connections(rank, world_size))
        {
            b->listeners = CommunicatorListeners::bind(bind_host, 0, 0); // the ACTUAL kernel-chosen ports are what gets serialized
            b->mine.control_port = b->listeners->control().port;
            b->mine.data_port = b->listeners->data().port;
        }
        *bootstrap = b.release();
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclBootstrapGetEndpoint(tbcclBootstrap_t bootstrap, tbcclEndpointBlob *blob)
{
    return guard([&]() -> tbcclResult_t {
        if (bootstrap == nullptr || blob == nullptr || blob->struct_size < TBCCL_ENDPOINT_BLOB_SIZE) return TBCCL_INVALID_ARGUMENT;
        serialize(bootstrap->mine, *blob);
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclBootstrapComplete(tbcclBootstrap_t bootstrap, const tbcclEndpointBlob *blobs, uint32_t blob_count, tbcclComm_t *comm)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        *comm = nullptr;
        if (bootstrap == nullptr || blobs == nullptr || bootstrap->used) return TBCCL_INVALID_ARGUMENT;
        if (blob_count != bootstrap->world) return TBCCL_INVALID_ARGUMENT; // mismatched world size

        // Validate everything before any listener is handed over, so a malformed input leaves the bootstrap intact.
        std::vector<EndpointRecord> records(blob_count);
        for (std::uint32_t i = 0; i < blob_count; ++i)
        {
            if (!parse(blobs[i], records[i])) return TBCCL_INVALID_ARGUMENT;                  // malformed blob, wrong format version
            if (records[i].id != bootstrap->id) return TBCCL_INVALID_ARGUMENT;               // wrong communicator id
            if (records[i].world != bootstrap->world) return TBCCL_INVALID_ARGUMENT;         // mismatched world size
            if (records[i].rank != i) return TBCCL_INVALID_ARGUMENT;                         // out of order or duplicate rank
        }
        const EndpointRecord &me = records[bootstrap->rank];
        if (me.control_port != bootstrap->mine.control_port || me.data_port != bootstrap->mine.data_port || me.host != bootstrap->mine.host)
            return TBCCL_INVALID_ARGUMENT; // this rank's own blob is not the one it published

        CommunicatorOptions options;
        options.rank = bootstrap->rank;
        options.world_size = bootstrap->world;
        options.communicator_id = bootstrap->id;
        options.bootstrap_timeout = bootstrap->timeout;
        for (const auto &r : records)
        {
            RankEndpoint e;
            e.rank = r.rank;
            e.control = Endpoint{r.host, r.control_port};
            e.data = Endpoint{r.host, r.data_port};
            options.rank_directory.entries.push_back(std::move(e));
        }
        options.listeners = bootstrap->listeners;
        bootstrap->used = true; // from here on the listeners belong to the communicator creation, whatever its outcome
        std::unique_ptr<tbcclComm_st> handle(new tbcclComm_st());
        handle->comm = Communicator::create(options);
        *comm = handle.release();
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclBootstrapDestroy(tbcclBootstrap_t bootstrap)
{
    return guard([&]() -> tbcclResult_t {
        delete bootstrap; // safe in the incomplete, completed and failed states; releases any listener still owned
        return TBCCL_SUCCESS;
    });
}

} // extern "C"
