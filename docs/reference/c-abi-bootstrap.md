# C API bootstrap (v1): caller-supplied exchange of opaque endpoint blobs

TBCCL does not discover peers. Every framework already has a way to exchange small blobs between ranks (MPI all-gather, a launcher, a key-value store, files, pipes, a parent process), so v1 asks the application to do exactly that, once, with one fixed-size blob per rank. There is **no hidden rendezvous server, no process-global registry** mapping a unique id to a server, and no TBCCL-owned bootstrap service. Several communicators coexist in one process naturally.

```
rank 0 ... rank N-1                                   (any process model)
  tbcclGetUniqueId(&id)            on ONE process
  <application distributes id to every rank>           the 16-byte tbcclUniqueId carries no addresses
  tbcclBootstrapBegin(rank, N, &id, &opts, &bs)        binds this rank's listeners (kernel-chosen ports), owns them
  tbcclBootstrapGetEndpoint(bs, &my_blob)              256-byte opaque blob with the ACTUAL bound ports
  <application all-gathers the N blobs, rank order>    MPI_Allgather(..., N * sizeof(tbcclEndpointBlob))
  tbcclBootstrapComplete(bs, blobs, N, &comm)          hands the listeners to the communicator and connects
  tbcclBootstrapDestroy(bs)                            always safe afterwards
```

## The unique id

`tbcclUniqueId` is 16 random bytes: the identity of one communicator/session. It is not a secret and has no cryptographic property; it only keeps independent communicators between the same hosts from cross-connecting (a rank presenting another id is rejected at the handshake). It embeds no network address. `tbcclGetUniqueId` is process-independent; whichever process generates it distributes it.

## Options and addresses

`tbcclBootstrapOptions` (all optional, `NULL` options = defaults):

| field | default | meaning |
|---|---|---|
| `bind_host` | `"127.0.0.1"` | the address this rank's listeners bind to (dotted-quad IPv4 in this release) |
| `advertise_host` | `bind_host` | the address written into the blob: the one the **peers** must dial |
| `timeout_ms` | 10000 | bounds the whole bootstrap (connect, accept, handshakes, capability exchange); expiry is `TBCCL_TIMEOUT` |

Strings are NUL-terminated UTF-8 and copied during `Begin`. For cross-host use the advertised address must be reachable by the peers. **Binding a wildcard (`0.0.0.0`) does not say which address to advertise**, so TBCCL never serializes one: a wildcard `bind_host` requires an explicit, non-wildcard `advertise_host`, and a wildcard `advertise_host` is `TBCCL_INVALID_ARGUMENT`. Non-IPv4 text is `TBCCL_INVALID_ARGUMENT` (the transport is IPv4 in this release).

## The endpoint blob

`tbcclEndpointBlob` is exactly 256 bytes (`TBCCL_ENDPOINT_BLOB_SIZE`), so an all-gather is a plain `N * 256` byte exchange with no length negotiation. Layout: `struct_size` (256), `format_version` (**1**), `used_bytes`, `reserved`, then `payload[240]`. Callers never parse the payload. It currently contains a magic, the rank, the world size, the communicator id, the **actual** control and data ports chosen by the kernel, and the advertised host; roughly 50 bytes are used, leaving more than four times headroom for longer hosts or more endpoint data in a later format version. The blob format version is independent of the C ABI version, the package version and the wire protocol version.

Which ranks bind: only a rank that accepts connections (every rank except the last, and none at world size 1). The last rank's blob carries its host and zero ports, and it is still valid input to everyone's `Complete`.

## Complete

`Complete(bs, blobs, world_size, &comm)` validates **everything first** and returns `TBCCL_INVALID_ARGUMENT` without touching the bootstrap for: a blob count different from the world size, a malformed blob, another blob format version, a communicator id different from this bootstrap's, a world size different from this bootstrap's, a rank that does not equal its array index (duplicate or out of order), and this rank's own blob differing from the one it published. Only then are the listeners handed to the communicator and the connection mesh built; connection failures are `TBCCL_TIMEOUT` / `TBCCL_TRANSPORT_ERROR` / `TBCCL_PROTOCOL_MISMATCH`.

`Complete` may be called once. After the call (successful or failed) the bootstrap can only be destroyed; a second `Complete` is `TBCCL_INVALID_ARGUMENT`. `tbcclBootstrapDestroy` is safe in the incomplete, completed and failed states and for `NULL`.

## Thread safety

One bootstrap handle is single-threaded. Different handles are independent ([C API thread safety](c-abi-thread-safety.md)).

## Running ranks on other machines (operator notes)

`examples/c_link_probe.c` is a two-or-three-host correctness probe: each rank writes its blob to a file, the operator copies the files together (`scp`), writes `blobs.all` (blobs concatenated in rank order) next to every rank, and each rank completes. On macOS keep the process inside a **live session**: a process started with `nohup ... &` from an ssh session that then ends (an orphan) was not allowed to reach the LAN and failed its bootstrap with `TBCCL_TIMEOUT`; keeping the ssh connection open, or running the rank inside `tmux`, works.
