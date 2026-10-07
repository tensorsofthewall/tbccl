# Error reference

## C++ error codes and C result codes

| `tbccl::ErrorCode` | C result (`tbcclResult_t`) | Value | Typical cause |
|---|---|---|---|
| `Success` | `TBCCL_SUCCESS` | 0 | |
| `InvalidArgument` | `TBCCL_INVALID_ARGUMENT` | 1 | null or malformed argument, rank or peer out of range or equal to self, size overflow, wrong id, world size or duplicate rank at bootstrap |
| `Unsupported` | `TBCCL_UNSUPPORTED` | 2 | unsupported datatype, operation, memory kind or world size; CUDA not built in |
| `ResourceExhausted` | `TBCCL_RESOURCE_EXHAUSTED` | 3 | an allocation failed while admitting an operation |
| `Aborted` | `TBCCL_ABORTED` | 4 | the communicator was aborted (explicitly, by a peer, or after a fatal failure) |
| `Timeout` | `TBCCL_TIMEOUT` | 5 | a bootstrap or runtime timeout (never a caller's `wait_for` expiring) |
| `ProtocolMismatch` | `TBCCL_PROTOCOL_MISMATCH` | 6 | ranks disagree: communicator id, world size, wire version, collective descriptor, P2P size |
| `TransportError`, `PeerFailure` | `TBCCL_TRANSPORT_ERROR` | 7 | socket failure or peer failure |
| `InternalError` | `TBCCL_INTERNAL_ERROR` | 8 | an unexpected exception or violated invariant |
| `DeviceError` | `TBCCL_DEVICE_ERROR` | 9 | a CUDA or device failure |

`tbcclGetResultString` returns static, generic text for a result. The operation-specific text is available per `Work` through `tbcclWorkGetErrorString` (C) or `Work::error()` (C++).

## Message tags

C++ messages start with a lowercase tag that matches the code: `invalid_argument:`, `unsupported:`, `transport_error:`, `peer_failure:`, `protocol_mismatch:`, `timeout:`. The tag is for humans and for adapters that map text to their own error types; code that must branch on the failure should use the structured `error_code()`.

## Common messages

| Message contains | Meaning | What to check |
|---|---|---|
| `protocol_mismatch: ... wire protocol` | a peer speaks a different wire protocol version | rebuild every rank and every adapter against the same installed TBCCL package ([Versioning](versioning.md)) |
| `protocol_mismatch: ... communicator` or `duplicate rank` | ranks were started with different communicator ids, or two processes claim one rank | every rank must receive the same id and a unique rank |
| `timeout: ... waiting for the connection(s)` | a peer did not connect all its connections within `bootstrap_timeout` | the peer is running, its endpoint is reachable and was published correctly |
| `protocol_mismatch: ... size mismatch` | a P2P receive was posted for a different byte count than the sender sent | match send and receive sizes |
| `unsupported: reduction dtype=...` | the datatype and operation combination is not available for this world size or memory kind | [Platform capabilities](platform-capabilities.md) |
| `unsupported: out-of-place all_reduce ... MemoryKind::Cuda` | CUDA all_reduce needs the same buffer for send and receive | pass the same view for both |
