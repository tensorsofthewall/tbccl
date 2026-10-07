# C API thread-safety contract (v1)

| operation | may be called concurrently with | notes |
|---|---|---|
| `tbcclSend` / `tbcclRecv` on one communicator | each other, from any threads | Submissions are linearized by TBCCL; **FIFO per (peer, direction) is the runtime's local linearization order**. The receiving application still matches the logical order per peer pair (there are no tags). |
| `tbcclBarrier` / `Broadcast` / `AllGather` / `AllReduce` | each other, locally thread-safe | **All ranks must submit collectives in the same logical order.** If two application threads issue A then B on one rank and B then A on another, a descriptor mismatch aborts the communicator; TBCCL does not reconcile concurrent collective order across ranks. **Collectives and P2P are independent ordering domains** (wire protocol 4: each peer pair has a separate connection and worker for each): they may overlap on one communicator, from the same or from different threads, and **the relative order of a collective and a P2P call need not be the same on every rank**. What must still match is each domain's own order: the collective sequence, and the P2P messages per (peer, direction). |
| `tbcclWorkTest` / `Wait` / `WaitFor` / `GetErrorString` | each other, and with operations still running | Safe from any thread, including several threads on the same Work. |
| `tbcclWorkDestroy` | **nothing using the same Work handle** | The one rule: do not destroy a handle while another thread is inside a Work call on it. Destroying one Work never affects another Work or the operation itself. |
| `tbcclCommAbort`, `tbcclCommIsAborted`, `tbcclCommGetAbortReason` | everything, from any thread | Abort is idempotent and communicator-wide; queued operations become terminal (`ABORTED`) without touching their buffers. |
| `tbcclCommGetRank` / `GetSize` / `GetCapabilities` / `SupportsAllReduce` | everything | Read-only. |
| `tbcclCommDestroy` | **nothing on that communicator** | The caller quiesces API entry first (no new send, recv, collective or query on the handle). Destruction with operations still outstanding is safe and bounded: it aborts them. Work handles created from the communicator stay valid to query and destroy afterwards (their operations are terminal). |
| `tbcclBootstrapBegin` / `GetEndpoint` / `Complete` / `Destroy` | **single-threaded per handle** | A bootstrap handle is not safe for concurrent use. Different bootstrap handles (different communicators, even in one process) are independent: there is no process-global rendezvous state. |
| `tbcclGetUniqueId`, `tbcclGet*Version`, `tbcclGetResultString`, `tbcclRegisterCudaSupport` | everything | Process-wide, idempotent. |

Stale handles: only `NULL` and currently valid handles are validated; using a handle after its `Destroy` is application undefined behaviour and is not detected.

Tests: `c_api_submission_test` (four submitting threads, six threads querying one Work, abort from a separate thread with 300 queued operations) runs clean under ASan, UBSan and TSan.
