# tbccl


### Algorithm selection: `TBCCL_ALGORITHM`

`all_gather()`, `reduce_scatter()`, and `all_reduce()` each have a
centralized ("reference") implementation and an optimized ring
implementation. By default, every call resolves which one to use via
`auto`: a conservative, transport-agnostic policy based only on world
size and message/segment/tensor size — the same value on every rank,
since it never looks at host identity, interface, OS, or measured
local latency. No collective outside these three (`barrier`,
`broadcast`, `reduce`) participates in algorithm selection.

Force a specific implementation with environment variables:

```bash
TBCCL_ALGORITHM=reference ./app   # force reference for all three
TBCCL_ALGORITHM=ring ./app        # force ring for all three
TBCCL_ALL_GATHER_ALGORITHM=ring ./app
```

Per-collective overrides (`TBCCL_ALL_GATHER_ALGORITHM`,
`TBCCL_REDUCE_SCATTER_ALGORITHM`, `TBCCL_ALL_REDUCE_ALGORITHM`) take
precedence over `TBCCL_ALGORITHM`, which takes precedence over `auto`.
Accepted values are `auto`, `reference`, `ring` (case-insensitive); an
explicitly-set unrecognized value throws rather than falling back
silently. Ring AllReduce additionally requires the element count to be
divisible by the world size; `auto` falls back to reference when a
tensor is large enough to want ring but isn't divisible, while a
forced `ring` request throws that requirement's error instead of
silently using reference.

**All ranks in a `World` must use identical `TBCCL_ALGORITHM`/
per-collective override values.** The current synchronous World
protocol has no channel to negotiate or detect a mismatched algorithm
choice across ranks, so a mismatch is unsupported and may deadlock.

Current provisional `auto` thresholds (every unlisted world size uses
reference):

```text
AllGather (threshold: contribution bytes per rank)
  N=3  : 16 MiB
  N=4  : 1 MiB
  N=8  : 512 KiB

ReduceScatter (threshold: output segment bytes per rank)
  N=2  : 512 KiB
  N=3  : 128 KiB
  N=4  : 512 KiB
  N=8  : 128 KiB

AllReduce (threshold: total tensor bytes per rank; N=2 and N=3
always use reference regardless of size)
  N=4  : 2 MiB
  N=8  : 512 KiB
```

These thresholds come from local-loopback and real Thunderbolt A/B
benchmark data gathered so far and are deliberately conservative (a
missed ring opportunity is preferred over a regression) — they are
**not** an API/ABI guarantee and may change between versions as
implementations and benchmark data improve. Applications that need a
deterministic implementation choice should use the explicit
overrides above rather than relying on the current thresholds.

### macOS Local Network permission

On macOS, `tb_pingpong` requires Local Network access.

If the client connects from Linux to macOS but stalls, or macOS returns
`No route to host` despite working ping/routing, check:

System Settings → Privacy & Security → Local Network

and allow `tb_pingpong`.

When developing over Remote SSH, macOS may not present the permission
dialog in the remote session. A local GUI session may be required to
approve it.

### Low-latency mode: `--busy-poll`

`tb_pingpong` supports Linux's `SO_BUSY_POLL` socket option to reduce
receive latency by spinning briefly on the NIC driver instead of
sleeping for an interrupt. It's disabled by default and only takes
effect on Linux; macOS ignores it.

```text
--busy-poll 0     normal blocking behavior, lowest CPU usage (default)
--busy-poll 100   current measured low-latency setting on Linux
```

Apply it on whichever socket is on the Linux side (server, client, or
both):

```bash
./build/tb_pingpong server --bind 192.168.3.2 --busy-poll 100
./build/tb_pingpong client --host 192.168.3.1 --busy-poll 100 --mode pingpong --sizes 64
```

Measured 64 B ping-pong RTT, Mac ↔ Linux over Thunderbolt:

```text
normal          ~146 µs
busy-poll 100   ~55 µs
```

This is hardware- and system-specific (CPU, NIC driver, kernel
scheduler); treat it as a starting point to tune for your own setup,
not a guaranteed result. Busy polling also spins the CPU on the
polling side for up to the configured duration on every blocking
socket call, so it trades CPU usage for latency — it does not
reduce bulk throughput, but check with `--mode stream` before relying
on it under load.