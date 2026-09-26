# tbccl


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