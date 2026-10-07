# Connecting a Mac and a Linux host over Thunderbolt 4

TBCCL carries ordinary TCP over whatever network path connects the ranks. A direct Thunderbolt 4 cable between a Mac and a Linux host gives a private, low-latency IP link. This guide describes the setup and the checks that were used when validating TBCCL on such a link. The addresses below are examples.

## The link

| | Linux | Mac |
|---|---|---|
| Interface | `thunderbolt0` (the `thunderbolt-net` driver) | `bridge0` |
| Example address | `192.168.3.2` | `192.168.3.1` |
| MTU | 9000 | 9000 |

A healthy round trip is about 0.3 to 0.5 ms. Check from Linux:

```sh
ping -c3 192.168.3.1
ip -br link show thunderbolt0
ip -br addr show thunderbolt0
ip -s link show thunderbolt0     # error and drop counters
```

On macOS use `ifconfig bridge0` and `system_profiler SPThunderboltDataType`. Use these addresses as the endpoints you publish when you bootstrap the communicators, with `bind_host` and `advertise_host` set to the link address ([C API bootstrap](../reference/c-abi-bootstrap.md)). The validated Linux host used an Intel Thunderbolt 4 (Maple Ridge) controller; other controllers have not been tested.

## macOS permissions

- **Local Network access.** The first time an application connects or listens on the local network, macOS asks for permission (System Settings, Privacy & Security, Local Network). If a Linux-to-Mac connection stalls or macOS reports `No route to host` although ping works, check this setting. Over Remote SSH the dialog may not appear; approve it from a local session.
- **Application firewall.** If bootstrap on the Mac fails while ping works, check the per-application firewall permission. A TCP handshake acknowledged by the kernel does not prove the application accepted the connection. Do not script changes to the macOS firewall.
- **Keep ranks in a live session.** A process started with `nohup` whose SSH session has ended cannot reach the local network on macOS and fails at bootstrap with a timeout. Run Mac ranks inside a live SSH session or `tmux`.

## Link health

Take a read-only snapshot before and after any real-link session and compare:

```sh
scripts/tb4_health_snapshot.sh > before.json      # streams the same read-only query over SSH
```

`scripts/tb4_health_snapshot.py` records the PCI topology of the Thunderbolt controller, interface counters, Thunderbolt enumeration, PCIe Advanced Error Reporting (AER) correctable and uncorrectable counters, and relevant kernel messages; on macOS it records `system_profiler SPThunderboltDataType` and `ifconfig bridge0`. It writes nothing to system state.

Practices that worked for validation:

- Stop immediately on any fatal or nonfatal (uncorrectable) PCIe error, rapidly growing correctable errors, the Thunderbolt interface disappearing, or any sign of data corruption.
- Do not hide hardware errors (for example with `pci=noaer`) instead of understanding them.
- Do not reset controllers, rescan PCI devices, unload modules or change power settings as an automatic fix. Verify the current hardware identity first; bus addresses can change across reboots.
- Prefer loopback tests for exhaustive sweeps and keep real-link runs small and focused.

If the link is missing, work through it in order: cable and device enumeration on both hosts (`/sys/bus/thunderbolt/devices/` on Linux), interface and address configuration, ping in both directions bound to the expected addresses, the listener address and port of your application, then per-application firewall permissions on the Mac.

## Latency: SO_BUSY_POLL

On Linux, `SO_BUSY_POLL` lets the receive path spin briefly on the network driver instead of sleeping for an interrupt. The `tb_pingpong` benchmark exposes it as `--busy-poll <us>` (default 0; no effect on macOS). In one measured setup the 64-byte round trip dropped from about 146 to about 55 microseconds with `--busy-poll 100`; this is hardware- and system-specific and costs CPU on the Linux receive path, so measure tail latency, CPU and bulk throughput before using it. `scripts/run_tb4_busy_poll_sweep.py` compares settings with raw samples and per-rank CPU accounting.

## GPU benchmarks

Before timing a GPU benchmark on the Linux host, check temperature and clock-throttle reasons (`nvidia-smi --query-gpu=temperature.gpu,clocks_event_reasons.active,clocks.current.sm --format=csv`). Numbers collected under thermal throttling are invalid; correctness testing may continue. Do not stop another user's process to free the GPU.
