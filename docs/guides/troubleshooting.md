# Troubleshooting

Find your symptom, check the likely cause with the diagnostic command, apply the fix. Start with [diagnostics](diagnostics.md) (`tbccl-info` on every machine). The full list of error messages is in the [error reference](../reference/errors.md).

## Start here

### Work through it in this order

Each step is marked **Observe** (it reads state; nothing starts, stops or changes) or **Action** (you start, stop or configure a process, or run a test). No step changes network, firewall, driver or hardware settings. Stop at the first step that fails: later steps depend on it.

```{table}
:class: tb-stack tb-stack-steps

| Step | Question | Check | If it fails |
|---|---|---|---|
| 1 | Which versions are installed? | **Observe:** `tbccl-info --json` on **every** machine (the wire protocol must be equal) | [Diagnostics](diagnostics.md); [Bootstrap](#bootstrap) for `protocol_mismatch` |
| 2 | Is the device or backend you need present? | **Observe:** `tbccl-info` lists the providers (`cuda`, `metal_shared`) | [Installing and linking](#installing-and-linking) |
| 3 | Is the link ready? | **Observe:** `python3 thunderbolt_link_doctor.py --peer <peer-link-address>` (with `--peer` it also sends a ping and opens one TCP connection to the peer, and changes nothing; [how to get it](thunderbolt-troubleshooting.md)) | [Thunderbolt link troubleshooting](thunderbolt-troubleshooting.md) |
| 4 | Do the ranks agree on who is who? | **Observe:** read the rank, world size and communicator id that your launcher gives each rank | [Bootstrap](#bootstrap) |
| 5 | Does the transport come up? | **Action:** set `TBCCL_TRACE=1` on every rank and start them again (more logging, nothing else changes); look for a `timeout:` within `bootstrap_timeout` | [Bootstrap](#bootstrap), [ping works but the ranks time out](#ping-works-but-the-ranks-time-out) |
| 6 | Does the collective run? | **Observe:** the same collectives, sizes and order on every rank | [Collectives](#collectives) |
| 7 | Did a failure leave something behind? | **Observe:** `ss -ltnp` (Linux) or `lsof -nP -iTCP -sTCP:LISTEN` (macOS) for stale listeners | [After a failure](#after-a-failure) |
| 8 | Is it only slow? | **Observe:** the benchmark methodology | [Performance looks wrong](#performance-looks-wrong) |
```

### Find your symptom

```{table}
:class: tb-stack tb-stack-find

| You see | Go to |
|---|---|
| the build or link fails; `find_package(TBCCL)` fails | [Installing and linking](#installing-and-linking) |
| `protocol_mismatch ... wire protocol` | [Bootstrap](#bootstrap); every rank and adapter must be built against the same TBCCL ([versioning](../reference/versioning.md)) |
| `timeout: ... waiting for the connection(s)` | [Bootstrap](#bootstrap) |
| ping works, but the ranks time out | [Ping works but the ranks time out](#ping-works-but-the-ranks-time-out) |
| no interface, no carrier or no address on the link | [Thunderbolt link troubleshooting](thunderbolt-troubleshooting.md) |
| a collective hangs or the data are wrong | [Collectives](#collectives) |
| `unsupported:` for a memory kind or datatype | [Installing and linking](#installing-and-linking), [platform capabilities](../reference/platform-capabilities.md) |
| a process does not exit after an error | [After a failure](#after-a-failure) |
| it works but is slower than expected | [Performance looks wrong](#performance-looks-wrong) |
```

### Which layer owns the error

Many failures seen while using TBCCL are not caused by TBCCL. Decide from the first words of the message and from where the symptom starts.

```{table}
:class: tb-stack tb-stack-owner

| Layer | What it looks like | What to do |
|---|---|---|
| Native TBCCL | a message that starts with a result tag (`protocol_mismatch:`, `timeout:`, `unsupported:`, ...; [errors](../reference/errors.md)) | this guide; report with `tbccl-info --json` |
| An adapter | a message that starts with `torch-tbccl:`, a vLLM or Exo placement error naming the adapter | the adapter's own troubleshooting guide ([torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/guides/troubleshooting.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/guides/troubleshooting.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/guides/troubleshooting.html)) |
| PyTorch, vLLM, Exo, MLX | scheduling, executor, model or discovery behavior with no TBCCL frame in the stack | the framework's own project; the adapter guides name the known upstream behaviors |
| Operating system | a prompt or block (macOS Local Network, firewall), a port in use, permissions | [Thunderbolt link troubleshooting](thunderbolt-troubleshooting.md) entries 9 and 10; change settings yourself; the link doctor never does |
| Network configuration | wrong subnet, traffic leaving by Wi-Fi, an MTU that differs between the ends | entries 5 to 8 of the same guide |
| External dependencies | the CUDA driver or toolkit, Xcode, a compiler | their own installation notes; `tbccl-info` shows what was found |
```

## Installing and linking

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `find_package(TBCCL)` cannot find the package | the prefix is not on CMake's search path | `cmake -LA . \| grep TBCCL_DIR` | pass `-DCMAKE_PREFIX_PATH=<extracted archive or install prefix>` |
| link errors such as `undefined reference to operator delete`, `__cxa_throw`, `std::string` when linking a **C** program with pkg-config | the static libraries use the C++ runtime and `tbccl.pc` does not name it | `pkg-config --libs tbccl` | add `-lstdc++` (Linux) or `-lc++` (macOS), or link through CMake, which adds it |
| undefined `cudaEvent...`, `cudaStream...` when using the CUDA archive with pkg-config | the CUDA runtime is not on the link line | `tbccl-info` shows `cuda` | add `-L<cuda>/lib64 -lcudart`, or use CMake, which finds the CUDA Toolkit |
| `tbccl-info` shows no `cuda` provider on a CUDA machine | you installed the host-only archive or a host-only source build | `tbccl-info --json` | use `tbccl-<version>-linux-x86_64-cuda13.tar.gz` or build with `-DTBCCL_ENABLE_CUDA=ON` |
| `unsupported:` when moving a CUDA buffer | CUDA support was not registered in this process | check the build and that you called `tbcclRegisterCudaSupport()` (C) / `tbccl::register_cuda_support()` (C++) once | register once before creating the communicator |

## Bootstrap

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `protocol_mismatch: ... wire protocol ...` | two ranks, or a rank and an adapter, were built against different TBCCL wire protocols (3 versus 4) | `tbccl-info --json` on each machine; the adapter's `info` command | rebuild or reinstall every rank and adapter against the same TBCCL ([versioning](../reference/versioning.md)); wire 3 and wire 4 cannot interoperate |
| `protocol_mismatch: ... communicator` / `duplicate rank` / `rank outside the world` | ranks got different communicator ids, the same rank twice, or a wrong world size | print the id, rank and world size on each rank | share one communicator id and give each process a unique rank |
| `timeout: ... waiting for the connection(s)` | a peer never connected, or only some connections arrived | `ping`, `ss -ltn` on the peer, `TBCCL_TRACE=1` | start every rank; fix the advertised address; free the control and data ports |
| address already in use | a stale process, or the data port (control port + 1000) is taken | `ss -ltnp \| grep <port>` | stop the stale process or pick other ports (port 0 lets the kernel choose when you publish the actual endpoints) |
| macOS rank times out while Linux is fine and ping works | Local Network permission, the application firewall, or a `nohup` job outside a live session | the [Thunderbolt link guide](thunderbolt-link.md) | approve the permission locally, allow the application, keep Mac ranks in a live SSH session or `tmux`. Do not script firewall changes |

Bootstrap is bounded: it fails with one of these errors within `bootstrap_timeout` instead of hanging.

## Collectives

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| a collective hangs or corrupts data at world size 2 | ranks disagree on count or size; the two-rank path has no descriptor check | compare the arguments on both ranks | make every rank issue the same collectives with the same sizes, in the same order |
| `protocol_mismatch:` from a collective at world size 3 or more | ranks issued different collectives or different arguments | `TBCCL_TRACE=1` on all ranks | issue the same collectives in the same order |
| `unsupported:` naming a rank | that rank's memory kind or datatype cannot do the reduction (for example Float16 at world size above 2, or Int8 in Metal shared memory) | [platform capabilities](../reference/platform-capabilities.md) | use a supported datatype or world size |
| `unsupported: out-of-place all_reduce ... Cuda` | CUDA `all_reduce` needs the same buffer for send and receive | the call site | pass the same view for both |

## Ping works but the ranks time out

**Symptom.** `ping` answers in both directions, yet a rank reports `timeout: ... waiting for the connection(s)` or hangs at start.

**What that means.** `ping` proves an ICMP path between two addresses. TBCCL needs TCP connections between the ranks' control and data ports, in both directions, over the address each rank advertised. A firewall, a wrong advertised address or traffic leaving by another interface breaks the second without touching the first.

**Checks.** The route and environment commands are observations. The connection test is an **action**: it opens one TCP connection and closes it at once, which the listening side may log as a failed handshake, so run it only when no job you care about is starting. The link doctor's listener on a spare port (`--listen` / `--tcp-port`) tests the same path without touching a running job. Run them while the listening side is waiting.

```sh
# ACTION (opens and closes one TCP connection) - Can a TCP connection to the peer's control port be made? (<port> is the rank's control port; run from the other machine)
python3 -c "import socket; socket.create_connection(('<peer-link-address>', <port>), timeout=3).close(); print('TCP connect ok')"
# observe - Does traffic to the peer leave by the link you intend?
ip route get <peer-link-address>          # Linux
route -n get <peer-link-address>          # macOS
```

- `TCP connect ok`: the path works for that port. Check that the rank advertised the address you expect, and that the data port (control port + 1000) is also free and reachable.
- `timed out` (no answer): something drops the packets: a firewall, a wrong address, or a route that goes out another interface. The second command shows which interface is used.
- `Connection refused`: the host is reachable, nothing listens on that port: the rank is not running yet, or it listens on another address.

**Next.** [Thunderbolt link troubleshooting](thunderbolt-troubleshooting.md), entries 8 to 10, and the doctor's TCP mode (`--listen` / `--tcp-port`, [Two machines over Thunderbolt](../getting-started/two-machine-thunderbolt.md)). On macOS also check the Local Network prompt and that the rank runs in a live session. **Do not** change firewall rules, MTU or interface settings as a first step.

## After a failure

A failure after admission fails the communicator on every rank and there is no recovery: discard the communicator and create a new one ([failure handling](../concepts/failure-handling.md)). If a peer may still be finishing its last collective, quiesce all ranks before aborting or destroying.

## Performance looks wrong

Check the [benchmark methodology](../development/benchmark-methodology.md): never verify results inside a timed loop, and note that an idle gap between operations makes every thread handoff slower ([execution and progress](../concepts/execution-and-progress.md)). On a GPU benchmark check temperature and clock throttling first ([Thunderbolt link guide](thunderbolt-link.md)).

## When to ask for help

Escalate when the checks above isolate the layer and the layer is TBCCL, or when you cannot isolate it. Collect `tbccl-info --json` from every machine, the exact error text, `python3 tools/collect_diagnostics.py` output (it reports versions and interface names, never your files or secrets, and uploads nothing) and what each rank was doing. See [diagnostics](diagnostics.md) and [getting help](../project/getting-help.md). Security problems follow the [security policy](../project/security-policy.md).
