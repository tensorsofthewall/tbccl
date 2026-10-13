# Troubleshooting

Find your symptom, check the likely cause with the diagnostic command, apply the fix. Start with [diagnostics](diagnostics.md) (`tbccl-info` on every machine). The full list of error messages is in the [error reference](../reference/errors.md).

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

## After a failure

A failure after admission fails the communicator on every rank and there is no recovery: discard the communicator and create a new one ([failure handling](../concepts/failure-handling.md)). If a peer may still be finishing its last collective, quiesce all ranks before aborting or destroying.

## Performance looks wrong

Check the [benchmark methodology](../development/benchmark-methodology.md): never verify results inside a timed loop, and note that an idle gap between operations makes every thread handoff slower ([execution and progress](../concepts/execution-and-progress.md)). On a GPU benchmark check temperature and clock throttling first ([Thunderbolt link guide](thunderbolt-link.md)).
