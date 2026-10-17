# Running on your hardware

The shipped examples never contain real addresses. You fill in your own and keep them out of version control.

## Two machines

1. **Build and stamp on each machine**, with the same TBCCL package, C ABI and wire protocol on both (the run is refused otherwise). Add `-DTBCCL_ENABLE_CUDA=ON` on a CUDA host and `-DTBCCL_ENABLE_METAL=ON` on a Mac for device memory. Then `python3 -m benchmarks.suite provenance stamp --build-dir BUILD`.
2. **Copy a profile** (`native.yaml` for host memory, `device.yaml` for CUDA and Metal) to `my-setup.local.yaml`. Files named `*.local.yaml` are git-ignored; anything else you pass with `--overlay FILE` is merged over the configuration.
3. **Fill the placeholders:** each host's link address and interface, the expected MTU, the remote build directory and the launcher prefix that reaches the second host (for example an `ssh` command; the suite does not assume ssh exists). Use the address of the link under test as `address`, and a *different* path to reach the host for management.
4. **Check without running anything:**

   ```sh
   python3 -m benchmarks.suite doctor my-setup.local.yaml --probe --allow-physical   # read-only: hardware, executables, versions, link health
   python3 -m benchmarks.suite run my-setup.local.yaml --dry-run                     # every command, ports not yet chosen
   ```

5. **Run a bounded measurement:** `python3 -m benchmarks.suite run my-setup.local.yaml --allow-physical`. Use the `limits` in the profile to cap the processes, bytes and time of a run.
6. **Report and compare** as in the [quickstart](quickstart.md).

## What the suite does for you on a real link

* Starts every remote benchmark process under a supervisor with a lease and heartbeats, so the process ends if the controller goes away, and checks afterwards that nothing it started is left. If cleanup cannot be confirmed the report says so.
* Refuses to launch over the data link: the management path must be a different interface, and the preflight checks it.
* Records the host hardware, the build identity (commit, executable hashes) and the TBCCL, C ABI and wire versions of each host, and refuses mismatched hosts.
* Takes read-only health snapshots (link state, interface counters, MTU, GPU temperature where available) before and between runs, and stops a run on a change.
* Never changes a firewall, a network setting or a power setting. Open the port range on both hosts yourself.

Read the [Thunderbolt link guide](../guides/thunderbolt-link.md) before you use a Thunderbolt cable, and the [benchmark suite guide](../development/benchmark-suite.md) for the full checklist.

## Hardware error visibility

A benchmark that reports "no errors" is only meaningful if errors would have been visible. On Linux, PCIe Advanced Error Reporting (AER) is disabled by the kernel parameter `pci=noaer`; then its counters stay at zero whatever the link does. `doctor --probe` reports AER as `observable`, `unavailable` or `unknown`, separately from link health, and reports say so. If AER is unavailable on your machine, the honest statement is *AER unavailable; no detected link-state or interface-health regression under the available checks*, not "zero errors".

## Without Git metadata

A source export (a tarball of the repository with no `.git`) works. The suite then has no commit to record: reports show the repository commit as unavailable instead of inventing one, and build provenance relies on the executable hashes and the TBCCL version, which come from the build itself.
