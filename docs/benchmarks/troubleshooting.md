# Troubleshooting

Start with `python3 -m benchmarks.suite doctor` (add your configuration and `--probe` for a multi-host setup). It is read-only and explains what is missing.

| Symptom | Likely cause and what to do |
|---|---|
| `executable ... not found` | Build the targets (`cmake --build build-release`) or point `build_dirs` / `$TBCCL_BUILD_DIR` at the build. The suite never builds anything. |
| a warning that the build is not optimised | You built Debug. Rebuild with `-DCMAKE_BUILD_TYPE=Release`. |
| the run is refused with unresolved `<PLACEHOLDER>` values | Fill them in a `*.local.yaml` file or an overlay (`--overlay FILE`). A dry run lists them. |
| a configuration with non-loopback hosts is refused | It is physical. Set `topology.physical: true` and pass `--allow-physical` once `doctor --probe` is clean. |
| the TBCCL, C ABI or wire versions of two hosts differ | Rebuild both from the same checkout. The run is refused on purpose. |
| a hang at start-up or a refused connection | A firewall is blocking the port range on one host. Open `defaults.port_range` yourself; the suite never changes a firewall. |
| the preflight says the launcher uses a data address | Reach the second host over a management path that is not the link under test (`management_address`). |
| `cleanup could NOT be confirmed` in a report | A supervised remote process could not be proven gone. Check that host with `ps` before the next run. |
| a point is `unsupported` | The build or the host lacks the backend you asked for (for example Metal on Linux). It is never replaced by host memory. |
| `providers` shows `environment missing` / `not configured` / `describe failed` | See [Framework benchmarks](frameworks.md). The native benchmarks are unaffected. |
| AER shown as `unavailable` | The kernel was booted with `pci=noaer` or the counters are not readable. Reports state this instead of "zero errors". Booting without `pci=noaer` is your decision. |
| results vary by 2x between process pairs at small sizes | Observed on some setups: small-message latency can be bimodal. Report the spread and use more repetitions; do not quote a single number. |
| `matplotlib is not installed: plots skipped` | Optional. Reports are written without plots. |

Include the output of `doctor`, the run's `manifest.json` and the failing record when you ask for help; read them first and remove anything private.
