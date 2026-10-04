# Mac access and Thunderbolt 4 link reference

Operational reference for the two-machine development setup this
project is built across: a Linux machine and a Mac, connected both over
the normal network (for SSH) and a direct Thunderbolt 4 cable (the
actual subject of this library). For deep hardware-failure diagnosis
and recovery, see `docs/tb4_recovery.md`; for the controlled latency/
busy-poll characterization methodology, see `docs/tb4_busy_poll.md`.
This document is the quicker "how do I get to the Mac and is the link
healthy" reference.

## Machines

| | Linux | Mac |
|---|---|---|
| Hostname | `skynet` | — |
| Repo path | `/mnt/BigChonk/projects/tbccl` | `~/projects/tbccl` (i.e. `/Users/ragnarok/projects/tbccl`) |
| Login user | `svb` | `ragnarok` |
| Git commit author | `sandesh-bharadwaj` (differs from login user — use `svb` for anything OS-level) | — |
| SSH access | — | `ssh tbccl-mac` (configured alias for `ragnarok@<mac-host>`) |

Older per-phase clones may exist alongside the live Mac repo (e.g.
`~/projects/tbccl-phase12`) — don't confuse these with
`~/projects/tbccl`, which is the one to sync.

**Non-interactive SSH shells on the Mac use zsh with a minimal PATH** —
`cmake` and `ctest` are not found by bare name. Use the full path
(`/opt/homebrew/bin/cmake`, `/opt/homebrew/bin/ctest`) or
`export PATH=/opt/homebrew/bin:$PATH` first in the same command.
macOS also lacks GNU coreutils' `timeout`; use a background process
with an explicit stop, or a small `perl -e 'alarm N; exec @ARGV'`
wrapper, instead.

## Thunderbolt 4 link

Direct cable between the two machines, independent of the normal
network path used for SSH.

| | Linux | Mac |
|---|---|---|
| Interface | `thunderbolt0` | `bridge0` |
| Address | `192.168.3.2` | `192.168.3.1` |
| MTU | 9000 | 9000 |

Healthy round-trip ping is ~0.3-0.5ms. Quick check from Linux:

```sh
ping -c3 192.168.3.1
ip -br link show thunderbolt0
ip -br addr show thunderbolt0
```

Linux-side PCIe hardware chain (Intel Maple Ridge controller) — exact
Bus/Device/Function (BDF) addresses are **examples from this machine's
history, not permanent identifiers**; they can change after a reboot
or hardware change, so re-verify current topology (`lspci -tv`,
`/sys/bus/thunderbolt/devices/`) before trusting a cited address:

```
PCIe bridge 0000:08:00.0
  -> 0000:09:00.0 .. 09:03.0
    -> NHI 0000:0a:00.0 / USB controller 0000:25:00.0
```

The historical NVMe root port `0000:00:1b.4` is unrelated — don't
conflate its AER noise with the Thunderbolt bridge's.

## Syncing code between machines

This project does not use a shared filesystem between the two
machines — code is synced explicitly (`scp` individual files, or
`git push` + `git pull` once committed). Workflow used throughout this
project's history:

1. Make and test changes on Linux first.
2. `scp` the changed files to the Mac (or commit on Linux and `git
   pull` on the Mac once ready) — **always get explicit approval before
   syncing/rebuilding on the Mac**, same as for pushing to origin.
3. Reconfigure and rebuild on the Mac (`cmake .` in the existing build
   directory picks up new CMakeLists.txt changes; a fresh target needs
   `cmake --build . --target <name>`).
4. Run the Mac's test suite (`ctest --output-on-failure`) before
   considering the change synced.

Before `git pull` on the Mac, check `git status` first — files scp'd
directly during iterative development often leave the Mac's working
tree with local modifications/untracked files that would conflict with
a clean fast-forward pull. Clean them deliberately:

```sh
git checkout -- <files matching committed content>
git clean -fd <untracked files/dirs that are now part of the commit>
git pull origin main
```

**Known gotchas** (hit in earlier phases of this project, worth
avoiding again):

- `rsync` with multiple source files and one destination directory
  silently flattens subdirectory structure
  (`rsync a/foo.py b/bar.py host:~/dest/` lands both directly in
  `~/dest/`, not under their original subdirectories) — sync one
  destination subdirectory at a time, or use `scp` per-file as this
  project generally does.
- `git stash -u` followed by `git stash drop` (instead of `git stash
  pop`) permanently discards the stashed untracked content — this once
  silently deleted a generated `Makefile` inside an untracked build
  directory. Always `git stash show -p --include-untracked` or pop
  before dropping an untracked-inclusive stash.

## Link health check

Read-only, no sudo required:

```sh
python3 scripts/tb4_health_snapshot.py > /tmp/snapshot.json
```

Discovers the controller from the kernel's Thunderbolt domain / NHI
driver links; records PCI topology, vendor/device/class identity,
bridge power attributes, interface counters, Thunderbolt enumeration,
and relevant current-boot kernel messages. On macOS it instead records
`system_profiler SPThunderboltDataType -json`, `ifconfig bridge0`, and
boot identity. Nothing it does writes to system state.

AER (PCIe Advanced Error Reporting) correctable-error count for the
Thunderbolt bridge specifically (not the unrelated NVMe root port):

```python
import json
d = json.load(open("/tmp/snapshot.json"))
for dev in d["pci_devices"]:
    if dev["bdf"] == "0000:08:00.0":  # verify this BDF is still current first
        print(dev["aer"]["aer_dev_correctable"])
```

Capture a snapshot before and after any real-hardware TB4 session and
compare `TOTAL_ERR_COR`. There is a shell wrapper,
`scripts/tb4_health_snapshot.sh`, that streams the same read-only query
over SSH without installing anything on the remote side — useful for a
quick before/after pair without round-tripping files.

## Safety rules (standing, across every phase of this project)

- Check AER/link health before and after every real-TB4 hardware
  session.
- **One isolated correctable error matching the established historical
  pattern (the `Timeout` class, growing by ~1 occasionally across a
  long session with heavy TB4 traffic) does not require stopping** —
  document it and continue. This has recurred several times across this
  project's history without ever correlating with a real problem.
- **Stop immediately** and do not push through: any fatal or nonfatal
  (uncorrectable) PCIe error, rapid correctable-error growth, the
  Thunderbolt interface disappearing, or any sign of data corruption in
  a transfer. Report the state; don't attempt to work around it
  silently.
- Never use `pci=noaer` or any other mechanism to mask/hide hardware
  errors instead of understanding them.
- Never automatically remove/rescan PCI devices, unload kernel modules,
  reset the controller, change ASPM/firmware settings, restart broad
  system services, or cycle power as a "fix" — these require explicit
  user authorization, case by case, verified against *current* hardware
  identity (BDFs can change across reboots).
- A sleep/wake experiment on either machine requires explicit approval
  plus reliable local or out-of-band access — never suspend a machine
  over its only TB4/SSH connection.
- Real-TB4 corroboration testing should be focused and minimal (a few
  representative checks) — prefer local/loopback testing for exhaustive
  parameter sweeps, and reserve the real link for what genuinely needs
  real hardware.

## GPU (CUDA) thermal/load safety

Applies whenever a benchmark drives the real CUDA GPU on the Linux
machine, independent of whether TB4 is also involved:

```sh
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,clocks.current.sm,clocks_event_reasons.active,power.draw,pstate --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

- Check this before any **authoritative/timed** GPU benchmark group.
  `clocks_event_reasons.active` containing the `SW_THERMAL_SLOWDOWN`
  bit (and a collapsed SM clock, e.g. ~210MHz against a nominal
  ~1700MHz+) means performance numbers collected right now are
  **invalid** — wait for a clean, idle snapshot before trusting timing
  data. Correctness testing may continue regardless of thermal state.
- If an unrelated compute process is using the GPU, treat any timed
  result collected during that window as contaminated — **never kill
  someone else's process to free the GPU**; wait for it to finish, or
  find GPU-idle correctness-only work to do in the meantime.
- Sustained back-to-back GPU workloads with no cooldown can genuinely
  thermal-throttle the hardware (observed and documented in this
  project's history) — this is a real hardware finding worth reporting
  honestly when it happens, not a bug to hide or silently work around.

## Operational notes added in Phases 52-53

- **Live sessions only on macOS.** A process started with `nohup ... &` from an ssh session that then ended (an orphan) is not allowed to reach
  the LAN: TBCCL bootstrap fails with `TBCCL_TIMEOUT`. Keep the rank inside a live ssh connection (a backgrounded `ssh -tt tbccl-mac '...'` works) or run it
  inside `tmux` (installed on both hosts). The same holds for any server process such as an exo node.
- **Bootstrap blobs between hosts** are small files: have each rank write `<purpose>.<rank>` and sync the directory with `rsync` in a loop
  (`examples/c_link_probe.c`, `../exo-tbccl/examples/link_probe.py`). The link itself is not used for the exchange.
- **Shared model folder:** the Linux machine exposes `/mnt/win_hf_models` and the Mac mounts it at `~/Desktop/win_hf_models` (SMB). Copy a model to
  local disk on each host before using it (reads over the share were slow/fragile); verify with `sha256sum` / `shasum -a 256`.
- **Model locations (Phase 48-53).** Never download weights; use what exists:
  - Shared folder (Linux `/mnt/win_hf_models`, Mac `~/Desktop/win_hf_models`): `Qwen3-0.6B` (bf16), `Qwen3-0.6B-8bit` (`mlx-community/Qwen3-0.6B-8bit`, MLX format, works on MLX-CUDA on Linux too), `SmolLM-135M-Instruct`.
  - Local copies: Mac `~/phase48_models/Qwen3-0.6B` (vLLM Phase 48 scripts, `PHASE48_MODEL_LINUX` / `PHASE48_MODEL_MAC`); both hosts `~/.exo_p53/local_models/Qwen3-0.6B-8bit` (exo, Phase 53).
  - exo needs `EXO_OFFLINE=true` (and `HF_HUB_OFFLINE=1`) and a model card plus `model.safetensors.index.json`; its own test suite otherwise downloads tokenizer/config files (about 617 MB once).
    Details: `../exo-tbccl/AGENTS.md` ("Models").
- **Non-git copies** (for example `exo-tbccl` on the Mac) are synced by `tar` + `scp`; a copy of a git repo is synced with `git format-patch` / `git am`.
- **Remote shell writes** (rsync/scp/ssh commands that modify the Mac) may need explicit permission in the agent session.
- **Current baseline at the end of Phase 53:** AER Timeout 15, nonfatal 0, fatal 0, `thunderbolt0` up, MTU 9000, RTT 0.32-0.43 ms, tx drops 8 (constant).
  A one-line check: sum `Timeout` over `/sys/bus/pci/devices/*/aer_dev_correctable`, plus `aer_dev_nonfatal` / `aer_dev_fatal` totals, plus `ip -s link show thunderbolt0`.
- **GPU thermals:** the laptop RTX 3070 Ti reaches 83-88 C under sustained decode and reports `SW_THERMAL_SLOWDOWN`; interleave A/B runs and state the caveat.

## macOS Application Firewall

If a rebuilt listener binary stops being accepted (enumeration and
ping still work, but application-level connections stall or are
refused), check per-app Local Network / firewall permission rather than
assuming a physical link failure:

```sh
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
/usr/libexec/ApplicationFirewall/socketfilterfw --listapps
```

If the user authorizes it, allowlist using the binary's actual current
absolute path (it changes every rebuild if the build directory moves):

```sh
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /absolute/path/to/binary
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /absolute/path/to/binary
```

Never disable the firewall globally. Over Remote SSH, macOS may not
present the Local Network permission dialog in the remote session at
all — a local GUI session on the Mac may be required to approve it the
first time.

## `sudo` in automated sessions

Non-interactive sessions cannot answer a password prompt. Prefer asking
the user to run a privileged command themselves. If passwordless access
is genuinely needed repeatedly, ask for a **narrowly-scoped** `NOPASSWD`
sudoers rule for the exact command and arguments (never `NOPASSWD:
ALL`), referencing the real login username (`svb` on Linux, not the git
author name). Literal colons in a sudoers command argument (e.g. a PCI
address like `0000:09:01.0`) must be backslash-escaped
(`0000\:09\:01.0`), or `visudo -c` reports a syntax error.
