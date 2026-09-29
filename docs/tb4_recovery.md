# TB4 diagnosis and deliberate recovery

Run the read-only snapshot first and retain its JSON outside tracked source:

```sh
scripts/tb4_health_snapshot.sh > results/phase20-local/health-before.json
```

The script discovers the controller from the kernel's Thunderbolt domain or
NHI-driver links, records resolved topology, vendor/device/class identity,
bridge siblings and power attributes, captures `lspci -D -tv`, interface counters,
Thunderbolt bus enumeration and relevant current-boot kernel messages. Missing
commands, devices and inaccessible values remain explicit. It performs no sudo
or writes to system files. On macOS the Python implementation records
`system_profiler SPThunderboltDataType -json`, `ifconfig bridge0` and boot identity.
The sweep streams that read-only query over SSH without installing a script.

Follow this order to identify the failure class:

1. Check the physical cable and device enumeration on both hosts. On macOS,
   inspect `system_profiler SPThunderboltDataType`; on Linux inspect
   `/sys/bus/thunderbolt/devices/` and the captured kernel messages.
2. Inspect current PCI topology and exact device identities. Addresses can
   change after reboot. The historical chain includes `0000:08:00.0` and
   downstream ports `0000:09:01.0` / `0000:09:03.0`; these are examples, not
   permanent identifiers. The historical `0000:00:1b.4` NVMe root port is separate.
3. Read **both** `power/control` and `power/runtime_status` for verified ports.
   `auto` is a policy; `suspended` is a runtime state. Neither alone establishes
   the cause of a missing network link. In Phase 20's initial snapshot both
   historical downstream ports were `auto/suspended` while the TB4 interface
   had carrier and zero RX/TX errors. Some bridge ports can be idle alongside a
   working network link. Do not change power policy on that observation alone.
4. Check `ip -br link`, `ip -br addr` and `ip -s link show thunderbolt0` on Linux;
   `ifconfig bridge0` on macOS. Missing physical enumeration and missing IP
   configuration are different failures. Confirm the expected addresses
   (`192.168.3.2`, `192.168.3.1`) and MTU 9000.
5. Check ping in both directions, bound to the expected source addresses.
   Do not infer application reachability merely from a successful ping.
6. Check the intended benchmark process, listener address/port, exit status
   and both endpoint logs. Confirm matching modes, sizes and iteration counts.
   Rank 1 connects to rank 0, independently of transfer source. Never kill a
   process simply because it occupies a desired port; choose another free port.
7. If Mac-side application accept/bootstrap fails while enumeration and ping
   work, inspect per-app firewall permissions. A kernel-ACKed TCP handshake
   does not establish that the application accepted the connection.

## Permission-gated runtime-power recovery

The Phase 19 recovery changed two verified downstream ports to `on`, followed by
manual cable reseating. Its historical success does not establish that the
power change alone caused recovery. Before considering it again, verify current
PCI topology, exact Intel bridge identity, target existence and runtime state,
and obtain explicit user authorization for the write. Stop if BDFs or identities
differ from the historical targets; do not blindly reuse the sudoers rule.

Only after those checks and authorization, the established narrow actions are:

```sh
# HISTORICAL TARGETS — verify identity and authorization before running.
echo on | sudo tee /sys/bus/pci/devices/0000:09:01.0/power/control
echo on | sudo tee /sys/bus/pci/devices/0000:09:03.0/power/control
```

The existing `/etc/sudoers.d/tbccl-tb-power` is deliberately limited to these
exact `tee` destinations. Do not broaden it, add generic root-writing helpers,
or make this a benchmark preflight or permanent power-management policy.

Cable reseating is a deliberate user-controlled step. Do not automatically
remove/rescan PCI devices, unload modules, reset controllers, change ASPM or
firmware, restart broad services or cycle power. Never use `pci=noaer` to hide
failures. A sleep/wake experiment requires explicit approval plus reliable local
or out-of-band access; do not suspend a machine over its only TB4 SSH connection.
Without those conditions record `NOT TESTED — safety/permission constraint`.

## macOS Application Firewall

Read-only inspection:

```sh
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
/usr/libexec/ApplicationFirewall/socketfilterfw --listapps
```

If a rebuilt listener is no longer accepted, and the user authorizes the
per-app change, use the **actual current absolute binary path**:

```sh
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /absolute/path/to/tbccl_tensor_transfer_bench
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /absolute/path/to/tbccl_tensor_transfer_bench
```

Do not disable the firewall globally. Recheck the registration after rebuilding
rather than interpreting an application block as physical enumeration failure.

## Validate before resuming

After authorized recovery, confirm enumeration, both interfaces/addresses,
bidirectional ping and a short verified application-level transfer. Capture a
new snapshot and compare within the same boot:

```sh
scripts/tb4_health_snapshot.sh > results/phase20-local/health-after.json
scripts/tb4_health_snapshot.sh --compare \
  results/phase20-local/health-before.json results/phase20-local/health-after.json
```

Inspect runtime-state changes, RX/TX errors/drops and new relevant kernel events.
If the boot changed, establish a new baseline; do not subtract old counts.
Stop testing on interface disappearance, fatal/uncorrectable PCIe errors,
repeated controller recovery failures, rapid correctable Thunderbolt error
growth or tensor corruption. Preserve logs and report the state before recovery.
