#!/usr/bin/env python3
"""Read-only latency-relevant runtime-state snapshot for TB4 investigation.

Phase 30 Part C. Extends tb4_health_snapshot.py (reused for boot_id, AER,
PCI topology, interface counters -- not duplicated here) with the broader
set of state that can plausibly explain a change in steady-state latency
or tail-event reproducibility across a reboot: IRQ effective affinity,
irqbalance activity, CPU frequency/governor/EPP and cpuidle counters for
the CPUs currently servicing the NHI's MSI-X vectors, PCI/TB link speed
and width, NIC offloads/qdisc/RPS/XPS, thunderbolt/thunderbolt_net module
parameters, and thermal/load state.

Every value here is read via a plain file read or a read-only command
(ethtool -k, tc qdisc show, sysctl -n, etc.) -- nothing in this script
writes to sysfs, procfs, or any device. A field that genuinely cannot be
read on this system is recorded as null, never fabricated or omitted
silently (see compare_tb4_runtime_state.py, which must distinguish an
unavailable field from an unchanged one).
"""
import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb4_health_snapshot as health  # noqa: E402

PROC_INTERRUPTS = Path("/proc/interrupts")
CPU_SYSFS = Path("/sys/devices/system/cpu")


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command(argv, timeout=20):
    if not shutil.which(argv[0]):
        return {"command": argv, "returncode": None, "stdout": "", "stderr": "unavailable command"}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"command": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "returncode": None, "stdout": "", "stderr": str(error)}


def irq_table(interrupts_text, irq_numbers):
    """Parse /proc/interrupts for the given IRQ numbers' per-CPU counts and name."""
    result = {}
    for line in (interrupts_text or "").splitlines():
        fields = line.split()
        if not fields:
            continue
        label = fields[0].rstrip(":")
        if label not in {str(n) for n in irq_numbers}:
            continue
        counts = [int(f) for f in fields[1:] if f.isdigit()]
        rest = [f for f in fields[1:] if not f.isdigit()]
        result[label] = {"per_cpu_counts": counts, "descriptor": " ".join(rest)}
    return result


def irq_affinity(irq_numbers):
    result = {}
    for irq in irq_numbers:
        base = Path(f"/proc/irq/{irq}")
        result[str(irq)] = {
            "smp_affinity_list": read(base / "smp_affinity_list"),
            "effective_affinity_list": read(base / "effective_affinity_list"),
        }
    return result


def irqbalance_state():
    active = command(["systemctl", "is-active", "irqbalance"])
    pids = command(["pgrep", "-a", "irqbalance"])
    return {
        "systemctl_is_active": active["stdout"].strip() if active["returncode"] is not None else None,
        "process_listing": pids["stdout"].strip(),
        "running": bool(pids["stdout"].strip()),
    }


def cpu_state(cpu):
    base = CPU_SYSFS / f"cpu{cpu}"
    cpufreq = base / "cpufreq"
    cpuidle_states = {}
    for state_dir in sorted((base / "cpuidle").glob("state*")) if (base / "cpuidle").exists() else []:
        cpuidle_states[state_dir.name] = {
            "name": read(state_dir / "name"),
            "usage": read(state_dir / "usage"),
            "time": read(state_dir / "time"),
        }
    return {
        "cpu": cpu,
        "online": read(base / "online") if (base / "online").exists() else "always-on (cpu0 or boot-cpu)",
        "scaling_governor": read(cpufreq / "scaling_governor"),
        "scaling_cur_freq": read(cpufreq / "scaling_cur_freq"),
        "scaling_min_freq": read(cpufreq / "scaling_min_freq"),
        "scaling_max_freq": read(cpufreq / "scaling_max_freq"),
        "scaling_driver": read(cpufreq / "scaling_driver"),
        "energy_performance_preference": read(cpufreq / "energy_performance_preference"),
        "cpuidle_driver": read(CPU_SYSFS / "cpuidle" / "current_driver"),
        "cpuidle_states": cpuidle_states,
    }


def pci_link_state(bdfs):
    result = {}
    for bdf in bdfs:
        base = Path(f"/sys/bus/pci/devices/{bdf}")
        if not base.exists():
            result[bdf] = None
            continue
        result[bdf] = {
            "current_link_speed": read(base / "current_link_speed"),
            "current_link_width": read(base / "current_link_width"),
            "max_link_speed": read(base / "max_link_speed"),
            "max_link_width": read(base / "max_link_width"),
        }
    return result


def lspci_verbose(bdfs):
    result = {}
    for bdf in bdfs:
        out = command(["lspci", "-D", "-vv", "-s", bdf])
        result[bdf] = out["stdout"]
    return result


def module_parameters(module):
    base = Path(f"/sys/module/{module}/parameters")
    if not base.exists():
        return None
    return {p.name: read(p) for p in sorted(base.iterdir())}


def net_queue_masks(interface):
    base = Path(f"/sys/class/net/{interface}/queues")
    if not base.exists():
        return {}
    result = {}
    for q in sorted(base.glob("rx-*")):
        result[q.name] = {"rps_cpus": read(q / "rps_cpus")}
    for q in sorted(base.glob("tx-*")):
        result.setdefault(q.name, {})["xps_cpus"] = read(q / "xps_cpus")
    return result


def thermal_state():
    base = Path("/sys/class/thermal")
    zones = {}
    if base.exists():
        for zone in sorted(base.glob("thermal_zone*")):
            zones[zone.name] = {
                "type": read(zone / "type"),
                "temp_millideg": read(zone / "temp"),
            }
    return zones


def relevant_sysctls():
    names = [
        "net.core.busy_read",
        "net.core.busy_poll",
        "net.ipv4.tcp_timestamps",
        "net.ipv4.tcp_low_latency",
    ]
    result = {}
    for name in names:
        out = command(["sysctl", "-n", name])
        result[name] = out["stdout"].strip() if out["returncode"] == 0 else None
    return result


GPU_QUERY_FIELDS = [
    "timestamp", "name", "pstate", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "power.draw", "temperature.gpu",
    "pcie.link.gen.current", "pcie.link.width.current",
]


def gpu_state():
    """Phase 31 Part E/B: read-only nvidia-smi telemetry. Returns None (not
    an error) when nvidia-smi is unavailable -- e.g. on the Mac, or a Linux
    machine with no NVIDIA GPU -- so callers never treat "no GPU" as a
    capture failure. A field nvidia-smi itself doesn't support on this
    driver/GPU comes back as the literal string it prints ("[N/A]" or
    similar) rather than being silently dropped, so it is still visible in
    the raw csv text even though it can't be parsed as a number."""
    if not shutil.which("nvidia-smi"):
        return None
    fields = ",".join(GPU_QUERY_FIELDS)
    out = command(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    if out["returncode"] != 0:
        return {"available": False, "error": out["stderr"]}
    values = [v.strip() for v in out["stdout"].strip().split(",")]
    parsed = dict(zip(GPU_QUERY_FIELDS, values)) if len(values) == len(GPU_QUERY_FIELDS) else None
    return {"available": True, "raw_csv": out["stdout"].strip(), "fields": parsed}


def active_compute_processes():
    """nvidia-smi's own compute-app listing -- empty (not None) means no
    CUDA compute process is currently running, which is the actual signal
    Part B needs (a graphics-only client like Xorg/Firefox does not show
    up here at all)."""
    if not shutil.which("nvidia-smi"):
        return None
    out = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader"])
    if out["returncode"] != 0:
        return {"available": False, "error": out["stderr"]}
    processes = []
    for line in out["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            processes.append({"pid": parts[0], "process_name": parts[1], "used_memory": parts[2]})
    return {"available": True, "processes": processes}


def gpu_graphics_processes():
    """nvidia-smi's default full-text listing (compute + graphics clients),
    for context on what is actually using the GPU/VRAM when
    active_compute_processes() is empty."""
    if not shutil.which("nvidia-smi"):
        return None
    full = command(["nvidia-smi"])
    return {"returncode": full["returncode"], "full_text": full["stdout"]}


def discover_irqs(descriptor_pattern):
    """Live-discover IRQ numbers from /proc/interrupts whose descriptor
    column matches descriptor_pattern (a compiled regex) -- never assumes
    a fixed IRQ number, since NVMe in particular allocates one MSI-X
    vector per queue and the exact numbers vary by boot/topology."""
    result = {}
    for line in (read(PROC_INTERRUPTS) or "").splitlines():
        fields = line.split()
        if not fields:
            continue
        label = fields[0].rstrip(":")
        if not label.isdigit():
            continue
        rest = " ".join(f for f in fields[1:] if not f.isdigit())
        if descriptor_pattern.search(rest):
            counts = [int(f) for f in fields[1:] if f.isdigit()]
            result[label] = {"total_count": sum(counts), "descriptor": rest}
    return result


def workload_irq_counters():
    return {
        "nvidia": discover_irqs(re.compile(r"\bnvidia\b", re.I)),
        "nvme": discover_irqs(re.compile(r"\bnvme\d+q\d+\b", re.I)),
    }


def diskstats_snapshot():
    """Selected /proc/diskstats fields for NVMe devices only (reads, read
    sectors, writes, write sectors) -- read-only, no active disk I/O is
    triggered by this capture."""
    result = {}
    for line in (read("/proc/diskstats") or "").splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2]
        if not name.startswith("nvme"):
            continue
        result[name] = {
            "reads_completed": int(fields[3]),
            "sectors_read": int(fields[5]),
            "writes_completed": int(fields[7]),
            "sectors_written": int(fields[9]),
        }
    return result


def meminfo_snapshot():
    fields = {"MemAvailable", "Cached", "Dirty", "Writeback", "SwapFree"}
    result = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        parts = line.split(":")
        if len(parts) == 2 and parts[0].strip() in fields:
            result[parts[0].strip()] = parts[1].strip()
    return result


def psi_snapshot():
    result = {}
    for name in ("cpu", "io", "memory"):
        result[name] = read(f"/proc/pressure/{name}")
    return result


def linux_runtime_state(interface="thunderbolt0", irq_numbers=(177, 178)):
    base = health.snapshot()
    interrupts_text = read(PROC_INTERRUPTS) or ""
    irqs = irq_table(interrupts_text, irq_numbers)
    affinity = irq_affinity(irq_numbers)
    cpus_by_irq = {}
    for irq_str, aff in affinity.items():
        eff = aff.get("effective_affinity_list")
        cpus_by_irq[irq_str] = eff
    cpu_numbers = sorted({int(c) for c in cpus_by_irq.values() if c and c.isdigit()})
    tb_bdfs = [d["bdf"] for d in base.get("pci_devices", [])]

    base["uptime"] = command(["uptime"])["stdout"].strip()
    base["cmdline"] = read("/proc/cmdline")
    base["load"] = read("/proc/loadavg")
    base["irq_table"] = irqs
    base["irq_affinity"] = affinity
    base["irq_effective_cpu"] = {irq: (int(cpu) if cpu and cpu.isdigit() else None)
                                  for irq, cpu in cpus_by_irq.items()}
    base["irqbalance"] = irqbalance_state()
    base["cpu_state"] = {cpu: cpu_state(cpu) for cpu in cpu_numbers}
    base["pci_link_state"] = pci_link_state(tb_bdfs)
    base["lspci_verbose"] = lspci_verbose(tb_bdfs)
    base["module_parameters"] = {
        "thunderbolt": module_parameters("thunderbolt"),
        "thunderbolt_net": module_parameters("thunderbolt_net"),
    }
    base["ethtool_offloads"] = command(["ethtool", "-k", interface])["stdout"]
    base["ethtool_coalesce"] = command(["ethtool", "-c", interface])["stdout"]
    base["qdisc"] = command(["tc", "qdisc", "show", "dev", interface])["stdout"].strip()
    base["queue_masks"] = net_queue_masks(interface)
    base["thermal"] = thermal_state()
    base["sysctls"] = relevant_sysctls()
    base["gpu"] = gpu_state()
    base["active_compute_processes"] = active_compute_processes()
    base["gpu_processes"] = gpu_graphics_processes()
    base["workload_irq_counters"] = workload_irq_counters()
    base["diskstats"] = diskstats_snapshot()
    base["meminfo"] = meminfo_snapshot()
    base["psi"] = psi_snapshot()
    return base


def mac_runtime_state():
    base = health.snapshot()
    base["uptime"] = command(["uptime"])["stdout"].strip()
    tb_state = command(["system_profiler", "SPThunderboltDataType"])
    base["thunderbolt_detail"] = tb_state["stdout"]
    power = command(["pmset", "-g", "therm"])
    base["thermal_power"] = power["stdout"]
    return base


def snapshot(interface="thunderbolt0", irq_numbers=(177, 178)):
    if platform.system() == "Linux":
        return linux_runtime_state(interface=interface, irq_numbers=irq_numbers)
    if platform.system() == "Darwin":
        return mac_runtime_state()
    return {"error": "unsupported platform", "system": platform.system()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="thunderbolt0")
    parser.add_argument("--irq", type=int, nargs="*", default=[177, 178])
    parser.add_argument("--output", help="write JSON here instead of stdout")
    args = parser.parse_args()
    result = snapshot(interface=args.interface, irq_numbers=args.irq)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output} ({len(text)} bytes)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
