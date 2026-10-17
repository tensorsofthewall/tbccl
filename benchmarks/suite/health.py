"""Per-host link and GPU health: independent read-only snapshots, explicit signals, before/after regression detection.

Nothing here changes the system. A signal is `ok`, `bad` or `unavailable`; an unavailable critical signal is reported as such and is a preflight failure for a
physical run unless the host's configuration explicitly allows it (`expect.allow_unavailable`). Health is never assumed.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from . import inventory
from .remote import agent, session

HEALTH_SCRIPT = inventory.REPO_ROOT / "scripts" / "tb4_health_snapshot.py"
CRITICAL = ("interface_present", "link_state", "ip_configured", "boot_identity")
DOWN_STATES = ("down", "lowerlayerdown", "notpresent", "dormant", "inactive")
# NVML clocks-throttle-reason bits: 0x1 GPU idle, 0x2 applications clocks setting, 0x4 software power cap, 0x8 hardware slowdown, 0x10 sync boost,
# 0x20 software thermal slowdown, 0x40 hardware thermal slowdown, 0x80 hardware power-brake slowdown, 0x100 display clock setting.
THROTTLE_BAD_BITS = {0x8: "hardware slowdown", 0x20: "software thermal slowdown", 0x40: "hardware thermal slowdown", 0x80: "hardware power-brake slowdown"}
THROTTLE_NOTE_BITS = {0x4: "software power cap"}


def throttle_findings(reasons):
    """(bad, notes) decoded from nvidia-smi's throttle-reason bitmask ('0x0000000000000001'). Idle and 'not active' are normal. Text that cannot be decoded is bad:
    health is never assumed."""
    text = str(reasons or "").strip().lower()
    if text in ("", "not active", "none"):
        return [], []
    try:
        mask = int(text, 16) if text.startswith("0x") else int(text)
    except ValueError:
        return [f"unreadable throttle reasons {text!r}"], []
    return [name for bit, name in THROTTLE_BAD_BITS.items() if mask & bit], [name for bit, name in THROTTLE_NOTE_BITS.items() if mask & bit]
DEFAULT_GPU_MAX_C = 83.0

AER_STATES = ("observable", "unavailable", "unknown")
AER_UNAVAILABLE_STATEMENT = ("AER unavailable; no detected link-state or interface-health regression under the available checks. "
                             "The absence of kernel-reported AER events is not evidence that there were no PCIe errors.")


def kernel_cmdline_tokens(text):
    """Whitespace-separated kernel parameters; the comma-separated options of 'pci=' are returned individually as 'pci=<option>'."""
    tokens = []
    for token in str(text or "").split():
        if token.startswith("pci=") and "," in token:
            tokens.extend("pci=" + option for option in token[4:].split(",") if option)
        else:
            tokens.append(token)
    return tokens


def local_cmdline(path="/proc/cmdline"):
    """The effective kernel command line of this machine (Linux), or None."""
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def cmdline_from_snapshot(snapshot):
    """The kernel command line recorded in a health snapshot's kernel log ('Command line: ...'), or None."""
    for event in (snapshot or {}).get("kernel_events") or []:
        message = str(event.get("message") or "")
        if message.startswith("Command line:"):
            return message[len("Command line:"):].strip()
    return None


def aer_observability(cmdline=None, snapshot=None, platform_name="Linux"):
    """Whether PCIe Advanced Error Reporting could have reported anything on a host. Read-only and separate from health: 'unavailable' says nothing about whether errors
    occurred, only that kernel-reported AER counters and events cannot show them.

    states: observable (command line known, AER not disabled on it, AER counters readable), unavailable (disabled with pci=noaer, counters missing, or not Linux),
    unknown (the command line could not be read). `cmdline` is the effective kernel command line (for the local host /proc/cmdline); without it the one recorded in
    `snapshot` is used. Nothing here enables or disables AER."""
    source = "argument" if cmdline is not None else None
    if cmdline is None:
        cmdline = cmdline_from_snapshot(snapshot)
        source = "snapshot kernel log" if cmdline is not None else None
    system = (snapshot or {}).get("system") or platform_name
    if str(system).lower() not in ("linux", ""):
        return {"state": "unavailable", "cmdline_source": source, "pci_noaer": None, "counters_readable": False,
                "detail": f"{system} exposes no PCIe AER counters to this suite", "statement": AER_UNAVAILABLE_STATEMENT}
    devices = (snapshot or {}).get("pci_devices")
    readable = None
    if devices is not None:
        readable = any(isinstance(d.get("aer"), dict) and any(v for v in d["aer"].values()) for d in devices)
    if cmdline is None:
        return {"state": "unknown", "cmdline_source": None, "pci_noaer": None, "counters_readable": readable,
                "detail": "the kernel command line could not be read, so it is not known whether AER reporting is enabled",
                "statement": "AER observability unknown; absent AER counters or events must not be reported as 'no PCIe errors'."}
    tokens = kernel_cmdline_tokens(cmdline)
    if "pci=noaer" in tokens:
        return {"state": "unavailable", "cmdline_source": source, "pci_noaer": True, "counters_readable": readable,
                "detail": "the kernel was booted with pci=noaer: AER reporting is disabled, so AER counters stay at zero and no AER events are logged whatever the link does",
                "statement": AER_UNAVAILABLE_STATEMENT}
    if readable is False:
        return {"state": "unavailable", "cmdline_source": source, "pci_noaer": False, "counters_readable": False,
                "detail": "AER is not disabled on the kernel command line but no AER counters could be read", "statement": AER_UNAVAILABLE_STATEMENT}
    return {"state": "observable", "cmdline_source": source, "pci_noaer": False, "counters_readable": readable,
            "detail": "AER is not disabled on the kernel command line" + ("" if readable is None else " and AER counters are readable"),
            "statement": "AER events and counters are observable on this host."}


def _load_script_module():
    spec = importlib.util.spec_from_file_location("tb4_health_snapshot", HEALTH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_host(host, busy=()):
    """A wrapper dict: {'interface', 'gpu', 'snapshot', 'snapshot_error'}. Raises RuntimeError when the host cannot be asked at all."""
    if host.health_command:
        argv = list(host.health_command)
        if not host.is_local:
            prefix = list(host.launcher["argv"])
            argv = prefix + ([subprocess.list2cmdline(argv)] if host.launcher.get("shell_join", True) else argv)
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        if done.returncode not in (0, 1) or not done.stdout.strip():
            raise RuntimeError(f"health command failed on '{host.id}' (exit {done.returncode}): {done.stderr.strip()[:200]}")
        return {"interface": None, "gpu": None, "snapshot": json.loads(done.stdout), "snapshot_error": None, "custom_command": True}
    if host.is_local and not host.launcher.get("supervised"):
        return agent.health(str(HEALTH_SCRIPT), host.interface, busy)
    if not host.health_script:
        raise RuntimeError(f"host '{host.id}' is remote and has neither health_command nor health_script; its link health cannot be monitored")
    data, error = session.run_agent_json(host, ["health", "--script", host.health_script] + (["--interface", host.interface] if host.interface else [])
                                         + [a for name in busy for a in ("--busy", name)], timeout=90, python=host.python)
    if data is None:
        raise RuntimeError(f"health snapshot failed on '{host.id}': {error}")
    return data


def _sig(status, detail):
    return {"status": status, "detail": detail}


def evaluate(host, wrapper, needs_gpu=False):
    """Signals from one snapshot wrapper."""
    exp = host.expect or {}
    s = {}
    iface = wrapper.get("interface")
    snap = wrapper.get("snapshot") or {}
    if not host.interface:
        for name in ("interface_present", "link_state", "ip_configured"):
            s[name] = _sig("unavailable", "no interface is configured for this host")
    elif not iface:
        for name in ("interface_present", "link_state", "ip_configured"):
            s[name] = _sig("unavailable", "no interface information (custom health command)")
    else:
        s["interface_present"] = _sig("ok" if iface.get("present") else "bad", f"{host.interface} {'present' if iface.get('present') else 'absent'}")
        state = (iface.get("operstate") or "").lower()
        carrier = str((snap.get("interface") or {}).get("carrier") or "")
        if not iface.get("present"):
            s["link_state"] = _sig("bad", "interface absent")
        elif state in DOWN_STATES or carrier == "0":
            s["link_state"] = _sig("bad", f"operstate '{state}', carrier '{carrier}'")
        elif state in ("up", "active") or carrier == "1":
            s["link_state"] = _sig("ok", f"operstate '{state}'")
        else:
            s["link_state"] = _sig("unavailable", f"operstate '{state or 'unknown'}' is not conclusive")
        addrs = [a.split("/")[0] for a in iface.get("addresses") or []]
        want = host.interfaces.get(host.interface) or host.address
        if not addrs:
            s["ip_configured"] = _sig("bad" if iface.get("present") else "unavailable", "no IP address on the interface")
        elif want and not want.startswith("<") and want not in addrs:
            s["ip_configured"] = _sig("bad", f"expected address is not configured (found {len(addrs)} other address(es))")
        else:
            s["ip_configured"] = _sig("ok", f"{len(addrs)} address(es)")
        if exp.get("mtu") is not None:
            have = iface.get("mtu")
            s["mtu"] = _sig("unavailable", "MTU unreadable") if have is None else _sig("ok" if have == exp["mtu"] else "bad", f"mtu {have}, expected {exp['mtu']}")
        else:
            s["mtu"] = _sig("ok", f"mtu {iface.get('mtu')} (no expectation configured)")
    s["boot_identity"] = _sig("ok", "boot id read") if snap.get("boot_id") else _sig("unavailable", "boot identity not available")
    counters = (snap.get("interface") or {}).get("counters")
    s["error_counters"] = _sig("ok", "interface counters read") if counters and any(v is not None for v in counters.values()) else _sig("unavailable", "interface error counters not available")
    if snap or wrapper.get("interface"):
        aer = aer_observability(snapshot=snap)
        s["aer_observability"] = _sig("ok" if aer["state"] == "observable" else "unavailable", f"AER {aer['state']}: {aer['detail']}")
    if snap.get("kernel_log_available"):
        s["thunderbolt_pcie_errors"] = _sig("ok", "kernel log readable")
    else:
        s["thunderbolt_pcie_errors"] = _sig("unavailable", "Thunderbolt/PCIe error events cannot be read on this host (kernel log unavailable or not Linux)")
    system = wrapper.get("system")
    if system and system.get("loadavg") and system.get("cpu_count"):
        limit = exp.get("max_load_fraction", 0.75)
        frac = system["loadavg"][0] / system["cpu_count"]
        s["system_load"] = _sig("bad" if frac > limit else "ok", f"1-minute load {system['loadavg'][0]:.2f} on {system['cpu_count']} CPUs ({frac:.2f} of capacity, limit {limit:g})")
    if wrapper.get("competing") is not None:
        s["competing_processes"] = _sig("bad" if wrapper["competing"] else "ok", f"benchmark processes already running: pids {wrapper['competing']}" if wrapper["competing"] else "no competing benchmark process")
    if needs_gpu:
        gpu = wrapper.get("gpu") or {}
        if not gpu.get("available"):
            s["gpu_available"] = _sig("unavailable", gpu.get("reason") or "no GPU information")
        else:
            s["gpu_available"] = _sig("ok", f"{len(gpu['gpus'])} GPU(s)")
            limit = exp.get("gpu_max_temp_c", DEFAULT_GPU_MAX_C)
            hot = [g["name"] for g in gpu["gpus"] if g.get("temperature_c") is not None and g["temperature_c"] > limit]
            findings = {g["name"]: throttle_findings(g.get("throttle_reasons")) for g in gpu["gpus"]}
            throttled = [f"{n}: {', '.join(bad)}" for n, (bad, _) in findings.items() if bad]
            notes = sorted({note for _, ns in findings.values() for note in ns})
            temps = ", ".join(f"{g['temperature_c']:g} C" for g in gpu["gpus"] if g.get("temperature_c") is not None)
            if hot or throttled:
                s["gpu_thermal"] = _sig("bad", f"above {limit:g} C or throttling: {', '.join(hot + throttled)}")
            else:
                s["gpu_thermal"] = _sig("ok", f"{temps or 'temperature unknown'}, no thermal or hardware-slowdown reason active" + (f" (note: {', '.join(notes)})" if notes else ""))
    return s


def preflight_problems(host, signals):
    """(failures, warnings) for a baseline. Critical unavailable signals fail unless the host explicitly allows them."""
    allowed = set((host.expect or {}).get("allow_unavailable") or [])
    failures, warnings = [], []
    for name, sig in signals.items():
        if sig["status"] == "bad":
            failures.append(f"{host.id}: {name}: {sig['detail']}")
        elif sig["status"] == "unavailable":
            (warnings if name not in CRITICAL or name in allowed else failures).append(f"{host.id}: {name} unavailable: {sig['detail']}" + ("" if name not in CRITICAL or name in allowed else " (critical; allow it explicitly with expect.allow_unavailable)"))
    return failures, warnings


def regressions(host, before, after, needs_gpu=False):
    """Reasons a measurement taken between two snapshots cannot be trusted (own checks; the script's stop policy is applied by the caller)."""
    reasons = []
    bi, ai = before.get("interface"), after.get("interface")
    if bi and ai:
        if bi.get("present") and not ai.get("present"):
            reasons.append("interface disappeared")
        if (bi.get("operstate") or "") != (ai.get("operstate") or "") and (ai.get("operstate") or "").lower() in DOWN_STATES:
            reasons.append(f"link state changed to '{ai.get('operstate')}'")
        if set(bi.get("addresses") or []) - set(ai.get("addresses") or []):
            reasons.append("an IP address was removed from the interface")
        if bi.get("mtu") is not None and ai.get("mtu") is not None and bi["mtu"] != ai["mtu"]:
            reasons.append(f"MTU changed from {bi['mtu']} to {ai['mtu']}")
    bs, as_ = before.get("snapshot") or {}, after.get("snapshot") or {}
    if bs.get("boot_id") and as_.get("boot_id") and bs["boot_id"] != as_["boot_id"]:
        reasons.append("boot identity changed (the host rebooted)")
    bc, ac = (bs.get("interface") or {}).get("counters") or {}, (as_.get("interface") or {}).get("counters") or {}
    for key in ("rx_errors", "tx_errors"):
        try:
            if int(ac.get(key)) > int(bc.get(key)):
                reasons.append(f"interface {key} increased")
        except (TypeError, ValueError):
            pass
    for key in ("rx_bytes", "tx_bytes", "rx_errors", "tx_errors"):
        try:
            if int(ac.get(key)) < int(bc.get(key)):
                reasons.append(f"interface counter {key} went backwards (counters were reset)")
                break
        except (TypeError, ValueError):
            pass
    if needs_gpu and (after.get("gpu") or {}).get("available"):
        sig = evaluate(host, after, True).get("gpu_thermal")
        if sig and sig["status"] == "bad":
            reasons.append(f"GPU thermal/throttle condition: {sig['detail']}")
    return reasons


def compare_wrapper(before, after, script_compare=None, strict=False):
    """Combine the script's stop policy (on the inner snapshots) with this module's checks. Returns {'stop_reasons': [...], 'script': {...}}.

    strict: the script only stops on fatal/uncorrectable growth or rapid correctable growth; for a physical campaign on a link that has shown BadDLLP and fatal AER
    errors, ANY growth of any AER counter on the Thunderbolt chain, and any new Thunderbolt kernel event, is a stop condition."""
    reasons, script = [], None
    if before.get("snapshot") and after.get("snapshot"):
        script_compare = script_compare or _load_script_module().compare
        script = script_compare(before["snapshot"], after["snapshot"])
        reasons += list(script.get("stop_reasons", []))
        if strict:
            grown = {k: v for k, v in (script.get("aer_counter_deltas") or {}).items() if v > 0}
            if grown:
                reasons.append("new AER events on the Thunderbolt PCIe chain: " + ", ".join(f"{k} +{v}" for k, v in sorted(grown.items())))
            events = [e for e in script.get("new_kernel_events", []) if e.get("source") == "thunderbolt"]
            if events:
                reasons.append(f"{len(events)} new Thunderbolt kernel event(s), first: {events[0].get('message', '')[:120]}")
            if script.get("new_tb_timeout") and not any("Timeout" in r for r in reasons):
                reasons.append("new Thunderbolt bridge AER Timeout")
            reasons[:] = list(dict.fromkeys(reasons))
    elif before.get("snapshot_error") or after.get("snapshot_error"):
        reasons.append(f"snapshot unavailable ({after.get('snapshot_error') or before.get('snapshot_error')})")
    return {"stop_reasons": reasons, "script": script}


def link_delta(before, after):
    """Byte/error/retransmission deltas between two snapshot wrappers of one host (data interface and TCP). Counters are never reset or edited."""
    def counters(w):
        return ((w.get("snapshot") or {}).get("interface") or {}).get("counters") or {}

    def num(v):
        return int(v) if v is not None and str(v).lstrip("-").isdigit() else None

    out = {}
    b, a = counters(before), counters(after)
    for key in ("rx_bytes", "tx_bytes", "rx_errors", "tx_errors", "rx_dropped", "tx_dropped", "drops"):
        if num(a.get(key)) is not None and num(b.get(key)) is not None:
            out[key] = num(a[key]) - num(b[key])
    br, ar = (before.get("system") or {}).get("tcp_retransmitted_segments"), (after.get("system") or {}).get("tcp_retransmitted_segments")
    if isinstance(br, int) and isinstance(ar, int):
        out["tcp_retransmitted_segments"] = ar - br
    if (after.get("system") or {}).get("loadavg"):
        out["load_1m_after"] = after["system"]["loadavg"][0]
    return out
