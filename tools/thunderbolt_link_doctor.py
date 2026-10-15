#!/usr/bin/env python3
"""Read-only diagnostic for a direct Thunderbolt IP link between two computers.

Standard library only: it needs neither TBCCL, PyTorch, vLLM nor Exo. It looks at
the controller, the network interface, its addresses and MTU, the route to the
peer, ICMP and (optionally) a TCP connection, and prints what it found.

It never repairs anything: it does not change addresses, routes, MTU, firewall
rules or device authorization, does not load or reload kernel modules, does not
reset controllers and does not use elevated privileges. Where a check needs
privileges or a tool that is missing, it says so and moves on.

    python3 tools/thunderbolt_link_doctor.py --peer 192.168.77.1
    python3 tools/thunderbolt_link_doctor.py --peer 192.168.77.1 --tcp-port 50555 --json report.json

The TCP check needs a listener on the peer. Start the harmless echo listener
bundled here on that machine (it only accepts connections on the address you
give it and exits by itself):

    python3 tools/thunderbolt_link_doctor.py --listen 192.168.77.1 --tcp-port 50555

Exit status: 0 when no check failed (warnings allowed), 1 when a check failed,
2 for a usage error.
"""

import argparse
import ipaddress
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
SCHEMA = "tbccl-thunderbolt-link-doctor/1"
MAC_RE = re.compile(r"\b([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})\b")
LINK_LOCAL_RE = re.compile(r"\bfe80::[0-9a-fA-F:]+")
MAC_FIREWALL = "/usr/libexec/ApplicationFirewall/socketfilterfw"


class Host:
    """The few things the doctor asks of the machine; tests substitute their own."""

    def system(self):
        return platform.system()

    def machine(self):
        return platform.machine()

    def run(self, argv, timeout=15):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            return 127, "", "command not found: " + argv[0]
        except subprocess.TimeoutExpired:
            return 124, "", "timed out"
        except OSError as e:
            return 126, "", str(e)
        return p.returncode, p.stdout, p.stderr

    def read(self, path):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        except OSError:
            return None

    def listdir(self, path):
        try:
            return sorted(os.listdir(path))
        except OSError:
            return None

    def readlink(self, path):
        try:
            return os.readlink(path)
        except OSError:
            return None


class Report:
    def __init__(self):
        self.results = []

    def add(self, status, check, summary, detail=None, advice=None):
        self.results.append({"status": status, "check": check, "summary": summary, "detail": detail or [], "advice": advice})

    def failed(self):
        return any(r["status"] == FAIL for r in self.results)

    def render(self):
        out = []
        for r in self.results:
            out.append("[%s] %s: %s" % (r["status"], r["check"], r["summary"]))
            for d in r["detail"]:
                out.append("       " + d)
            if r["advice"]:
                out.append("       -> " + r["advice"])
        counts = {s: sum(1 for r in self.results if r["status"] == s) for s in (PASS, WARN, FAIL, INFO, SKIP)}
        out.append("")
        out.append("%d passed, %d warnings, %d failed, %d informational, %d skipped" % (counts[PASS], counts[WARN], counts[FAIL], counts[INFO], counts[SKIP]))
        if counts[FAIL] == 0 and counts[WARN] == 0:
            out.append("No problem found. This does not prove the link is healthy under load; see docs/guides/thunderbolt-troubleshooting.md.")
        out.append("This tool only reads state. It changed nothing on this machine.")
        return "\n".join(out)


def sanitize(text):
    text = MAC_RE.sub("xx:xx:xx:xx:xx:xx", text)
    return LINK_LOCAL_RE.sub("fe80::xxxx", text)


def sanitize_json(value):
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    return value


def json_report(report, facts):
    return sanitize_json({"schema": SCHEMA, "facts": facts, "results": report.results})


# ---------------------------------------------------------------- Linux


def _ip_json(host, argv):
    rc, out, _ = host.run(argv)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def linux_controllers(host, report, facts):
    base = "/sys/bus/thunderbolt/devices"
    names = host.listdir(base)
    if names is None:
        report.add(FAIL, "controller", "no Thunderbolt/USB4 bus is exposed by this kernel (%s is missing)" % base,
                   advice="Use a Linux kernel with USB4/Thunderbolt support (CONFIG_USB4) on a machine with a Thunderbolt or USB4 port.")
        return
    domains = [n for n in names if n.startswith("domain")]
    peers = [n for n in names if re.fullmatch(r"\d+-\d+(\.\d+)*", n) and not n.endswith("-0")]
    if not domains:
        report.add(FAIL, "controller", "the Thunderbolt bus exists but has no domain (no controller bound)",
                   advice="Check that the controller is enabled in firmware and appears in `lspci`.")
        return
    security = [host.read("%s/%s/security" % (base, d)) for d in domains]
    facts["tb_domains"] = len(domains)
    facts["tb_security"] = [s for s in security if s]
    report.add(PASS, "controller", "%d Thunderbolt/USB4 domain(s), security level: %s" % (len(domains), ", ".join(s or "unknown" for s in security)))
    if not peers:
        report.add(FAIL, "peer-device", "no remote Thunderbolt device is attached to the controller",
                   advice="Connect the cable directly between the two computers' Thunderbolt/USB4 ports. If the cable is connected, "
                          "the device may need authorization: see the Linux section of docs/getting-started/two-machine-thunderbolt.md.")
        return
    facts["tb_peers"] = len(peers)
    unauthorized = []
    labels = []
    for p in peers:
        authorized = host.read("%s/%s/authorized" % (base, p))
        name = host.read("%s/%s/device_name" % (base, p)) or "unnamed"
        labels.append("%s (%s)%s" % (p, name, "" if authorized in (None, "1", "2") else ", authorized=%s" % authorized))
        if authorized == "0":
            unauthorized.append(p)
    if unauthorized:
        report.add(WARN, "peer-device", "remote device present but not authorized: " + ", ".join(unauthorized),
                   advice="Authorize it with the mechanism your distribution provides (for example `boltctl`); this tool will not do it for you.")
    else:
        report.add(PASS, "peer-device", "remote Thunderbolt device attached", labels)


def linux_module(host, report):
    mods = host.read("/proc/modules") or ""
    if re.search(r"^thunderbolt_net\b", mods, re.M):
        report.add(PASS, "driver", "thunderbolt_net (Thunderbolt IP networking) is loaded")
    elif host.listdir("/sys/module/thunderbolt_net") is not None:
        report.add(PASS, "driver", "thunderbolt_net is built into the kernel")
    else:
        rc, _, _ = host.run(["modinfo", "thunderbolt_net"])
        if rc == 0:
            report.add(FAIL, "driver", "thunderbolt_net is available but not loaded",
                       advice="Load it yourself with `sudo modprobe thunderbolt_net` (this tool never loads modules).")
        else:
            report.add(FAIL, "driver", "thunderbolt_net is neither loaded nor installed",
                       advice="Your kernel lacks Thunderbolt IP networking (CONFIG_USB4_NET); use a kernel that includes it.")


def linux_candidates(host):
    out = []
    for name in host.listdir("/sys/class/net") or []:
        link = host.readlink("/sys/class/net/%s/device/driver" % name)
        if link and os.path.basename(link) == "thunderbolt-net":
            out.append(name)
    return out


def _addr_info(entries, name):
    for e in entries or []:
        if e.get("ifname") == name:
            return e
    return None


def linux_interface(host, report, facts, name, addrs, peer):
    sys_dir = "/sys/class/net/" + name
    carrier = host.read(sys_dir + "/carrier")
    oper = host.read(sys_dir + "/operstate")
    e = _addr_info(addrs, name) or {}
    mtu = e.get("mtu") or _int(host.read(sys_dir + "/mtu"))
    v4 = ["%s/%s" % (a["local"], a["prefixlen"]) for a in e.get("addr_info", []) if a.get("family") == "inet"]
    v6 = [a["local"] for a in e.get("addr_info", []) if a.get("family") == "inet6" and a["local"].lower().startswith("fe80")]
    facts["interface"] = {"os": "linux", "mtu": mtu, "operstate": oper, "carrier": carrier, "ipv4": v4, "ipv6_link_local": bool(v6)}
    if oper == "up" and carrier == "1":
        report.add(PASS, "interface", "%s is up with carrier" % name)
    elif carrier == "0" or oper in ("down", "lowerlayerdown"):
        report.add(FAIL, "interface", "%s exists but has no carrier (operstate=%s)" % (name, oper),
                   advice="The interface appears when the cable and the peer are present. Check the cable, the other computer's Thunderbolt interface, and device authorization.")
    else:
        report.add(WARN, "interface", "%s state is operstate=%s carrier=%s" % (name, oper, carrier))
    if v4:
        report.add(PASS, "address", "IPv4 address(es) on %s: %s" % (name, ", ".join(v4)))
    else:
        report.add(FAIL, "address", "%s has no IPv4 address" % name,
                   advice="Give it a manual address in a private subnet that is not used elsewhere (see the Linux section of the setup guide).")
    report.add(PASS if v6 else WARN, "ipv6-link-local", "IPv6 link-local address present" if v6 else "no IPv6 link-local address on %s" % name,
               advice=None if v6 else "Exo discovery uses IPv6 link-local multicast. Keep IPv6 enabled on this interface (ipv6.method link-local).")
    report_mtu(report, mtu)
    if peer:
        peer_ip = ipaddress.ip_address(peer)
        nets = [ipaddress.ip_interface(a).network for a in v4]
        if nets and not any(peer_ip in n for n in nets):
            report.add(FAIL, "subnet", "peer %s is not in the same subnet as %s" % (peer, ", ".join(v4)),
                       advice="Both computers need addresses in one subnet, for example 192.168.77.1/24 and 192.168.77.2/24.")
    scan_aer(host, report)
    scan_dmesg(host, report)


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def report_mtu(report, mtu):
    if mtu is None:
        return
    if mtu == 1500:
        report.add(INFO, "mtu", "MTU is 1500 (the default). This works; jumbo frames are optional.")
    elif mtu > 1500:
        report.add(INFO, "mtu", "MTU is %d. Jumbo frames only work if BOTH computers use the same value." % mtu)
    else:
        report.add(WARN, "mtu", "MTU is %d, below the usual 1500" % mtu)


def scan_aer(host, report):
    base = "/sys/bus/pci/devices"
    devices = host.listdir(base)
    if devices is None:
        report.add(SKIP, "pcie-errors", "PCI sysfs is not readable here")
        return
    found = checked = 0
    for dev in devices:
        if (host.read("%s/%s/class" % (base, dev)) or "").startswith("0x0c0340"):
            found += 1
            counts = {}
            for kind in ("correctable", "nonfatal", "fatal"):
                text = host.read("%s/%s/aer_dev_%s" % (base, dev, kind))
                if text is None:
                    continue
                m = re.search(r"TOTAL_ERR_\w+\s+(\d+)", text)
                if m:
                    counts[kind] = int(m.group(1))
            if not counts:
                continue
            checked += 1
            bad = {k: v for k, v in counts.items() if k != "correctable" and v}
            if bad:
                report.add(FAIL, "pcie-errors", "controller %s reports uncorrectable PCIe errors: %s" % (dev, bad),
                           advice="Stop heavy traffic, reseat or replace the cable, and capture `dmesg` before reconnecting. Do not disable error reporting.")
            elif counts.get("correctable"):
                report.add(WARN, "pcie-errors", "controller %s has %d correctable PCIe errors since boot" % (dev, counts["correctable"]),
                           advice="Run the doctor again after a while: steadily growing counts point at the cable or the port.")
            else:
                report.add(PASS, "pcie-errors", "controller %s reports no PCIe errors" % dev)
    if not found:
        report.add(SKIP, "pcie-errors", "no Thunderbolt/USB4 host controller found in PCI sysfs")
    elif not checked:
        report.add(SKIP, "pcie-errors", "AER counters are not exposed for the controller")


def scan_dmesg(host, report):
    rc, out, err = host.run(["dmesg"])
    if rc != 0:
        report.add(SKIP, "kernel-log", "kernel log is not readable without privileges (%s)" % (err.strip() or "denied"),
                   advice="Optional: rerun with permission to read the log, or check `journalctl -k` yourself.")
        return
    pat = re.compile(r"thunderbolt|usb4|AER|pcieport.*(error|fatal)", re.I)
    lines = [ln for ln in out.splitlines() if pat.search(ln) and re.search(r"error|fail|timeout|AER|disconnect|reset", ln, re.I)]
    if lines:
        report.add(WARN, "kernel-log", "%d recent Thunderbolt/PCIe warning line(s) in the kernel log" % len(lines), lines[-5:])
    else:
        report.add(PASS, "kernel-log", "no Thunderbolt/PCIe warnings in the kernel log")


def linux_route(host, report, name, peer):
    data = _ip_json(host, ["ip", "-j", "route", "get", peer])
    if not data:
        report.add(FAIL, "route", "no route to %s" % peer, advice="Configure an address in the peer's subnet on the Thunderbolt interface.")
        return None
    dev = data[0].get("dev")
    gw = data[0].get("gateway")
    if dev == name and not gw:
        report.add(PASS, "route", "traffic to %s leaves through %s directly" % (peer, dev))
    elif dev != name:
        report.add(FAIL, "route", "traffic to %s would leave through %s, not %s" % (peer, dev, name),
                   advice="Another network (VPN, overlay, Wi-Fi) covers that subnet. Choose an unused subnet for the Thunderbolt link.")
    else:
        report.add(WARN, "route", "traffic to %s goes through gateway %s" % (peer, gw))
    return dev


def run_linux(host, report, facts, peer, interface):
    linux_controllers(host, report, facts)
    linux_module(host, report)
    cands = [interface] if interface else linux_candidates(host)
    facts["candidate_interfaces"] = len(cands)
    if not cands:
        report.add(FAIL, "interface", "no Thunderbolt network interface (driver thunderbolt-net) exists",
                   advice="It is created when the cable is connected and the peer is authorized and ready. Check the controller results above.")
        return
    addrs = _ip_json(host, ["ip", "-j", "addr", "show"])
    if addrs is None:
        report.add(SKIP, "interface", "`ip -j` is unavailable; install iproute2 to enable interface checks")
        return
    name = cands[0]
    if len(cands) > 1:
        report.add(WARN, "interface", "several Thunderbolt interfaces: %s; examining %s (pick with --interface)" % (", ".join(cands), name))
    if peer:
        routed = linux_route(host, report, name, peer)
        if routed and not interface and routed in cands:
            name = routed
    linux_interface(host, report, facts, name, addrs, peer)


# ---------------------------------------------------------------- macOS


def parse_hardware_ports(text):
    ports, cur = [], None
    for line in text.splitlines():
        if line.startswith("Hardware Port:"):
            cur = {"port": line.split(":", 1)[1].strip()}
            ports.append(cur)
        elif cur is not None and line.startswith("Device:"):
            cur["device"] = line.split(":", 1)[1].strip()
    return [p for p in ports if p.get("device")]


def parse_ifconfig(text):
    info = {"ipv4": [], "link_local": False, "members": [], "status": None, "mtu": None}
    m = re.search(r"\bmtu (\d+)", text)
    if m:
        info["mtu"] = int(m.group(1))
    for m in re.finditer(r"^\s+inet (\d+\.\d+\.\d+\.\d+) netmask (0x[0-9a-f]+)", text, re.M):
        prefix = bin(int(m.group(2), 16)).count("1")
        info["ipv4"].append("%s/%d" % (m.group(1), prefix))
    info["link_local"] = bool(re.search(r"^\s+inet6 fe80::", text, re.M))
    info["members"] = re.findall(r"^\s+member: (\S+)", text, re.M)
    m = re.search(r"^\s+status: (\S+)", text, re.M)
    info["status"] = m.group(1) if m else None
    return info


def parse_route_interface(text):
    m = re.search(r"^\s*interface: (\S+)", text, re.M)
    return m.group(1) if m else None


def parse_thunderbolt_profile(text):
    connected = len(re.findall(r"Status: Device connected", text))
    return {"ports_connected": connected, "ports_total": len(re.findall(r"^\s+Status: ", text, re.M))}


def run_macos(host, report, facts, peer, interface):
    rc, out, err = host.run(["networksetup", "-listallhardwareports"])
    if rc != 0:
        report.add(SKIP, "services", "`networksetup` failed: %s" % (err.strip() or rc))
        ports = []
    else:
        ports = parse_hardware_ports(out)
    bridge = next((p for p in ports if p["port"] == "Thunderbolt Bridge"), None)
    tb_ports = [p for p in ports if re.match(r"Thunderbolt \d", p["port"])]
    facts["thunderbolt_ports"] = len(tb_ports)
    if not tb_ports and not bridge:
        report.add(FAIL, "controller", "this Mac lists no Thunderbolt ports",
                   advice="Use a Mac with Thunderbolt ports. Plain USB-C ports do not carry Thunderbolt networking.")
        return
    report.add(PASS, "controller", "%d Thunderbolt port(s) listed" % len(tb_ports))
    rc, out, _ = host.run(["system_profiler", "SPThunderboltDataType"], timeout=30)
    if rc == 0:
        prof = parse_thunderbolt_profile(out)
        facts["thunderbolt_ports_connected"] = prof["ports_connected"]
        if prof["ports_connected"]:
            report.add(PASS, "peer-device", "%d Thunderbolt port(s) report a connected device" % prof["ports_connected"])
        else:
            report.add(FAIL, "peer-device", "no Thunderbolt port reports a connected device",
                       advice="Connect the cable directly to the other computer's Thunderbolt/USB4 port, using a Thunderbolt-capable cable.")
    else:
        report.add(SKIP, "peer-device", "`system_profiler SPThunderboltDataType` unavailable")
    if not bridge and not interface:
        report.add(FAIL, "interface", "no 'Thunderbolt Bridge' network service/port exists",
                   advice="Add one in System Settings > Network (or run `networksetup -listallnetworkservices` to see what exists); see the macOS section of the setup guide.")
        return
    name = interface or bridge["device"]
    rc, out, err = host.run(["ifconfig", name])
    if rc != 0:
        report.add(FAIL, "interface", "`ifconfig %s` failed: %s" % (name, err.strip() or rc))
        return
    info = parse_ifconfig(out)
    facts["interface"] = {"os": "macos", "mtu": info["mtu"], "status": info["status"], "ipv4": info["ipv4"], "ipv6_link_local": info["link_local"], "members": len(info["members"])}
    if info["status"] == "active":
        report.add(PASS, "interface", "%s is active" % name)
    elif info["status"] == "inactive":
        report.add(FAIL, "interface", "%s is inactive (no link)" % name,
                   advice="No Thunderbolt port has a working link. Check the cable and that the other computer's interface is up.")
    else:
        report.add(WARN, "interface", "%s status is %s" % (name, info["status"] or "unknown"))
    if info["members"]:
        report.add(INFO, "bridge-members", "%s bridges: %s (the IP address belongs on the bridge, not on a member)" % (name, ", ".join(info["members"])))
    if info["ipv4"]:
        report.add(PASS, "address", "IPv4 address(es) on %s: %s" % (name, ", ".join(info["ipv4"])))
    else:
        report.add(FAIL, "address", "%s has no IPv4 address" % name,
                   advice="Configure the address manually in System Settings > Network > Thunderbolt Bridge > Details > TCP/IP.")
    report.add(PASS if info["link_local"] else WARN, "ipv6-link-local", "IPv6 link-local address present" if info["link_local"] else "no IPv6 link-local address on %s" % name,
               advice=None if info["link_local"] else "Exo discovery uses IPv6 link-local multicast; keep IPv6 on Automatic for this service.")
    report_mtu(report, info["mtu"])
    if peer:
        peer_ip = ipaddress.ip_address(peer)
        nets = [ipaddress.ip_interface(a).network for a in info["ipv4"]]
        if nets and not any(peer_ip in n for n in nets):
            report.add(FAIL, "subnet", "peer %s is not in the same subnet as %s" % (peer, ", ".join(info["ipv4"])),
                       advice="Both computers need addresses in one subnet, for example 192.168.77.1/24 and 192.168.77.2/24.")
        rc, out, _ = host.run(["route", "-n", "get", peer])
        routed = parse_route_interface(out) if rc == 0 else None
        if routed == name:
            report.add(PASS, "route", "traffic to %s leaves through %s" % (peer, routed))
        elif routed:
            report.add(FAIL, "route", "traffic to %s would leave through %s, not %s" % (peer, routed, name),
                       advice="Another network (VPN, Wi-Fi, Ethernet) covers that subnet. Choose an unused subnet for the Thunderbolt link.")
        else:
            report.add(FAIL, "route", "no route to %s" % peer)
    rc, out, _ = host.run([MAC_FIREWALL, "--getglobalstate"])
    if rc == 0:
        report.add(INFO, "firewall", out.strip().splitlines()[0] if out.strip() else "state unknown",
                   ["If connections hang until you approve a prompt, allow the specific program (Python) in System Settings > Network > Firewall; never turn the firewall off."])


# ---------------------------------------------------------------- reachability


def check_ping(host, report, peer, system):
    wait = ["-W", "2"] if system == "Linux" else ["-t", "5"]
    rc, out, err = host.run(["ping", "-c", "3"] + wait + [peer], timeout=20)
    if rc == 127:
        report.add(SKIP, "ping", "`ping` is not installed")
        return
    m = re.search(r"(\d+) (?:packets )?received", out)
    got = int(m.group(1)) if m else 0
    if got == 3:
        report.add(PASS, "ping", "3/3 replies from %s" % peer)
    elif got:
        report.add(WARN, "ping", "%d/3 replies from %s (packet loss)" % (got, peer),
                   advice="Loss on a direct cable suggests a marginal cable or port; see the troubleshooting guide.")
    else:
        report.add(FAIL, "ping", "no reply from %s" % peer,
                   advice="Run this doctor on the other computer too: ping may fail in only one direction, or a firewall may drop ICMP. TCP can still work.")


def tcp_check(peer, port, timeout=5.0, connect=socket.create_connection):
    token = ("tbccl-link-doctor-%d" % os.getpid()).encode()
    try:
        with connect((peer, port), timeout=timeout) as s:
            s.sendall(token)
            s.settimeout(timeout)
            got = b""
            while len(got) < len(token):
                chunk = s.recv(len(token) - len(got))
                if not chunk:
                    break
                got += chunk
    except OSError as e:
        return False, "%s: %s" % (type(e).__name__, e)
    return got == token, "echoed %d bytes" % len(got)


def check_tcp(report, peer, port):
    ok, detail = tcp_check(peer, port)
    if ok:
        report.add(PASS, "tcp", "TCP connect to %s:%d and echo succeeded" % (peer, port))
    else:
        report.add(FAIL, "tcp", "TCP to %s:%d failed (%s)" % (peer, port, detail),
                   advice="Make sure the listener runs on the peer (--listen <its Thunderbolt IP> --tcp-port %d). If ping works but this fails, a host firewall is the usual cause." % port)


def listen(bind, port, seconds):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(4)
    print("echo listener on %s:%d for up to %d s (Ctrl-C to stop); it only echoes the bytes it receives" % (bind, port, seconds), flush=True)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            srv.settimeout(max(0.1, min(1.0, end - time.monotonic())))
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(5)
                try:
                    data = conn.recv(4096)
                    conn.sendall(data)
                except OSError:
                    pass
                print("served a connection from %s" % addr[0], flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
    return 0


# ---------------------------------------------------------------- driver


def diagnose(host, peer=None, tcp_port=None, interface=None, tcp=check_tcp):
    report, facts = Report(), {}
    system = host.system()
    facts["os"] = system
    facts["arch"] = host.machine()
    if system == "Linux":
        report.add(INFO, "platform", "Linux %s" % host.machine())
        run_linux(host, report, facts, peer, interface)
    elif system == "Darwin":
        report.add(INFO, "platform", "macOS %s" % host.machine())
        run_macos(host, report, facts, peer, interface)
    else:
        report.add(FAIL, "platform", "%s is not supported by this tool (Linux and macOS only)" % system)
        return report, facts
    if peer:
        check_ping(host, report, peer, system)
        if tcp_port:
            tcp(report, peer, tcp_port)
        else:
            report.add(SKIP, "tcp", "no --tcp-port given; ping alone does not prove TCP works",
                       advice="Start the bundled listener on the peer and pass --tcp-port.")
    else:
        report.add(SKIP, "reachability", "no --peer given; only local state was inspected")
    return report, facts


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Read-only diagnostic for a direct Thunderbolt IP link.")
    ap.add_argument("--peer", help="the other computer's Thunderbolt IPv4 address")
    ap.add_argument("--interface", help="inspect this interface instead of detecting one")
    ap.add_argument("--tcp-port", type=int, help="TCP port of a listener on the peer (or the port to listen on with --listen)")
    ap.add_argument("--listen", metavar="BIND_IP", help="run the harmless echo listener on this local address instead of diagnosing")
    ap.add_argument("--listen-seconds", type=int, default=120)
    ap.add_argument("--json", metavar="FILE", help="also write a sanitized JSON report")
    args = ap.parse_args(argv)
    for opt in ("peer", "listen"):
        value = getattr(args, opt)
        if value:
            try:
                ipaddress.IPv4Address(value)
            except ValueError:
                ap.error("--%s must be an IPv4 address, got %r" % (opt, value))
    if args.listen and not args.tcp_port:
        ap.error("--listen needs --tcp-port")
    return args


def main(argv=None, host=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.listen:
        return listen(args.listen, args.tcp_port, args.listen_seconds)
    report, facts = diagnose(host or Host(), args.peer, args.tcp_port, args.interface)
    print(report.render())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(json_report(report, facts), f, indent=2)
        print("sanitized report written to %s" % args.json)
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
