#!/usr/bin/env python3
"""tools/thunderbolt_link_doctor.py against sanitized captures from real machines; no hardware, no network changes."""
import importlib.util
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "thunderbolt"
spec = importlib.util.spec_from_file_location("thunderbolt_link_doctor", ROOT / "tools" / "thunderbolt_link_doctor.py")
doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc)


def fx(name):
    return (FIX / name).read_text()


class FakeHost:
    def __init__(self, system, files=None, links=None, commands=None, machine="x86_64"):
        self._system, self._machine = system, machine
        self.files = dict(files or {})
        self.links = dict(links or {})
        self.commands = dict(commands or {})
        self.ran = []

    def system(self):
        return self._system

    def machine(self):
        return self._machine

    def run(self, argv, timeout=15):
        self.ran.append(list(argv))
        for prefix, result in self.commands.items():
            if tuple(argv[: len(prefix)]) == prefix:
                return result
        return 127, "", "command not found: " + argv[0]

    def read(self, path):
        return self.files.get(path)

    def listdir(self, path):
        path = path.rstrip("/") + "/"
        names = {p[len(path):].split("/")[0] for p in list(self.files) + list(self.links) if p.startswith(path)}
        names |= {d[len(path):].split("/")[0] for d in getattr(self, "dirs", []) if d.startswith(path)}
        return sorted(names) if names else None

    def readlink(self, path):
        return self.links.get(path)


def linux_host(peer_attached=True, carrier="1", module=True, ping=None, aer=None, route=None, ifname="thunderbolt0", with_interface=True):
    tb = "/sys/bus/thunderbolt/devices"
    files = {tb + "/domain0/security": "none", tb + "/0-0/device_name": "host", "/proc/modules": "thunderbolt_net 1 0 - Live 0x0\n" if module else "x 1 0 - Live 0x0\n",
             "/sys/class/net/%s/carrier" % ifname: carrier, "/sys/class/net/%s/operstate" % ifname: "up" if carrier == "1" else "down",
             "/sys/bus/pci/devices/0000:0a:00.0/class": "0x0c0340"}
    if peer_attached:
        files[tb + "/0-1/device_name"] = "Mac16,10"
        files[tb + "/0-1/authorized"] = "1"
    for kind, n in (aer or {"correctable": 0, "nonfatal": 0, "fatal": 0}).items():
        files["/sys/bus/pci/devices/0000:0a:00.0/aer_dev_" + kind] = "RxErr 0\nTOTAL_ERR_X %d\n" % n
    links = {"/sys/class/net/%s/device/driver" % ifname: "../../../bus/thunderbolt/drivers/thunderbolt-net"} if with_interface else {}
    cmds = {
        ("ip", "-j", "addr"): (0, fx("linux_ip_addr.json").replace("thunderbolt0", ifname), ""),
        ("ip", "-j", "route"): (0, (route if route is not None else fx("linux_route_get.json")).replace("thunderbolt0", ifname), ""),
        ("ping",): ping or (0, fx("linux_ping.txt"), ""),
        ("dmesg",): (1, "", "dmesg: read kernel buffer failed: Operation not permitted"),
    }
    return FakeHost("Linux", files, links, cmds)


def mac_host(profile=None, ifconfig=None, route=None, ping=None, bridge="bridge0"):
    cmds = {
        ("networksetup", "-listallhardwareports"): (0, fx("mac_hardware_ports.txt").replace("bridge0", bridge), ""),
        ("system_profiler",): (0, profile if profile is not None else fx("mac_system_profiler.txt"), ""),
        ("ifconfig",): (0, (ifconfig if ifconfig is not None else fx("mac_ifconfig_bridge0.txt")).replace("bridge0", bridge), ""),
        ("route",): (0, (route if route is not None else fx("mac_route_get.txt")).replace("bridge0", bridge), ""),
        ("ping",): ping or (0, fx("mac_ping.txt"), ""),
        (doc.MAC_FIREWALL,): (0, fx("mac_firewall.txt"), ""),
    }
    return FakeHost("Darwin", commands=cmds, machine="arm64")


def status_of(report, check):
    return [r["status"] for r in report.results if r["check"] == check]


class Parsers(unittest.TestCase):
    def test_hardware_ports(self):
        ports = doc.parse_hardware_ports(fx("mac_hardware_ports.txt"))
        names = {p["port"]: p["device"] for p in ports}
        self.assertEqual(names["Thunderbolt Bridge"], "bridge0")
        self.assertEqual(names["Thunderbolt 4"], "en4")

    def test_ifconfig_bridge(self):
        info = doc.parse_ifconfig(fx("mac_ifconfig_bridge0.txt"))
        self.assertEqual((info["mtu"], info["ipv4"], info["status"]), (9000, ["192.168.3.1/24"], "active"))
        self.assertTrue(info["link_local"])
        self.assertEqual(info["members"], ["en2", "en3", "en4"])

    def test_route_and_profile(self):
        self.assertEqual(doc.parse_route_interface(fx("mac_route_get.txt")), "bridge0")
        prof = doc.parse_thunderbolt_profile(fx("mac_system_profiler.txt"))
        self.assertEqual(prof["ports_connected"], 1)


class Linux(unittest.TestCase):
    def test_healthy_link_with_default_mtu_note(self):
        rep, _ = doc.diagnose(linux_host(), peer="192.168.3.1", tcp_port=None)
        self.assertFalse(rep.failed(), rep.render())
        self.assertEqual(status_of(rep, "route"), [doc.PASS])
        self.assertEqual(status_of(rep, "ping"), [doc.PASS])
        self.assertEqual(status_of(rep, "kernel-log"), [doc.SKIP])
        self.assertEqual(status_of(rep, "tcp"), [doc.SKIP])

    def test_no_peer_attached(self):
        rep, _ = doc.diagnose(linux_host(peer_attached=False), peer="192.168.3.1")
        self.assertEqual(status_of(rep, "peer-device"), [doc.FAIL])

    def test_no_carrier(self):
        rep, _ = doc.diagnose(linux_host(carrier="0"), peer="192.168.3.1")
        self.assertIn(doc.FAIL, status_of(rep, "interface"))

    def test_module_missing(self):
        host = linux_host(module=False)
        host.commands[("modinfo",)] = (1, "", "not found")
        rep, _ = doc.diagnose(host)
        self.assertEqual(status_of(rep, "driver"), [doc.FAIL])
        self.assertFalse(any(a[:2] == ["modprobe"] or a[0] == "sudo" for a in host.ran))

    def test_wrong_subnet_and_route_capture(self):
        rep, _ = doc.diagnose(linux_host(), peer="10.9.9.9", interface="thunderbolt0")
        self.assertIn(doc.FAIL, status_of(rep, "subnet"))
        other = json.dumps([{"dst": "192.168.3.1", "dev": "zt0", "prefsrc": "192.168.3.9", "flags": []}])
        rep, _ = doc.diagnose(linux_host(route=other), peer="192.168.3.1")
        self.assertIn(doc.FAIL, status_of(rep, "route"))

    def test_ping_loss_and_failure(self):
        out = "3 packets transmitted, 1 received, 66% packet loss"
        rep, _ = doc.diagnose(linux_host(ping=(1, out, "")), peer="192.168.3.1")
        self.assertEqual(status_of(rep, "ping"), [doc.WARN])
        rep, _ = doc.diagnose(linux_host(ping=(1, "3 packets transmitted, 0 received", "")), peer="192.168.3.1")
        self.assertEqual(status_of(rep, "ping"), [doc.FAIL])

    def test_aer(self):
        rep, _ = doc.diagnose(linux_host(aer={"correctable": 7, "nonfatal": 0, "fatal": 0}))
        self.assertEqual(status_of(rep, "pcie-errors"), [doc.WARN])
        rep, _ = doc.diagnose(linux_host(aer={"correctable": 0, "nonfatal": 1, "fatal": 0}))
        self.assertEqual(status_of(rep, "pcie-errors"), [doc.FAIL])

    def test_only_read_only_commands(self):
        host = linux_host()
        doc.diagnose(host, peer="192.168.3.1")
        for argv in host.ran:
            self.assertIn(argv[0], {"ip", "ping", "dmesg", "modinfo"})
            if argv[0] == "ip":
                self.assertIn(argv[2], {"addr", "route"})
                self.assertTrue(argv[3] in {"show", "get"})


class MacOS(unittest.TestCase):
    def test_healthy(self):
        rep, _ = doc.diagnose(mac_host(), peer="192.168.3.2")
        self.assertFalse(rep.failed(), rep.render())
        self.assertEqual(status_of(rep, "route"), [doc.PASS])
        self.assertEqual(status_of(rep, "bridge-members"), [doc.INFO])

    def test_inactive_bridge(self):
        text = fx("mac_ifconfig_bridge0.txt").replace("status: active", "status: inactive")
        rep, _ = doc.diagnose(mac_host(ifconfig=text), peer="192.168.3.2")
        self.assertIn(doc.FAIL, status_of(rep, "interface"))

    def test_no_device_connected(self):
        text = fx("mac_system_profiler.txt").replace("Status: Device connected", "Status: No device connected")
        rep, _ = doc.diagnose(mac_host(profile=text))
        self.assertEqual(status_of(rep, "peer-device"), [doc.FAIL])

    def test_no_bridge_service(self):
        host = mac_host()
        ports = "\n".join(b for b in fx("mac_hardware_ports.txt").split("\n\n") if "Bridge" not in b)
        host.commands[("networksetup", "-listallhardwareports")] = (0, ports, "")
        rep, _ = doc.diagnose(host)
        self.assertIn(doc.FAIL, status_of(rep, "interface"))

    def test_route_through_other_interface(self):
        text = fx("mac_route_get.txt").replace("bridge0", "utun4")
        rep, _ = doc.diagnose(mac_host(route=text), peer="192.168.3.2")
        self.assertIn(doc.FAIL, status_of(rep, "route"))


def ok_tcp(report, peer, port):
    report.add(doc.PASS, "tcp", "stub")


def bad_tcp(report, peer, port):
    report.add(doc.FAIL, "tcp", "stub")


class Readiness(unittest.TestCase):
    def test_ready_only_when_route_ping_and_tcp_all_ran(self):
        rep, facts = doc.diagnose(linux_host(), peer="192.168.3.1", tcp_port=5, tcp=ok_tcp)
        self.assertEqual(rep.readiness(), "ready")
        self.assertEqual(facts["route_interface"], "thunderbolt0")
        self.assertIn("verdict        : READY", rep.render())

    def test_success_without_tcp_or_peer_is_incomplete_not_ready(self):
        rep, _ = doc.diagnose(linux_host(), peer="192.168.3.1")
        self.assertEqual((rep.failed(), rep.readiness()), (False, "incomplete"))
        self.assertIn("tcp", " ".join(rep.not_performed()))
        rep, _ = doc.diagnose(linux_host())
        self.assertEqual(rep.readiness(), "incomplete")
        self.assertIn("INCOMPLETE", rep.render())

    def test_exit_codes(self):
        self.assertEqual(doc.main(["--peer", "192.168.3.1"], host=linux_host()), 3)
        self.assertEqual(doc.main(["--peer", "192.168.3.1"], host=linux_host(peer_attached=False)), 1)

    def test_partial_connectivity_ping_ok_tcp_failed(self):
        rep, facts = doc.diagnose(linux_host(), peer="192.168.3.1", tcp_port=5, tcp=bad_tcp)
        self.assertEqual((rep.readiness(), status_of(rep, "ping"), status_of(rep, "tcp")), ("not ready", [doc.PASS], [doc.FAIL]))

    def test_json_summary_lists_what_was_not_performed(self):
        rep, facts = doc.diagnose(linux_host(), peer="192.168.3.1")
        s = doc.json_report(rep, facts)["summary"]
        self.assertEqual((s["verdict"], s["link_interface"], s["peer"], s["route_interface"], s["ping"], s["tcp"]), ("incomplete", "thunderbolt0", "192.168.3.1", "thunderbolt0", "ok", "not run"))
        self.assertTrue(any(x.startswith("tcp") for x in s["not_performed"]))
        self.assertNotIn("skynet", json.dumps(doc.json_report(rep, facts)))


class Names(unittest.TestCase):
    def test_renamed_linux_interfaces(self):
        for name in ("enx02aabbccddee", "tb-direct", "thunderbolt1"):
            rep, facts = doc.diagnose(linux_host(ifname=name), peer="192.168.3.1")
            self.assertFalse(rep.failed(), (name, rep.render()))
            self.assertEqual(facts["interface"]["name"], name)

    def test_other_macos_bridge_identifiers(self):
        for name in ("bridge1", "bridge102"):
            rep, facts = doc.diagnose(mac_host(bridge=name), peer="192.168.3.2")
            self.assertFalse(rep.failed(), (name, rep.render()))
            self.assertEqual((facts["interface"]["name"], facts["route_interface"]), (name, name))

    def test_interface_override_on_linux(self):
        rep, facts = doc.diagnose(linux_host(ifname="tb-direct"), peer="192.168.3.1", interface="tb-direct")
        self.assertEqual(facts["interface"]["name"], "tb-direct")

    def test_multiple_linux_interfaces_warn(self):
        host = linux_host()
        host.links["/sys/class/net/thunderbolt1/device/driver"] = "../../../bus/thunderbolt/drivers/thunderbolt-net"
        rep, _ = doc.diagnose(host, peer="192.168.3.1")
        self.assertTrue(any("several Thunderbolt interfaces" in r["summary"] for r in rep.results))


class DegradedEnvironments(unittest.TestCase):
    def test_missing_ip_utility_skips_instead_of_guessing(self):
        host = linux_host()
        host.commands.pop(("ip", "-j", "addr"))
        rep, _ = doc.diagnose(host, peer="192.168.3.1")
        self.assertEqual(status_of(rep, "interface"), [doc.SKIP])
        self.assertNotEqual(rep.readiness(), "ready")

    def test_malformed_json_is_reported_not_crashed_on(self):
        host = linux_host()
        host.commands[("ip", "-j", "addr")] = (0, "{not json", "")
        rep, _ = doc.diagnose(host, peer="192.168.3.1")
        self.assertEqual(status_of(rep, "interface"), [doc.SKIP])
        self.assertIn("parsed", rep.results[[r["check"] for r in rep.results].index("interface")]["summary"])
        host = linux_host(route="garbage")
        rep, _ = doc.diagnose(host, peer="192.168.3.1")
        self.assertEqual(status_of(rep, "route"), [doc.SKIP])

    def test_route_permission_error_is_skipped(self):
        host = linux_host()
        host.commands[("ip", "-j", "route")] = (127, "", "command not found: ip")
        rep, _ = doc.diagnose(host, peer="192.168.3.1")
        self.assertEqual(status_of(rep, "route"), [doc.SKIP])

    def test_missing_ping_is_skipped(self):
        host = linux_host()
        host.commands[("ping",)] = (127, "", "command not found: ping")
        rep, facts = doc.diagnose(host, peer="192.168.3.1")
        self.assertEqual((status_of(rep, "ping"), facts["ping"]), ([doc.SKIP], "not run"))

    def test_macos_missing_networksetup_and_ifconfig(self):
        host = mac_host()
        host.commands.pop(("networksetup", "-listallhardwareports"))
        rep, _ = doc.diagnose(host)
        self.assertIn(doc.SKIP, status_of(rep, "services"))
        self.assertNotEqual(rep.readiness(), "ready")
        host = mac_host()
        host.commands[("ifconfig",)] = (127, "", "command not found: ifconfig")
        rep, _ = doc.diagnose(host, peer="192.168.3.2")
        self.assertEqual(status_of(rep, "interface"), [doc.SKIP])

    def test_macos_garbled_ifconfig_does_not_crash(self):
        rep, _ = doc.diagnose(mac_host(ifconfig="???"), peer="192.168.3.2")
        self.assertIn(doc.FAIL, status_of(rep, "address"))

    def test_untested_architecture_warns(self):
        host = linux_host()
        host._machine = "aarch64"
        rep, _ = doc.diagnose(host, peer="192.168.3.1")
        self.assertIn(doc.WARN, status_of(rep, "platform"))

    def test_peer_attached_without_interface_is_named_precisely(self):
        rep, _ = doc.diagnose(linux_host(with_interface=False), peer="192.168.3.1")
        text = " ".join(r["summary"] for r in rep.results if r["check"] == "interface")
        self.assertIn("peer is attached but no network interface", text)


class Platforms(unittest.TestCase):
    def test_unsupported_system(self):
        rep, _ = doc.diagnose(FakeHost("Windows"))
        self.assertTrue(rep.failed())

    def test_linux_without_thunderbolt_support(self):
        rep, _ = doc.diagnose(FakeHost("Linux"))
        self.assertEqual(status_of(rep, "controller"), [doc.FAIL])

    def test_usage_errors(self):
        for argv in (["--peer", "not-an-ip"], ["--listen", "192.168.77.1"]):
            with self.assertRaises(SystemExit) as cm:
                doc.parse_args(argv)
            self.assertEqual(cm.exception.code, 2)


class Output(unittest.TestCase):
    def test_json_is_sanitized(self):
        rep = doc.Report()
        rep.add(doc.INFO, "x", "mac 02:11:22:33:44:55 addr fe80::aaaa:bbbb:cccc:dddd%bridge0")
        text = json.dumps(doc.json_report(rep, {}))
        self.assertNotIn("02:11:22", text)
        self.assertNotIn("aaaa", text)
        self.assertEqual(json.loads(text)["schema"], doc.SCHEMA)

    def test_main_writes_json_and_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "r.json")
            rc = doc.main(["--peer", "192.168.3.1", "--json", path], host=linux_host(peer_attached=False))
            self.assertEqual(rc, 1)
            self.assertEqual(json.load(open(path))["schema"], doc.SCHEMA)


class Tcp(unittest.TestCase):
    def test_echo_roundtrip_on_loopback(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        def serve():
            c, _ = srv.accept()
            with c:
                c.sendall(c.recv(100))

        t = threading.Thread(target=serve)
        t.start()
        ok, _ = doc.tcp_check("127.0.0.1", port)
        t.join()
        srv.close()
        self.assertTrue(ok)

    def test_refused(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        ok, detail = doc.tcp_check("127.0.0.1", port, timeout=1)
        self.assertFalse(ok)
        self.assertTrue(detail)


if __name__ == "__main__":
    unittest.main()
