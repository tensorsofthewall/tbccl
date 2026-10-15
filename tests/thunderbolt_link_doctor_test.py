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


def linux_host(peer_attached=True, carrier="1", module=True, ping=None, aer=None, route=None):
    tb = "/sys/bus/thunderbolt/devices"
    files = {tb + "/domain0/security": "none", tb + "/0-0/device_name": "host", "/proc/modules": "thunderbolt_net 1 0 - Live 0x0\n" if module else "x 1 0 - Live 0x0\n",
             "/sys/class/net/thunderbolt0/carrier": carrier, "/sys/class/net/thunderbolt0/operstate": "up" if carrier == "1" else "down",
             "/sys/bus/pci/devices/0000:0a:00.0/class": "0x0c0340"}
    if peer_attached:
        files[tb + "/0-1/device_name"] = "Mac16,10"
        files[tb + "/0-1/authorized"] = "1"
    for kind, n in (aer or {"correctable": 0, "nonfatal": 0, "fatal": 0}).items():
        files["/sys/bus/pci/devices/0000:0a:00.0/aer_dev_" + kind] = "RxErr 0\nTOTAL_ERR_X %d\n" % n
    links = {"/sys/class/net/thunderbolt0/device/driver": "../../../bus/thunderbolt/drivers/thunderbolt-net"}
    cmds = {
        ("ip", "-j", "addr"): (0, fx("linux_ip_addr.json"), ""),
        ("ip", "-j", "route"): (0, route if route is not None else fx("linux_route_get.json"), ""),
        ("ping",): ping or (0, fx("linux_ping.txt"), ""),
        ("dmesg",): (1, "", "dmesg: read kernel buffer failed: Operation not permitted"),
    }
    return FakeHost("Linux", files, links, cmds)


def mac_host(profile=None, ifconfig=None, route=None, ping=None):
    cmds = {
        ("networksetup", "-listallhardwareports"): (0, fx("mac_hardware_ports.txt"), ""),
        ("system_profiler",): (0, profile if profile is not None else fx("mac_system_profiler.txt"), ""),
        ("ifconfig",): (0, ifconfig if ifconfig is not None else fx("mac_ifconfig_bridge0.txt"), ""),
        ("route",): (0, route if route is not None else fx("mac_route_get.txt"), ""),
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
