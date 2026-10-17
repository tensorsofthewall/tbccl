"""AER observability is reported separately from health, and never as 'no errors' when the kernel cannot report them."""
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite import health

CMDLINE_NOAER = "BOOT_IMAGE=/boot/vmlinuz-linux root=UUID=0000 rw quiet pcie_aspm=off pci=noaer"
CMDLINE_PLAIN = "BOOT_IMAGE=/boot/vmlinuz-linux root=UUID=0000 rw quiet pcie_aspm=off"
AER_DEVICE = {"bdf": "0000:00:1d.4", "aer": {"aer_dev_correctable": "RxErr 0\nBadTLP 0", "aer_dev_nonfatal": "Undefined 0"}}


class Detection(unittest.TestCase):
    def test_pci_noaer_means_unavailable_even_when_counters_are_readable_and_zero(self):
        snapshot = {"system": "Linux", "pci_devices": [AER_DEVICE]}
        result = health.aer_observability(cmdline=CMDLINE_NOAER, snapshot=snapshot)
        self.assertEqual(result["state"], "unavailable")
        self.assertTrue(result["pci_noaer"])
        self.assertTrue(result["counters_readable"])
        self.assertIn("AER unavailable; no detected link-state or interface-health regression under the available checks", result["statement"])

    def test_noaer_among_other_pci_options(self):
        for line in ("pci=realloc,noaer", "pci=noaer,realloc=off", "quiet pci=noaer"):
            self.assertEqual(health.aer_observability(cmdline=line)["state"], "unavailable", line)

    def test_similar_parameters_are_not_noaer(self):
        for line in ("pci=pcie_bus_perf", "pcie_aspm=off", "pci=noaerx", "noaer"):
            self.assertNotEqual(health.aer_observability(cmdline=line, snapshot={"pci_devices": [AER_DEVICE]})["state"], "unavailable", line)

    def test_observable_needs_a_known_command_line_without_noaer_and_readable_counters(self):
        result = health.aer_observability(cmdline=CMDLINE_PLAIN, snapshot={"system": "Linux", "pci_devices": [AER_DEVICE]})
        self.assertEqual(result["state"], "observable")
        self.assertFalse(result["pci_noaer"])

    def test_missing_counters_make_it_unavailable(self):
        result = health.aer_observability(cmdline=CMDLINE_PLAIN, snapshot={"system": "Linux", "pci_devices": [{"aer": {}}, {"aer": None}]})
        self.assertEqual(result["state"], "unavailable")
        self.assertFalse(result["counters_readable"])

    def test_unknown_when_the_command_line_cannot_be_read(self):
        result = health.aer_observability(snapshot={"system": "Linux", "pci_devices": [AER_DEVICE]})
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["pci_noaer"])
        self.assertIn("must not be reported as 'no PCIe errors'", result["statement"])

    def test_command_line_is_taken_from_the_snapshot_kernel_log(self):
        events = [{"message": "something else"}, {"message": "Command line: " + CMDLINE_NOAER}]
        result = health.aer_observability(snapshot={"system": "Linux", "kernel_events": events})
        self.assertEqual((result["state"], result["cmdline_source"]), ("unavailable", "snapshot kernel log"))

    def test_other_operating_systems_expose_no_aer(self):
        result = health.aer_observability(cmdline="", snapshot={"system": "Darwin"})
        self.assertEqual(result["state"], "unavailable")
        self.assertIn("Darwin", result["detail"])

    def test_wording_never_claims_zero_errors(self):
        for result in (health.aer_observability(cmdline=CMDLINE_NOAER), health.aer_observability(), health.aer_observability(cmdline="", snapshot={"system": "Darwin"})):
            self.assertNotIn("zero AER", result["statement"])
            self.assertNotIn("no AER errors", result["statement"])
            self.assertNotIn("0 AER", result["statement"])

    def test_states_are_the_documented_three(self):
        self.assertEqual(health.AER_STATES, ("observable", "unavailable", "unknown"))

    def test_nothing_is_written_or_changed(self):
        # a pure function of its inputs: calling it twice gives the same answer and takes no action
        a = health.aer_observability(cmdline=CMDLINE_NOAER)
        b = health.aer_observability(cmdline=CMDLINE_NOAER)
        self.assertEqual(a, b)


class LocalCommandLine(unittest.TestCase):
    def test_reads_a_file_and_tolerates_a_missing_one(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cmdline"
            path.write_text(CMDLINE_NOAER + "\n")
            self.assertEqual(health.local_cmdline(path), CMDLINE_NOAER)
            self.assertIsNone(health.local_cmdline(Path(d) / "absent"))

    def test_tokens_expand_pci_options(self):
        self.assertEqual(health.kernel_cmdline_tokens("a pci=noaer,realloc b"), ["a", "pci=noaer", "pci=realloc", "b"])


class SignalIntegration(unittest.TestCase):
    class Host:
        id, interface, address, interfaces, expect, launcher = "h", "eth0", None, {}, {}, {}

    def wrapper(self, events=None, devices=None):
        snap = {"boot_id": "b", "kernel_log_available": True, "system": "Linux", "interface": {"carrier": "1", "counters": {"rx_errors": "0", "tx_errors": "0"}}}
        if events is not None:
            snap["kernel_events"] = events
        if devices is not None:
            snap["pci_devices"] = devices
        return {"interface": {"present": True, "operstate": "up", "mtu": 1500, "addresses": ["192.0.2.1/24"]}, "gpu": None, "snapshot": snap, "snapshot_error": None}

    def test_health_is_ok_while_observability_is_unavailable_and_both_are_reported(self):
        sig = health.evaluate(self.Host(), self.wrapper([{"message": "Command line: pci=noaer"}], [AER_DEVICE]))
        self.assertEqual(sig["link_state"]["status"], "ok")
        self.assertEqual(sig["aer_observability"]["status"], "unavailable")
        failures, warnings = health.preflight_problems(self.Host(), sig)
        self.assertEqual(failures, [])
        self.assertTrue(any("aer_observability" in w and "pci=noaer" in w for w in warnings))

    def test_observable_host_has_no_aer_warning(self):
        sig = health.evaluate(self.Host(), self.wrapper([{"message": "Command line: " + CMDLINE_PLAIN}], [AER_DEVICE]))
        self.assertEqual(sig["aer_observability"]["status"], "ok")
        self.assertEqual(health.preflight_problems(self.Host(), sig), ([], []))

    def test_a_real_shaped_snapshot(self):
        # the shape the snapshot script records: pci_devices[*].aer is a dict of counter files, the kernel log carries the command line
        events = [{"cursor": "s=x", "monotonic_us": "3854936", "message": "Command line: BOOT_IMAGE=/boot/vmlinuz-linux-lts root=UUID=x rw pcie_aspm=off pci=noaer", "source": "unattributed"}]
        device = {"bdf": "0000:08:00.0", "present": True, "aer": {"aer_dev_correctable": "RxErr 0\nBadTLP 0\nBadDLLP 0\nRollover 0\nTimeout 0\nNonFatalErr 0\nCorrIntErr 0\nHeaderOF 0\nTOTAL_ERR_COR 0"}}
        self.assertEqual(health.aer_observability(snapshot={"system": "Linux", "kernel_events": events, "pci_devices": [device]})["state"], "unavailable")


if __name__ == "__main__":
    unittest.main()
