#!/usr/bin/env python3
"""Tests for capture_tb4_runtime_state.py and compare_tb4_runtime_state.py.

Phase 30 Part C/D/W. Covers the pure parsing helpers (against fixture
sysfs/procfs trees, never the real machine) and the snapshot-diff
classifier's changed/unchanged/unavailable categorization.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import capture_tb4_runtime_state as rt  # noqa: E402
import compare_tb4_runtime_state as cmp_mod  # noqa: E402


class IrqTableTests(unittest.TestCase):
    def test_parses_named_irqs_only(self):
        text = (
            "           CPU0       CPU1\n"
            " 177:          0          5  IR-PCI-MSIX-0000:0a:00.0    2-edge      thunderbolt\n"
            " 178:          0          9  IR-PCI-MSIX-0000:0a:00.0    3-edge      thunderbolt\n"
            "   9:          1          0  IO-APIC    9-fasteoi   acpi\n"
        )
        result = rt.irq_table(text, [177, 178])
        self.assertEqual(set(result), {"177", "178"})
        self.assertEqual(result["177"]["per_cpu_counts"], [0, 5])
        self.assertIn("thunderbolt", result["177"]["descriptor"])

    def test_missing_irq_absent_from_result(self):
        result = rt.irq_table(" 177:  0  0  thunderbolt\n", [177, 999])
        self.assertEqual(set(result), {"177"})


class IrqAffinityTests(unittest.TestCase):
    def test_reads_smp_and_effective_affinity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "proc/irq/177"
            base.mkdir(parents=True)
            (base / "smp_affinity_list").write_text("0-15\n")
            (base / "effective_affinity_list").write_text("13\n")
            with patch.object(rt, "Path", side_effect=lambda p: Path(tmp) / str(p).lstrip("/")
                               if str(p).startswith("/proc/irq") else Path(p)):
                result = rt.irq_affinity([177])
            self.assertEqual(result["177"]["smp_affinity_list"], "0-15")
            self.assertEqual(result["177"]["effective_affinity_list"], "13")


class CpuStateTests(unittest.TestCase):
    def test_reads_governor_freq_epp_and_cpuidle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpufreq = root / "cpu13/cpufreq"
            cpufreq.mkdir(parents=True)
            (cpufreq / "scaling_governor").write_text("powersave")
            (cpufreq / "scaling_cur_freq").write_text("2900000")
            (cpufreq / "scaling_min_freq").write_text("800000")
            (cpufreq / "scaling_max_freq").write_text("3400000")
            (cpufreq / "scaling_driver").write_text("intel_pstate")
            (cpufreq / "energy_performance_preference").write_text("balance_performance")
            idle_state0 = root / "cpu13/cpuidle/state0"
            idle_state0.mkdir(parents=True)
            (idle_state0 / "name").write_text("POLL")
            (idle_state0 / "usage").write_text("100")
            (idle_state0 / "time").write_text("5000")
            (root / "cpuidle").mkdir(parents=True, exist_ok=True)
            (root / "cpuidle/current_driver").write_text("intel_idle")
            with patch.object(rt, "CPU_SYSFS", root):
                result = rt.cpu_state(13)
            self.assertEqual(result["scaling_governor"], "powersave")
            self.assertEqual(result["scaling_cur_freq"], "2900000")
            self.assertEqual(result["energy_performance_preference"], "balance_performance")
            self.assertEqual(result["cpuidle_driver"], "intel_idle")
            self.assertEqual(result["cpuidle_states"]["state0"]["name"], "POLL")
            self.assertEqual(result["cpuidle_states"]["state0"]["usage"], "100")

    def test_missing_fields_are_none_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rt, "CPU_SYSFS", Path(tmp)):
                result = rt.cpu_state(99)
            self.assertIsNone(result["scaling_governor"])
            self.assertEqual(result["cpuidle_states"], {})


class ModuleParametersTests(unittest.TestCase):
    def test_reads_all_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp) / "sys/module/thunderbolt_net/parameters"
            params.mkdir(parents=True)
            (params / "e2e").write_text("Y")
            with patch.object(rt, "Path", side_effect=lambda p: Path(tmp) / str(p).lstrip("/")
                               if str(p).startswith("/sys/module") else Path(p)):
                result = rt.module_parameters("thunderbolt_net")
            self.assertEqual(result, {"e2e": "Y"})

    def test_missing_module_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rt, "Path", side_effect=lambda p: Path(tmp) / str(p).lstrip("/")
                               if str(p).startswith("/sys/module") else Path(p)):
                result = rt.module_parameters("nonexistent_module")
            self.assertIsNone(result)


class DiffSnapshotsTests(unittest.TestCase):
    def base(self):
        return {
            "boot_id": "boot-a",
            "irq_effective_cpu": {"177": 13, "178": 14},
            "cpu_state": {"13": {"scaling_governor": "powersave", "scaling_cur_freq": "2900000"}},
            "module_parameters": {"thunderbolt_net": {"e2e": "Y"}},
            "pci_link_state": {"0000:0a:00.0": {"current_link_width": "4"}},
            "ethtool_offloads": "generic-receive-offload: on",
        }

    def test_equal_snapshots_all_unchanged(self):
        before = self.base()
        after = dict(before)
        result = cmp_mod.diff_snapshots(before, after)
        self.assertEqual(result["changed"], {})
        self.assertGreater(result["unchanged_count"], 0)

    def test_irq_cpu_changed(self):
        before = self.base()
        after = self.base()
        after["irq_effective_cpu"]["178"] = 5
        result = cmp_mod.diff_snapshots(before, after)
        self.assertIn("irq_effective_cpu.178", result["changed"])
        self.assertEqual(result["changed"]["irq_effective_cpu.178"], {"before": 14, "after": 5})

    def test_governor_same_but_frequency_changed(self):
        before = self.base()
        after = self.base()
        after["cpu_state"]["13"]["scaling_cur_freq"] = "800000"
        result = cmp_mod.diff_snapshots(before, after)
        self.assertNotIn("cpu_state.13.scaling_governor", result["changed"])
        self.assertIn("cpu_state.13.scaling_cur_freq", result["changed"])

    def test_module_param_changed(self):
        before = self.base()
        after = self.base()
        after["module_parameters"]["thunderbolt_net"]["e2e"] = "N"
        result = cmp_mod.diff_snapshots(before, after)
        self.assertIn("module_parameters.thunderbolt_net.e2e", result["changed"])

    def test_link_width_changed(self):
        before = self.base()
        after = self.base()
        after["pci_link_state"]["0000:0a:00.0"]["current_link_width"] = "2"
        result = cmp_mod.diff_snapshots(before, after)
        self.assertIn("pci_link_state.0000:0a:00.0.current_link_width", result["changed"])

    def test_offload_changed(self):
        before = self.base()
        after = self.base()
        after["ethtool_offloads"] = "generic-receive-offload: off"
        result = cmp_mod.diff_snapshots(before, after, ignore_prefixes=())
        self.assertIn("ethtool_offloads", result["changed"])

    def test_field_missing_from_one_side_is_unavailable_not_changed(self):
        before = self.base()
        after = self.base()
        del after["module_parameters"]["thunderbolt_net"]["e2e"]
        result = cmp_mod.diff_snapshots(before, after)
        self.assertNotIn("module_parameters.thunderbolt_net.e2e", result["changed"])
        self.assertIn("module_parameters.thunderbolt_net.e2e", result["unavailable"])

    def test_none_value_is_unavailable_not_changed(self):
        before = self.base()
        after = self.base()
        after["cpu_state"]["13"]["scaling_governor"] = None
        result = cmp_mod.diff_snapshots(before, after)
        self.assertIn("cpu_state.13.scaling_governor", result["unavailable"])
        self.assertNotIn("cpu_state.13.scaling_governor", result["changed"])

    def test_boot_id_changed(self):
        before = self.base()
        after = self.base()
        after["boot_id"] = "boot-b"
        result = cmp_mod.diff_snapshots(before, after)
        self.assertEqual(result["changed"]["boot_id"], {"before": "boot-a", "after": "boot-b"})

    def test_aer_reset_after_reboot_shows_as_changed(self):
        before = {"pci_devices": [{"bdf": "0000:08:00.0", "aer": {"TOTAL_ERR_COR": 5}}]}
        after = {"pci_devices": [{"bdf": "0000:08:00.0", "aer": {"TOTAL_ERR_COR": 0}}]}
        result = cmp_mod.diff_snapshots(before, after)
        self.assertEqual(result["changed"]["pci_devices.0.aer.TOTAL_ERR_COR"],
                          {"before": 5, "after": 0})
        self.assertNotIn("pci_devices.0.bdf", result["changed"])

    def test_default_ignore_prefixes_exclude_timestamp(self):
        before = {"timestamp_utc": "2026-01-01T00:00:00Z", "boot_id": "x"}
        after = {"timestamp_utc": "2026-01-02T00:00:00Z", "boot_id": "x"}
        result = cmp_mod.diff_snapshots(before, after, ignore_prefixes=cmp_mod.DEFAULT_IGNORE_PREFIXES)
        self.assertNotIn("timestamp_utc", result["changed"])
        self.assertNotIn("timestamp_utc", result["unavailable"])


class CliSmokeTests(unittest.TestCase):
    def test_compare_cli_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            out_path = Path(tmp) / "diff.json"
            before_path.write_text(json.dumps({"boot_id": "a"}))
            after_path.write_text(json.dumps({"boot_id": "b"}))
            with patch("sys.argv", ["compare_tb4_runtime_state.py", str(before_path),
                                     str(after_path), "--output", str(out_path)]):
                cmp_mod.main()
            result = json.loads(out_path.read_text())
            self.assertEqual(result["changed"]["boot_id"], {"before": "a", "after": "b"})


if __name__ == "__main__":
    unittest.main()
