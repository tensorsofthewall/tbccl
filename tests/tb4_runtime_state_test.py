#!/usr/bin/env python3
"""Tests for capture_tb4_runtime_state.py and compare_tb4_runtime_state.py.

The sustained-session reproduction work. Covers the pure parsing helpers (against fixture
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


class GpuStateTests(unittest.TestCase):
    """GPU/workload capture must degrade gracefully
    (None, not an exception) when nvidia-smi is absent, and must not treat
    an unsupported field or an empty compute-process list as a failure."""

    def test_nvidia_smi_absent_returns_none(self):
        with patch.object(rt.shutil, "which", return_value=None):
            self.assertIsNone(rt.gpu_state())
            self.assertIsNone(rt.active_compute_processes())
            self.assertIsNone(rt.gpu_graphics_processes())

    def test_gpu_state_parses_full_row(self):
        csv_line = ("2026/09/30 15:56:10.462, NVIDIA GeForce RTX 3070 Ti Laptop GPU, P8, "
                     "5 %, 3 %, 680 MiB, 8192 MiB, 18.17 W, 61, 1, 8")
        fake = {"command": [], "returncode": 0, "stdout": csv_line, "stderr": ""}
        with patch.object(rt, "shutil") as fake_shutil, patch.object(rt, "command", return_value=fake):
            fake_shutil.which.return_value = "/usr/bin/nvidia-smi"
            result = rt.gpu_state()
        self.assertTrue(result["available"])
        self.assertEqual(result["fields"]["pstate"], "P8")
        self.assertEqual(result["fields"]["pcie.link.width.current"], "8")

    def test_gpu_state_unsupported_field_does_not_crash(self):
        # nvidia-smi prints "[Not Supported]" for a field this driver/GPU
        # doesn't expose -- still 11 comma-separated values, just parses as
        # a literal string rather than a clean number.
        csv_line = "2026/09/30 00:00:00.000, GPU, P0, 0 %, 0 %, 0 MiB, 0 MiB, [Not Supported], 0, 0, 0"
        fake = {"command": [], "returncode": 0, "stdout": csv_line, "stderr": ""}
        with patch.object(rt, "shutil") as fake_shutil, patch.object(rt, "command", return_value=fake):
            fake_shutil.which.return_value = "/usr/bin/nvidia-smi"
            result = rt.gpu_state()
        self.assertTrue(result["available"])
        self.assertEqual(result["fields"]["power.draw"], "[Not Supported]")

    def test_active_compute_processes_empty_is_not_an_error(self):
        fake = {"command": [], "returncode": 0, "stdout": "", "stderr": ""}
        with patch.object(rt, "shutil") as fake_shutil, patch.object(rt, "command", return_value=fake):
            fake_shutil.which.return_value = "/usr/bin/nvidia-smi"
            result = rt.active_compute_processes()
        self.assertTrue(result["available"])
        self.assertEqual(result["processes"], [])

    def test_active_compute_processes_lists_multiple(self):
        stdout = "1234, python3, 4096 MiB\n5678, python3, 2048 MiB\n"
        fake = {"command": [], "returncode": 0, "stdout": stdout, "stderr": ""}
        with patch.object(rt, "shutil") as fake_shutil, patch.object(rt, "command", return_value=fake):
            fake_shutil.which.return_value = "/usr/bin/nvidia-smi"
            result = rt.active_compute_processes()
        self.assertEqual(len(result["processes"]), 2)
        self.assertEqual(result["processes"][0]["pid"], "1234")

    def test_gpu_graphics_processes_multiple_unrelated_clients(self):
        full_text = (
            "Processes:\n"
            "|    0   N/A  N/A            1887      G   /usr/lib/Xorg           487MiB |\n"
            "|    0   N/A  N/A            6343      G   /usr/lib/firefox/firefox 10MiB |\n"
        )
        fake = {"command": [], "returncode": 0, "stdout": full_text, "stderr": ""}
        with patch.object(rt, "shutil") as fake_shutil, patch.object(rt, "command", return_value=fake):
            fake_shutil.which.return_value = "/usr/bin/nvidia-smi"
            result = rt.gpu_graphics_processes()
        self.assertEqual(result["returncode"], 0)
        self.assertIn("Xorg", result["full_text"])
        self.assertIn("firefox", result["full_text"])


class DiscoverIrqsTests(unittest.TestCase):
    def test_finds_nvidia_and_nvme_irqs_without_assuming_numbers(self):
        text = (
            " 211:      0  446841  IR-PCI-MSI-0000:01:00.0    0-edge      nvidia\n"
            " 150:  16166       0  IR-PCI-MSIX-0000:04:00.0    1-edge      nvme0q1\n"
            "   9:      1       0  IO-APIC    9-fasteoi   acpi\n"
        )
        with patch.object(rt, "read", return_value=text):
            nvidia = rt.discover_irqs(rt.re.compile(r"\bnvidia\b", rt.re.I))
            nvme = rt.discover_irqs(rt.re.compile(r"\bnvme\d+q\d+\b", rt.re.I))
        self.assertEqual(set(nvidia), {"211"})
        self.assertEqual(nvidia["211"]["total_count"], 446841)
        self.assertEqual(set(nvme), {"150"})
        self.assertNotIn("9", nvidia)
        self.assertNotIn("9", nvme)


class DiskstatsMeminfoPsiTests(unittest.TestCase):
    def test_diskstats_filters_to_nvme_only(self):
        text = (
            " 259       0 nvme0n1 166668 0 16212321 0 50688 0 6743674 0 0 0 0 0 0 0 0\n"
            "   8       0 sda 100 0 200 0 50 0 300 0 0 0 0 0 0 0 0\n"
        )
        with patch.object(rt, "read", return_value=text):
            result = rt.diskstats_snapshot()
        self.assertEqual(set(result), {"nvme0n1"})
        self.assertEqual(result["nvme0n1"]["reads_completed"], 166668)

    def test_meminfo_selected_fields_only(self):
        text = "MemTotal:       32000000 kB\nMemAvailable:   23982104 kB\nCached:          7745388 kB\n"
        with patch.object(rt, "read", return_value=text):
            result = rt.meminfo_snapshot()
        self.assertEqual(result, {"MemAvailable": "23982104 kB", "Cached": "7745388 kB"})
        self.assertNotIn("MemTotal", result)

    def test_psi_reads_all_three_pressure_files(self):
        with patch.object(rt, "read", side_effect=lambda p: f"fixture:{p}"):
            result = rt.psi_snapshot()
        self.assertEqual(set(result), {"cpu", "io", "memory"})
        self.assertIn("/proc/pressure/cpu", result["cpu"])

    def test_psi_missing_file_is_none_not_error(self):
        with patch.object(rt, "read", return_value=None):
            result = rt.psi_snapshot()
        self.assertIsNone(result["cpu"])


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
