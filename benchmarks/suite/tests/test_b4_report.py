"""B4 report tables: bimodal repetitions are split, native and framework arms are matched only through their interleaved tags, DDP rows keep per-rank phases."""
import unittest

from benchmarks.suite.reporting import b4


def rec(bench, op, size, median, tag, backends=("host", "host"), tier="t2-framework", extra=None, options=None):
    hosts = [{"host_id": "host-a", "os": "EndeavourOS (kernel 6)"}, {"host_id": "host-b", "os": "macOS 26"}]
    opts = {"tag": tag, **(options or {})}
    return {"_tier": tier, "status": {"state": "success"}, "verification": {"status": "passed"}, "benchmark": {"name": bench, "operation": op},
            "configuration": {"payload_bytes": size, "iterations": 30, "options": opts, "repetition_index": 0, "data_direction": {"mode": "directional", "source_host": "host-a", "destination_host": "host-b"}},
            "environment": {"hosts": hosts}, "artifacts": {"ranks": [{"host_id": "host-a", "memory_backend": backends[0]}, {"host_id": "host-b", "memory_backend": backends[1]}]},
            "measurements": {"timing": {"metric": "m", "statistics": {"median": median, "p95": median}, "primary_statistic": "median", "sample_count": 30, "samples": [median] * 30, "per_rank": None},
                             "throughput": [], "extra": {k: {"value": v, "unit": "u"} for k, v in (extra or {}).items()}}}


class Report(unittest.TestCase):
    def test_modes_are_split_only_when_repetitions_differ_by_more_than_the_ratio(self):
        self.assertIsNone(b4.modes([125.0, 127.0, 130.0]))
        m = b4.modes([127.0, 249.0, 250.0, 126.0, 248.0])
        self.assertEqual((m["low_count"], m["high_count"]), (2, 3))
        self.assertAlmostEqual(m["high_mode_us"], 249.0)
        self.assertIsNone(b4.modes([100.0]))

    def test_framework_rows_pair_native_and_torch_arms_by_block_tag_and_keep_every_repetition(self):
        recs = []
        for block, (a1, b1, b2, a2) in enumerate(((100, 140, 150, 104), (101, 141, 151, 99)), 1):
            for arm, n, v in (("A1", "tensor_transfer", a1), ("B1", "torch_p2p", b1), ("B2", "torch_p2p", b2), ("A2", "tensor_transfer", a2)):
                op = "end_to_end" if n == "tensor_transfer" else "p2p_ack"
                recs.append(rec(n, op, 4096, v, f"b{block}-cpu-cpu-s0-{arm}"))
        recs.append(rec("torch_p2p", "p2p_ack", 4096, 999, "untagged-B1"))
        rows = {r["identity"]: r for r in b4.framework_rows(recs)}
        r = rows["cpu-cpu-s0"]
        self.assertEqual((r["native_n"], r["torch_n"]), (4, 4))
        self.assertAlmostEqual(r["native_median_us"], 100.5)
        self.assertAlmostEqual(r["torch_median_us"], 145.5)
        self.assertAlmostEqual(r["difference_us"], 45.0)
        self.assertEqual(r["native_repetition_medians_us"], "100.0;104.0;101.0;99.0")
        self.assertNotIn("untagged", " ".join(rows))

    def test_a_bimodal_arm_is_flagged_not_averaged_away(self):
        recs = []
        for i, v in enumerate((127, 249, 126, 250)):
            recs.append(rec("tensor_transfer", "end_to_end", 4096, v, f"b1-x-s0-A{i}"))
        recs.append(rec("torch_p2p", "p2p_ack", 4096, 240, "b1-x-s0-B1"))
        row = b4.framework_rows(recs)[0]
        self.assertTrue(row["native_bimodal"])
        self.assertFalse(row["torch_bimodal"])

    def test_ddp_rows_keep_per_rank_phases_and_flag_imbalance(self):
        ex = {"model": "m", "grad_bytes_per_step": 4202496, "compute_imbalance_ratio": 5.8, "slowest_compute_rank": 1, "ddp_bucket_sizes": "1,2", "correctness_max_grad_rel_err": 1e-7,
              "correctness_max_param_rel_err": 1e-8, "correctness_cross_rank_max_abs_diff": 0.0}
        for rk, (f, b, o, ns, nb, no, nf, exp) in enumerate(((4e3, 120e3, 500, 13e3, 12.8e3, 100, 100, 108e3), (30e3, 90e3, 1e3, 75e3, 70e3, 1e3, 3e3, 20e3))):
            ex.update({f"rank{rk}_device": "x", f"rank{rk}_forward_median_us": f, f"rank{rk}_backward_median_us": b, f"rank{rk}_optimizer_median_us": o, f"rank{rk}_phased_step_median_us": f + b + o,
                       f"rank{rk}_nosync_forward_median_us": nf, f"rank{rk}_nosync_backward_median_us": nb, f"rank{rk}_nosync_optimizer_median_us": no, f"rank{rk}_nosync_step_median_us": ns,
                       f"rank{rk}_exposed_communication_inferred_us": exp, f"rank{rk}_peak_memory_bytes": 2**20 * 100, f"rank{rk}_cpu_cores": 1.0})
        r = rec("torch_ddp", "train", 4202496, 128000.0, "M-b2048-bk25", ("cuda", "host"), "t4b-ddp-compute", ex, {"layers": 8, "width": 1024, "batch": 2048})
        r["measurements"]["throughput"] = [{"kind": "training_samples_per_second", "value": 32000.0, "unit": "samples/s"}]
        row = b4.ddp_rows([r])[0]
        self.assertEqual((row["pairing"], row["tag"]), ("cuda/cpu", "M-b2048-bk25"))
        self.assertEqual(row["rank0_nosync_step_us"], 13e3)
        self.assertEqual(row["rank1_exposed_comm_inferred_us"], 20e3)
        self.assertAlmostEqual(row["steps_per_s"], 1e6 / 128000.0)
        self.assertEqual(row["compute_imbalance_ratio"], 5.8)


if __name__ == "__main__":
    unittest.main()
