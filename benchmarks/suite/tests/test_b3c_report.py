"""The B3C campaign report tables: matched host controls, overlap bounds, sync-vs-async pairing and the observed-minus-local-staging column."""
import unittest

from benchmarks.suite.reporting import b3c


def record(bench, op, tier, pair, size, median, *, source="host-a", orient=("host-a", "host-b"), extra=None, options=None, algo=None, per_rank=None, iterations=30, throughput=None, samples=None):
    hosts = [{"host_id": "host-a", "os": "EndeavourOS (kernel 6)"}, {"host_id": "host-b", "os": "macOS 26"}]
    backends = dict(zip(orient, pair))
    dest = "host-b" if source == "host-a" else "host-a"
    return {"_tier": tier, "status": {"state": "success"}, "verification": {"status": "passed"},
            "benchmark": {"name": bench, "operation": op, "algorithm_requested": algo},
            "configuration": {"payload_bytes": size, "iterations": iterations, "options": options or {}, "rank_order": list(orient),
                              "data_direction": {"mode": "directional", "source_host": source, "destination_host": dest}},
            "environment": {"hosts": hosts, "memory_backend": list(pair)},
            "artifacts": {"ranks": [{"rank": i, "host_id": h, "memory_backend": backends[h]} for i, h in enumerate(orient)]},
            "measurements": {"timing": {"metric": "m", "statistics": {"median": median, "p95": median * 1.1}, "primary_statistic": "median", "sample_count": iterations, "samples": samples, "per_rank": per_rank},
                             "throughput": [{"kind": "k", "value": throughput}] if throughput else [], "extra": {k: {"value": v, "unit": "us"} for k, v in (extra or {}).items()}}}


class Tables(unittest.TestCase):
    def test_incremental_cost_uses_the_host_control_of_the_same_run_and_the_local_staging_sum(self):
        tier = "b3c1:t1-transfer"
        recs = [record("tensor_transfer", "end_to_end", tier, ("host", "host"), 1048576, 800.0 + d, extra={}) for d in (0, 2, 4)]
        recs += [record("tensor_transfer", "end_to_end", tier, ("cuda-pinned", "metal-shared"), 1048576, 900.0 + d) for d in (0, 2, 4)]
        staging = [{"host_os": "Linux", "memory_backend": "cuda-pinned", "operation": "staging_d2h", "payload_bytes": 1048576, "median_us": 84.0},
                   {"host_os": "macOS", "memory_backend": "metal-shared", "operation": "staging_h2d", "payload_bytes": 1048576, "median_us": 0.0}]
        rows = {r["pair"]: r for r in b3c.transfer_rows(recs, None, staging)}
        dev = rows["cuda-pinned -> metal-shared"]
        self.assertAlmostEqual(dev["incremental_vs_host_us"], 100.0)
        self.assertAlmostEqual(dev["ratio_vs_host"], 902.0 / 802.0)
        self.assertEqual(dev["local_staging_sum_us"], 84.0)
        self.assertAlmostEqual(dev["incremental_minus_local_staging_us"], 16.0)
        self.assertNotIn("incremental_vs_host_us", rows["host -> host"])

    def test_host_control_is_never_taken_from_another_direction_or_run(self):
        a = record("tensor_transfer", "end_to_end", "b3c1:t1-transfer", ("host", "host"), 4096, 100.0)
        b = record("tensor_transfer", "end_to_end", "b3c1:t1-transfer", ("cuda-pinned", "metal-shared"), 4096, 200.0, source="host-b")  # other direction
        c = record("tensor_transfer", "end_to_end", "b3c1:t5-confirm", ("cuda-pinned", "metal-shared"), 4096, 200.0)  # other run
        rows = b3c.transfer_rows([a, b, c])
        self.assertTrue(all("incremental_vs_host_us" not in r for r in rows))

    def test_a_zero_staging_median_is_flagged_not_turned_into_throughput(self):
        rec = record("tensor_staging", "staging_d2h", "b3c2:staging-mac", ("metal-shared",), 4096, 0.0, orient=("host-b",))
        rec["measurements"]["throughput"] = []
        row = b3c.staging_rows([rec])[0]
        self.assertTrue(row["below_clock_resolution"])
        self.assertIsNone(row["throughput_GBps"])

    def test_overlap_is_derived_from_measured_totals_and_bad_fractions_are_flagged(self):
        def rows(compute, comm, serial, overlap):
            recs = []
            for op, v in (("compute_only", compute), ("comm_only", comm), ("serial", serial), ("overlap", overlap)):
                recs += [record("cuda_overlap", op, "b3c1:t3-overlap", ("cuda-chunked", "metal-shared"), 4194304, v + d, options={"compute_rounds": 300, "bucket_count": 4}, extra={"enqueue_median_us": 1.0}) for d in (0, 1, 2)]
            return b3c.cuda_overlap_rows(recs)[0]
        good = rows(6000.0, 8700.0, 14700.0, 10000.0)
        self.assertTrue(good["valid"], good["invalid_reason"])
        self.assertAlmostEqual(good["time_saved_us"], 4700.0)
        self.assertAlmostEqual(good["overlap_fraction_of_smaller_phase"], 4700.0 / 6001.0, places=3)
        self.assertGreater(good["serial_over_overlap"], 1.4)
        self.assertAlmostEqual(good["pipeline_bound_us"], 8701.0 + 6001.0 / 4, places=1)
        too_good = rows(6000.0, 8700.0, 14700.0, 5000.0)  # faster than either phase alone: impossible
        self.assertFalse(too_good["valid"])
        self.assertIn("outside [0, 1]", too_good["invalid_reason"])
        self.assertGreater(too_good["overlap_fraction_of_smaller_phase"], 1)  # reported as measured, not clipped
        negative = rows(6000.0, 8700.0, 14700.0, 15500.0)
        self.assertFalse(negative["valid"])
        self.assertLess(negative["overlap_fraction_of_smaller_phase"], 0)

    def test_sync_vs_async_pairs_identical_memory_and_labels_the_chunked_comparison(self):
        sync = [{"tier": "b3c1:t1-transfer", "operation": "end_to_end", "timing_scope": "ready", "pair": p, "direction": "Linux -> macOS", "orientation": "o", "payload_bytes": 1048576,
                 "median_us": m, "median_range_rel": 0.01} for p, m in (("cuda-pinned -> metal-shared", 900.0), ("host -> host", 700.0))]
        asyn = [{"tier": "b3c1:t2-async", "pair": p, "direction": "Linux -> macOS", "orientation": "o", "chunk_bytes": ch, "metal_async_path": "adapter", "payload_bytes": 1048576, "completion_median_us": c,
                 "completion_range_rel": 0.02, "enqueue_median_us": 1.5} for p, ch, c in (("cuda-pinned -> metal-shared", 0, 990.0), ("cuda-chunked -> metal-shared", 262144, 800.0), ("host -> host", 0, 650.0))]
        out = {(r["pair_async"], r["chunk_bytes"]): r for r in b3c.sync_vs_async(sync, asyn)}
        self.assertEqual(out[("cuda-pinned -> metal-shared", 0)]["match"], "same memory pair")
        self.assertAlmostEqual(out[("cuda-pinned -> metal-shared", 0)]["async_over_sync"], 1.1)
        chunked = out[("cuda-chunked -> metal-shared", 262144)]
        self.assertEqual(chunked["pair_sync"], "cuda-pinned -> metal-shared")
        self.assertIn("not the same memory", chunked["match"])
        self.assertAlmostEqual(chunked["async_over_sync"], 800.0 / 900.0)

    def test_hetero_pairs_async_with_sync_and_keeps_both_ranks(self):
        recs = []
        for algo, m in (("sync", 1000.0), ("async", 1500.0)):
            for d in (0, 5):
                recs.append(record("hetero_allreduce", "all_reduce", "b3c1:t4-hetero", ("cuda", "metal-shared"), 1048576, m + d, algo=algo,
                                   per_rank=[{"rank": 0, "statistics": {"median": m}}, {"rank": 1, "statistics": {"median": m + d}}]))
        rows = {r["algorithm"]: r for r in b3c.hetero_rows(recs)}
        self.assertAlmostEqual(rows["async"]["async_over_sync"], 1502.5 / 1002.5)
        self.assertEqual(rows["sync"]["memory_pair"], "cuda / metal-shared")
        self.assertEqual(rows["sync"]["rank0_median_us"], 1000.0)

    def test_raw_samples_show_a_configuration_that_is_bimodal_across_repetitions(self):
        recs = [record("tensor_transfer", "end_to_end", "b3c1:t6-diag4k", ("cuda-pinned", "host"), 4096, m, samples=[m - 1, m, m + 1]) for m in (127.0, 249.0, 249.0, 128.0, 248.0)]
        row = b3c.sample_mode_rows(recs)[0]
        self.assertEqual((row["repetitions"], row["repetitions_above_split"], row["samples_pooled"]), (5, 3, 15))
        self.assertAlmostEqual(row["fraction_samples_above_split"], 9 / 15)
        self.assertEqual(row["repetition_medians_us"], "127.0;249.0;249.0;128.0;248.0")
        self.assertEqual(b3c.sample_mode_rows([record("tensor_transfer", "end_to_end", "t", ("host", "host"), 4096, 100.0)]), [])  # no raw samples, no row

    def test_run_to_run_ratio(self):
        rows = [{"tier": "a", "k": 1, "v": 100.0, "s": 0.01, "repetitions": 3}, {"tier": "b", "k": 1, "v": 110.0, "s": 0.02, "repetitions": 5}, {"tier": "b", "k": 2, "v": 5.0, "s": 0.0, "repetitions": 5}]
        out = b3c.run_to_run(rows, "a", "b", ("k",), "v", "s")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["ratio_confirm_over_main"], 1.1)


if __name__ == "__main__":
    unittest.main()
