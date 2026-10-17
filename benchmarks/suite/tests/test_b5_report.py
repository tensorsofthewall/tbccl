"""The B5 report module on synthetic runs: tables are derived from the raw timelines, percentiles with too few observations are withheld, closed-loop modes stay separate, order
comparisons carry the pass-to-pass spread, and the figures are produced (when matplotlib is installed) without plotting withheld values."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite import schema
from benchmarks.suite.reporting import b5
from benchmarks.suite.vllm_bench import metrics


def request(index, phase, ttft_ms, itl_ms, tokens, status="ok"):
    tl = metrics.Timeline({"phase": phase, "round": 0, "slot": 0, "request_index": index, "prompt": "p128", "prompt_tokens_planned": 128, "max_tokens": tokens}, 0)
    t = ttft_ms
    for k in range(1, tokens + 1):
        tl.on_event("data", {"choices": [{"token_ids": [k], "text": "x"}], "usage": {"prompt_tokens": 128, "completion_tokens": k}}, int(t * 1e6))
        t += itl_ms
    tl.on_event("done", None, int(t * 1e6))
    tl.finish(status, int(t * 1e6))
    return tl.to_record()


def make_run(root, label, config, phases, tier="t2-x", repetition=0, gpu=True):
    """phases: {name: (kind, extra_summary, [requests])}. config: ('cuda','metal') etc. Returns the run directory."""
    run_id = f"20261017T000000Z-b5-{label}-0001"
    run = Path(root) / run_id
    point = "vllm-latency-x"
    rep = run / "raw" / point / f"rep{repetition}"
    (rep / "vllm").mkdir(parents=True)
    raw_files, summaries = {}, {}
    for name, (kind, extra, recs) in phases.items():
        (rep / "vllm" / f"{name}.requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        raw_files[name] = {"requests": f"vllm/{name}.requests.jsonl"}
        summaries[name] = {"kind": kind, "phase": name, **extra}
    stages = [{"stage": i, "host": f"host-{'ab'[i]}", "device": d} for i, d in enumerate(config)]
    samples = {f"host-{'ab'[i]}": {"samples": 10, "cpu_percent": {"median": 12.0, "max": 40.0}, "mem_used_percent_max": 40.0, "process_rss_mib_max": 1500,
                                   "gpu": ({"temperature_c_max": 58.0, "utilization_percent": {"median": 20.0}, "memory_used_mib_max": 3400.0, "throttle_reasons_seen": ["0x1"]} if config[i] == "cuda" and gpu else None),
                                   "cpu_thermal": {"x86_pkg_temp_max": 96.0, "package_throttle_count_first": 1, "package_throttle_count_last": 51} if config[i] == "cuda" else None,
                                   "link_delta": {"tx_bytes": 1000, "rx_bytes": 10, "rx_errors": 0, "tx_errors": 0, "rx_dropped": 0, "tx_dropped": 0}, "engine_lost": False} for i in range(len(config))}
    ev = {"hosts": {f"host-{'ab'[i]}": {"device": d} for i, d in enumerate(config)}, "samples": samples}
    meas = {"timing": {"metric": "m", "unit": "us", "primary_statistic": "median", "statistics": {"median": 1.0}, "statistics_provenance": {"median": "computed"}},
            "throughput": [], "extra": {"inference": {"phases": summaries, "raw_files": raw_files, "transport_evidence": ev, "placement": {"api_host": "host-a", "stages": stages}}}, "warnings": []}
    record = schema.new_record(run_id=run_id, record_id=f"{run_id}/{point}/rep{repetition}", benchmark={"name": "vllm_latency", "adapter": "vllm_latency", "category": "vllm_latency", "operation": "single_request",
                                                                                                         "source_repository": "x", "provenance": "native"},
                               environment=schema.empty_environment(), configuration={"payload_bytes": 1, "payload_semantics": "n/a", "rank_count": len(config), "repetitions": 1, "repetition_index": repetition,
                                                                                     "timing_method": "t", "completion_semantics": "c", "inference": {"stage_devices": list(config)}},
                               measurements=meas, artifacts={"ranks": [{"role": "api-node", "supervised": False, "cleanup_confirmed": True, "terminated_by_runner": True}]})
    (run / "results.jsonl").write_text(json.dumps(record) + "\n")
    (run / "manifest.json").write_text(json.dumps({"status": "success", "counts": {"success": 1}, "reason": None}))
    return run


def cell(n, ttft, itl, tokens, name):
    return [request(i, name, ttft + i * 0.1, itl, tokens) for i in range(n)]


def lat_phases(scale=1.0, passes=3, n=10):
    out = {}
    for k in range(1, passes + 1):
        for p, ttft in ((16, 30), (128, 55), (512, 150)):
            for o in (1, 32, 64):
                name = f"lat-r{k}-p{p}-o{o}"
                out[name] = ("rounds", {"loop": "closed-rounds", "concurrency": 1, "prompt": f"p{p}", "max_tokens": o}, cell(n, ttft * scale * (1 + 0.01 * k), 20 * scale, o, name))
    return out


class Report(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.root, True)

    def campaign(self):
        make_run(self.root, "t2-cm", ("cuda", "metal"), lat_phases(1.0))
        make_run(self.root, "t2-mc", ("metal", "cuda"), lat_phases(1.3))
        return b5.load(self.root, "b5")

    def test_load_names_the_tier_and_requires_a_manifest(self):
        runs = self.campaign()
        self.assertEqual([r[0] for r in runs], ["t2-cm", "t2-mc"])
        (self.root / "stray").mkdir()
        self.assertEqual(len(b5.load(self.root, "b5")), 2)

    def test_cells_pool_the_passes_and_report_the_pass_to_pass_spread(self):
        runs = self.campaign()
        cells = b5.cell_rows(runs, b5.phase_rows(runs))
        c = next(x for x in cells if x["config"] == "cuda>metal" and x["prompt_tokens"] == 128 and x["output_tokens"] == 32)
        self.assertEqual((c["passes"], c["ttft_n"], c["tpot_n"], c["itl_n"]), (3, 30, 30, 30 * 31))
        self.assertAlmostEqual(c["tpot_p50_ms"], 20.0, places=3)
        self.assertGreater(c["ttft_pass_spread"], 1.0)
        self.assertEqual(len(c["ttft_pass_medians_ms"].split(";")), 3)
        one = next(x for x in cells if x["config"] == "cuda>metal" and x["output_tokens"] == 1 and x["prompt_tokens"] == 16)
        self.assertIsNone(one["tpot_p50_ms"])  # undefined for a one-token output, not zero
        self.assertEqual(one["tpot_undefined"], 30)

    def test_percentiles_without_enough_observations_are_withheld(self):
        runs = self.campaign()
        cells = b5.cell_rows(runs, b5.phase_rows(runs))
        c = next(x for x in cells if x["prompt_tokens"] == 128 and x["output_tokens"] == 32)
        self.assertIsNotNone(c["ttft_p95_ms"])  # 30 observations
        self.assertIsNone(c["ttft_p99_ms"])  # fewer than 100
        self.assertIsNotNone(c["itl_p99_ms"])  # 930 pooled intervals

    def test_order_comparison_carries_the_spread_and_only_flags_real_differences(self):
        runs = self.campaign()
        ratios = b5.ratio_rows(b5.cell_rows(runs, b5.phase_rows(runs)))
        r = next(x for x in ratios if x["a"] == "cuda>metal" and x["b"] == "metal>cuda" and x["metric"] == "tpot" and x["prompt_tokens"] == 128 and x["output_tokens"] == 32)
        self.assertAlmostEqual(r["ratio_b_over_a"], 1.3, places=2)
        self.assertTrue(r["exceeds_variation"])
        # identical runs: the ratio is 1 and nothing is claimed
        root2 = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root2, True)
        make_run(root2, "t2-cm", ("cuda", "metal"), lat_phases(1.0))
        make_run(root2, "t2-mc", ("metal", "cuda"), lat_phases(1.0))
        runs2 = b5.load(root2, "b5")
        flat = b5.ratio_rows(b5.cell_rows(runs2, b5.phase_rows(runs2)))
        self.assertTrue(flat and not any(x["exceeds_variation"] for x in flat))

    def test_concurrency_rows_keep_rounds_and_continuous_loops_apart(self):
        phases = {}
        for mode, kind in (("rounds", "rounds"), ("cont", "continuous")):
            for c in (1, 4):
                name = f"conc-r1-c{c}-{mode}"
                recs = cell(8 * c, 100, 20, 32, name)
                phases[name] = (kind, {"loop": "closed", "concurrency": c, "prompt": "p128", "max_tokens": 32}, recs)
        make_run(self.root, "t3-cm", ("cuda", "metal"), phases, tier="t3")
        runs = b5.load(self.root, "b5")
        rows = b5.concurrency_rows(b5.phase_rows(runs))
        self.assertEqual({(r["mode"], r["concurrency"]) for r in rows}, {("rounds", 1), ("rounds", 4), ("cont", 1), ("cont", 4)})
        self.assertTrue(all(r["output_tokens_per_s_median"] > 0 and r["failed"] == 0 for r in rows))

    def test_failed_requests_are_counted_not_averaged_in(self):
        recs = cell(9, 50, 10, 8, "x") + [request(99, "x", 0, 10, 0, status="timeout")]
        recs[-1]["events"], recs[-1]["usage"] = [], None
        make_run(self.root, "t2-cm", ("cuda", "metal"), {"lat-r1-p128-o8": ("rounds", {"concurrency": 1}, recs)})
        runs = b5.load(self.root, "b5")
        row = b5.phase_rows(runs)[0]
        self.assertEqual((row["requests"], row["ok"], row["timeouts"]), (10, 9, 1))
        self.assertAlmostEqual(row["error_rate"], 0.1)
        self.assertEqual(row["ttft_n"], 9)

    def test_resources_record_gpu_cpu_thermal_and_link_counters(self):
        runs = self.campaign()
        rows = b5.resource_rows(runs)
        a = next(r for r in rows if r["host"] == "host-a" and r["device"] == "cuda")
        self.assertEqual((a["gpu_temperature_c_max"], a["gpu_memory_used_mib_max"], a["cpu_pkg_temp_max_c"], a["cpu_package_throttle_events"]), (58.0, 3400.0, 96.0, 50))

    def test_figures_use_only_available_values_and_name_every_requested_figure(self):
        if b5.b3b._plt() is None:
            self.skipTest("matplotlib is not installed")
        runs = self.campaign()
        phases = {}
        for c in (1, 2, 4):
            for mode, kind in (("rounds", "rounds"), ("cont", "continuous")):
                name = f"conc-r1-c{c}-{mode}"
                phases[name] = (kind, {"loop": "x", "concurrency": c, "prompt": "p128", "max_tokens": 32}, cell(8 * c, 100 + 20 * c, 20, 32, name))
        make_run(self.root, "t3-cm", ("cuda", "metal"), phases, tier="t3")
        runs = b5.load(self.root, "b5")
        rows = b5.phase_rows(runs)
        cells = b5.cell_rows(runs, rows)
        out = self.root / "fig"
        made = b5.figures(out, cells, b5.concurrency_rows(rows), b5.itl_samples(runs, 128, 64), b5.ratio_rows(cells), b5.resource_rows(runs))
        names = {Path(m).name for m in made}
        for want in ("ttft_vs_input_tokens.png", "tpot_vs_output_tokens.png", "itl_distributions.png", "throughput_vs_concurrency.png", "ttft_vs_throughput.png", "tpot_vs_throughput.png",
                     "e2e_vs_concurrency.png", "order_ttft_ratio.png", "order_tpot_ratio.png", "gpu_memory.png"):
            self.assertIn(want, names)
            self.assertGreater((out / want).stat().st_size, 1000)

    def test_trace_events_are_summarised_per_process_and_the_decomposition_never_sums_stages(self):
        run = make_run(self.root, "t8-trace", ("cuda", "metal"), {"lat-r1-p128-o32": ("rounds", {"concurrency": 1, "prompt": "p128", "max_tokens": 32}, cell(10, 50, 20, 32, "lat-r1-p128-o32"))}, tier="t8")
        rec = json.loads((run / "results.jsonl").read_text())
        rep = b5.rep_dir(run, rec)
        (rep / "trace" / "host-a").mkdir(parents=True)
        (rep / "trace" / "host-b").mkdir(parents=True)
        hook = [{"kind": "step", "entry_ns": 0, "done_ns": 4_500_000, "tokens": 1, "requests": 1}] * 5 + [{"kind": "step", "entry_ns": 0, "done_ns": 30_000_000, "tokens": 128, "requests": 1}] \
            + [{"kind": "step", "entry_ns": 0, "done_ns": 5_000_000, "tokens": 4, "requests": 4}, {"kind": "isend", "entry_ns": 0, "done_ns": 130_000, "bytes": 4096}]
        (rep / "trace" / "host-a" / "hook.1.jsonl").write_text("".join(json.dumps(e) + "\n" for e in hook))
        (rep / "trace" / "host-b" / "tbccl.2.json").write_text(json.dumps([{"op": "step_forward", "entry_ns": 0, "done_ns": 13_000_000, "bytes": 0}, {"op": "step_sample", "entry_ns": 0, "done_ns": 4_000_000, "bytes": 0},
                                                                       {"op": "metal_recv_upstream", "entry_ns": 0, "done_ns": 6_000_000, "bytes": 4096}]))
        rec["measurements"]["extra"]["inference"]["trace"] = {"files": {"host-a": ["trace/host-a/hook.1.jsonl"], "host-b": ["trace/host-b/tbccl.2.json"]}}
        (run / "results.jsonl").write_text(json.dumps(rec) + "\n")
        runs = b5.load(self.root, "b5")
        rows = b5.trace_rows(runs)
        by = {(r["host"], r["event"], r["step_class"]): r for r in rows}
        self.assertEqual(by[("host-a", "step", "decode1")]["n"], 5)
        self.assertAlmostEqual(by[("host-a", "step", "decode1")]["dur_p50_ms"], 4.5)
        self.assertAlmostEqual(by[("host-a", "step", "prefill")]["dur_p50_ms"], 30.0)
        self.assertEqual(by[("host-a", "step", "decode")]["n"], 1)  # four tokens for four requests: decode, not single-request decode
        self.assertEqual(by[("host-b", "metal_recv_upstream", "all")]["bytes_p50"], 4096)
        d = b5.decomposition_rows(runs, rows)[0]
        self.assertAlmostEqual(d["client_tpot_p50_ms"], 20.0, places=3)
        self.assertAlmostEqual(d["stage0_cuda_step_decode1_p50_ms"], 4.5)
        self.assertAlmostEqual(d["stage1_metal_step_forward_p50_ms"], 13.0)
        self.assertFalse(any("total" in k or "sum" in k for k in d))  # components are listed side by side, never added up
        self.assertEqual(b5.step_class({"tokens": 2, "requests": 1}), "prefill")
        if b5.b3b._plt() is not None:
            made = b5.figures(self.root / "fig2", [], [], {}, [], [], None, b5.decomposition_rows(runs, rows))
            self.assertIn("stage_components.png", {Path(m).name for m in made})

    def test_server_side_metrics_give_queue_depth_and_phase_means_from_histogram_deltas(self):
        run = make_run(self.root, "t3-cm", ("cuda", "metal"), {"conc-r1-c4-cont": ("continuous", {"concurrency": 4, "prompt": "p128", "max_tokens": 32}, cell(8, 100, 20, 32, "conc-r1-c4-cont"))}, tier="t3")
        rec = json.loads((run / "results.jsonl").read_text())
        rep = b5.rep_dir(run, rec)
        names = ["vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds"]
        def sample(t, running, waiting, scale):
            v = {'vllm:num_requests_running{engine="0"}': running, 'vllm:num_requests_waiting{engine="0"}': waiting, 'vllm:num_requests_waiting_by_reason{engine="0",reason="x"}': 99.0, 'vllm:kv_cache_usage_perc{engine="0"}': 0.03}
            for n in names:
                v[n + '_sum{engine="0"}'] = 0.5 * scale
                v[n + '_count{engine="0"}'] = 10.0 * scale
            return {"t_ns": t, "values": v}
        (rep / "vllm" / "conc-r1-c4-cont.server-metrics.jsonl").write_text(json.dumps(sample(1, 4.0, 1.0, 1)) + "\n" + json.dumps(sample(2, 3.0, 0.0, 3)) + "\n")
        rec["measurements"]["extra"]["inference"]["raw_files"]["conc-r1-c4-cont"]["server_metrics"] = "vllm/conc-r1-c4-cont.server-metrics.jsonl"
        (run / "results.jsonl").write_text(json.dumps(rec) + "\n")
        row = b5.server_rows(b5.load(self.root, "b5"))[0]
        self.assertEqual((row["running_max"], row["waiting_max"], row["scrapes"]), (4.0, 1.0, 2))  # the by-reason gauge is not double counted
        self.assertAlmostEqual(row["server_queue_mean_ms"], 50.0)  # (1.5 - 0.5) s over (30 - 10) requests
        self.assertEqual(row["server_queue_n"], 20.0)
        self.assertIsNone(row["server_ttft_mean_ms"])  # a family the server did not report

    def test_confirmation_lifecycles_are_summarised_per_configuration_with_their_spread(self):
        for i, (config, scale) in enumerate(((("cuda", "metal"), 1.0), (("metal", "cuda"), 0.75), (("cuda", "metal"), 1.02), (("metal", "cuda"), 0.74))):
            phases = {f"lat-r1-p128-o32": ("rounds", {"concurrency": 1, "prompt": "p128", "max_tokens": 32}, cell(10, 50 * scale, 20 * scale, 32, "lat-r1-p128-o32")),
                      "conc-r1-c4-cont": ("continuous", {"concurrency": 4, "prompt": "p128", "max_tokens": 32}, cell(16, 100, 20 * scale, 32, "conc-r1-c4-cont"))}
            make_run(self.root, f"t9-c{i}", config, phases, tier="t9")
        rows = b5.confirm_rows(b5.load(self.root, "b5"))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["config"] for r in rows], ["cuda>metal", "metal>cuda", "cuda>metal", "metal>cuda"])
        summary = {(x["config"], x["figure"]): x for x in b5.confirm_summary(rows)}
        a = summary[("cuda>metal", "tpot_p128o32")]
        b = summary[("metal>cuda", "tpot_p128o32")]
        self.assertEqual(a["lifecycles"], 2)
        self.assertLess(b["max"], a["min"])  # the ranges do not overlap: a difference beyond the lifecycle-to-lifecycle spread
        self.assertAlmostEqual(a["spread_max_over_min"], 1.02, places=2)

    def test_boundary_ping_pong_points_have_their_own_table_and_do_not_break_the_others(self):
        run = make_run(self.root, "t6b-backend", ("cuda", "host"), {}, tier="t6b")
        rec = json.loads((run / "results.jsonl").read_text())
        inf = rec["measurements"]["extra"]["inference"]
        inf.pop("phases"), inf.pop("raw_files"), inf.pop("transport_evidence"), inf.pop("placement")
        inf["devices"] = ["cuda:0", "cpu"]
        inf["boundary"] = {"1": {"bytes_per_direction": 4096, "rtt_us": {"p50": 640.0, "p95": 790.0, "min": 600.0, "max": 900.0, "mean": 650.0, "n": 300}}}
        (run / "results.jsonl").write_text(json.dumps(rec) + "\n")
        runs = b5.load(self.root, "b5")
        rows = b5.backend_rows(runs)
        self.assertEqual((rows[0]["tokens"], rows[0]["rtt_p50_us"], rows[0]["rank0_device"]), (1, 640.0, "cuda:0"))
        self.assertFalse([k for k in rows[0] if "half" in k or "one_way" in k])  # a round trip is never turned into a one-way figure
        self.assertEqual(b5.phase_rows(runs), [])
        self.assertEqual(b5.cell_rows(runs, []), [])
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.assertEqual(b5.main([str(self.root), "b5", "--out", str(self.root / "o2")]), 0)

    def test_main_writes_tables_and_an_exclusion_list(self):
        self.campaign()
        bad = self.root / "20261017T000000Z-b5-t9-zz-0002"
        bad.mkdir()
        (bad / "results.jsonl").write_text("")
        (bad / "manifest.json").write_text(json.dumps({"status": "aborted", "counts": {}, "reason": "health"}))
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.assertEqual(b5.main([str(self.root), "b5", "--out", str(self.root / "out")]), 0)
        for table in ("phases", "latency_cells", "concurrency", "order_and_pairing_ratios", "resources", "lifecycle"):
            self.assertTrue((self.root / "out" / "tables" / f"{table}.csv").exists() or table == "concurrency", table)
        excluded = json.loads((self.root / "out" / "excluded_runs.json").read_text())
        self.assertEqual(excluded[0]["status"], "aborted")

    def test_tables_are_written_without_matplotlib(self):
        import io
        from contextlib import redirect_stdout
        from unittest import mock
        self.campaign()
        with mock.patch.object(b5.b3b, "_have_mpl", return_value=False), redirect_stdout(io.StringIO()):
            self.assertEqual(b5.main([str(self.root), "b5", "--out", str(self.root / "no-figures")]), 0)
        self.assertTrue((self.root / "no-figures" / "tables" / "phases.csv").exists())
        figures = self.root / "no-figures" / "figures"
        self.assertEqual(list(figures.glob("*.png")) if figures.exists() else [], [])


if __name__ == "__main__":
    unittest.main()
