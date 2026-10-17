"""The B6 report module on synthetic Exo runs: tables come from the raw timelines, the lifecycle is the replicated unit, backends are compared only inside an identical placement and with
at least two lifecycles on each side, CPU seconds and link bytes are taken over the measured window, and gates, exclusions and figures are produced."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite import schema
from benchmarks.suite.stream import exo_metrics as metrics
from benchmarks.suite.reporting import b6
from benchmarks.suite.tests.common import approx

T0 = 1_792_000_000.0  # wall clock of the first request of a synthetic lifecycle


def request(index, phase, ttft_ms, itl_ms, tokens, wall):
    tl = metrics.ExoTimeline({"phase": phase, "index": index, "max_tokens": tokens, "wall_submit": wall}, 0)
    t = ttft_ms
    for k in range(1, tokens + 1):
        ev = {"choices": [{"index": 0, "delta": {"content": "x"}}]}
        if k == tokens:
            ev["choices"][0]["finish_reason"] = "length"
            ev["usage"] = {"prompt_tokens": 20, "completion_tokens": tokens}
        tl.on_event("data", ev, int(t * 1e6))
        t += itl_ms
    tl.on_event("done", None, int(t * 1e6))
    tl.finish("ok", int(t * 1e6))
    return tl.to_record()


def samples(host, wall0, seconds, cpu_per_proc=50.0, procs=2, tx0=0):
    rows = []
    for i in range(seconds):
        wall = wall0 + i
        rows.append({"role": host, "t_mono_s": float(i), "wall_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(wall)),
                     "processes": [{"pid": p, "role": "runner", "cpu_percent": cpu_per_proc, "threads": 10, "rss_mib": 100} for p in range(procs)],
                     "link": {"tx_bytes": tx0 + 1000 * i, "rx_bytes": 500 * i, "rx_errors": 0, "tx_errors": 0, "rx_dropped": 0, "tx_dropped": 0},
                     "gpu": {"utilization_gpu_percent": "30"} if host == "host-a" else None})
    return rows


def make_run(root, label, backend, stage_hosts=("host-a", "host-b"), layers=(9, 19), tpot_ms=10.0, ttft_ms=100.0, cpu=50.0, state="success", n=6, point="lat-1", variant="default"):
    run_id = f"20261017T000000Z-b6-{label}-0001"
    run = Path(root) / run_id
    rep = run / "raw" / point / "rep0"
    (rep / "exo").mkdir(parents=True)
    phases, raw_files, wall = {}, {}, T0
    for o in (1, 16):
        for k, prompt in enumerate(("p16", "p128")):
            name = f"lat-r1-{prompt}-o{o}"
            recs = []
            for i in range(n):
                recs.append(request(i, name, ttft_ms * (1 + k), tpot_ms, o, wall))
                wall += 1.0
            (rep / "exo" / f"{name}.requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            raw_files[name] = {"requests": f"exo/{name}.requests.jsonl"}
            phases[name] = {"kind": "rounds", "phase": name}
    gate = {"kind": "check", "phase": "gate", "check": {"passed": True, "outcome": {"exact": 3, "near_tie": 1}, "tie_margin": 0.251}}
    phases["gate"] = gate
    files = {}
    for host in stage_hosts:
        p = rep / "exo" / f"samples-{host}.jsonl"
        p.write_text("".join(json.dumps(s) + "\n" for s in samples(host, T0 - 5, int(wall - T0) + 12, cpu_per_proc=cpu)))
        files[host] = f"exo/{p.name}"
    raw_files["samples"] = files
    inference = {"phases": phases, "raw_files": raw_files, "load_s": 12.0, "placement": {"stages": [{"host": h, "n_layers": n_} for h, n_ in zip(stage_hosts, layers)]}}
    record = schema.new_record(run_id=run_id, record_id=f"{run_id}/{point}/rep0", benchmark={"name": "exo_latency", "adapter": "exo_latency", "category": "exo_latency", "operation": "single_request",
                                                                                                "source_repository": "x", "provenance": "native"},
                               environment=schema.empty_environment(), configuration={"payload_bytes": 1, "payload_semantics": "n/a", "rank_count": 2, "repetitions": 1, "repetition_index": 0,
                                                                                     "timing_method": "t", "completion_semantics": "c",
                                                                                     "inference": {"backend": backend, "stage_hosts": list(stage_hosts), "stage_layers": list(layers), "pipeline": True, "variant": variant}},
                               measurements={"timing": {"metric": "m", "unit": "us", "primary_statistic": "median", "statistics": {"median": 1.0}, "statistics_provenance": {"median": "computed"}},
                                             "throughput": [], "extra": {"inference": inference}, "warnings": []},
                               artifacts={"ranks": [{"role": "exo-node", "supervised": False, "cleanup_confirmed": True, "terminated_by_runner": True}]})
    if state != "success":
        record["status"]["state"] = state
        record["status"]["reason"] = "synthetic failure"
    (run / "results.jsonl").write_text(json.dumps(record) + "\n")
    (run / "manifest.json").write_text(json.dumps({"status": "success" if state == "success" else "aborted", "counts": {state: 1}, "reason": None}))
    return run


class Report(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.root, True)

    def campaign(self, ring=(10.0, 10.4), tbccl=(10.5, 11.0), **kw):
        for i, v in enumerate(ring):
            make_run(self.root, f"t4-r{i}", "MlxRing", tpot_ms=v, point=f"lat-{2 * i + 1}", **kw)
        for i, v in enumerate(tbccl):
            make_run(self.root, f"t4-t{i}", "MlxTbccl", tpot_ms=v, point=f"lat-{2 * i + 2}", **kw)
        return b6.load(self.root, "b6")

    def test_cells_are_per_lifecycle_and_derived_from_the_raw_timelines(self):
        runs = self.campaign()
        cells = b6.cell_rows(runs)
        self.assertEqual(len(cells), 4 * 4)  # four lifecycles x (2 prompts x 2 output lengths)
        row = next(c for c in cells if c["backend"] == "MlxRing" and c["prompt"] == "p16" and c["output_tokens_cfg"] == 16)
        approx(self, row["tpot_p50_ms"], 10.0)
        approx(self, row["ttft_p50_ms"], 100.0)
        self.assertEqual(row["n"], 6)
        one = next(c for c in cells if c["output_tokens_cfg"] == 1)
        self.assertIsNone(one["tpot_p50_ms"])  # a one-token output has no TPOT: undefined, never zero
        self.assertEqual(one["tpot_n"], 0)

    def test_ab_compares_identical_placements_with_enough_lifecycles(self):
        ab = b6.ab_rows(b6.cell_rows(self.campaign()))
        row = next(r for r in ab if r["metric"] == "tpot" and r["prompt"] == "p128" and r["output_tokens_cfg"] == 16)
        self.assertEqual((row["ring_lifecycles"], row["tbccl_lifecycles"]), (2, 2))
        approx(self, row["ratio_tbccl_over_ring"], 10.75 / 10.2)
        self.assertNotIn("not_compared", row)

    def test_one_lifecycle_per_side_is_not_compared(self):
        ab = b6.ab_rows(b6.cell_rows(self.campaign(ring=(10.0,), tbccl=(11.0,))))
        self.assertTrue(all("not_compared" in r and "ratio_tbccl_over_ring" not in r for r in ab))

    def test_different_placements_are_never_merged(self):
        make_run(self.root, "t4-a", "MlxRing", stage_hosts=("host-a", "host-b"), layers=(9, 19), point="lat-1")
        make_run(self.root, "t4-b", "MlxRing", stage_hosts=("host-a", "host-b"), layers=(9, 19), point="lat-2")
        make_run(self.root, "t4-c", "MlxTbccl", stage_hosts=("host-b", "host-a"), layers=(19, 9), point="lat-3")
        make_run(self.root, "t4-d", "MlxTbccl", stage_hosts=("host-b", "host-a"), layers=(19, 9), point="lat-4")
        ab = b6.ab_rows(b6.cell_rows(b6.load(self.root, "b6")))
        self.assertEqual({r["placement"] for r in ab}, {"host-a>host-b|9/19", "host-b>host-a|19/9"})
        self.assertTrue(all("not_compared" in r for r in ab))

    def test_a_difference_inside_the_lifecycle_spread_is_not_called_a_difference(self):
        ab = b6.ab_rows(b6.cell_rows(self.campaign(ring=(10.0, 12.0), tbccl=(10.5, 12.5))))
        row = next(r for r in ab if r["metric"] == "tpot" and r["prompt"] == "p16" and r["output_tokens_cfg"] == 16)
        self.assertFalse(row["exceeds_variation"])

    def test_resources_use_the_measured_window_and_divide_by_generated_tokens(self):
        runs = self.campaign(cpu=50.0)
        rows = b6.resource_rows(runs)
        self.assertEqual({r["host"] for r in rows}, {"host-a", "host-b"})
        r = next(x for x in rows if x["backend"] == "MlxRing" and x["host"] == "host-a")
        self.assertEqual(r["output_tokens"], 6 * (1 + 16) * 2)
        approx(self, r["cpu_seconds_per_output_token"] * r["output_tokens"], r["process_cpu_seconds"])
        approx(self, r["process_cpu_percent_median"], 100.0)  # two processes at 50 %
        self.assertEqual(r["gpu_utilization_median"], 30.0)
        self.assertGreater(r["link_bytes_per_output_token"], 0)

    def test_resource_ab_compares_cpu_per_token_between_backends(self):
        for i in range(2):
            make_run(self.root, f"t4-r{i}", "MlxRing", cpu=80.0, point=f"lat-{i}a")
            make_run(self.root, f"t4-t{i}", "MlxTbccl", cpu=40.0, point=f"lat-{i}b")
        ab = b6.resource_ab(b6.resource_rows(b6.load(self.root, "b6")))
        row = next(r for r in ab if r["host"] == "host-a")
        approx(self, row["cpu_seconds_per_output_token_ratio"], 0.5)

    def test_older_runs_without_sample_files_fall_back_to_the_raw_rank_output(self):
        run = make_run(self.root, "t4-old", "MlxRing")
        rec = json.loads((run / "results.jsonl").read_text())
        rec["measurements"]["extra"]["inference"]["raw_files"].pop("samples")
        (run / "results.jsonl").write_text(json.dumps(rec) + "\n")
        rep = run / "raw" / "lat-1" / "rep0"
        (rep / "rank2.stdout").write_text("".join("TBCCL_EXO_SAMPLE " + json.dumps(s) + "\n" for s in samples("host-a", T0 - 5, 40)))
        rows = b6.resource_rows(b6.load(self.root, "b6"))
        self.assertEqual([r["host"] for r in rows], ["host-a"])

    def test_failed_runs_are_listed_as_excluded_and_not_used(self):
        make_run(self.root, "t4-ok", "MlxRing", point="a")
        make_run(self.root, "t4-bad", "MlxTbccl", state="failure", point="b")
        everything = b6.load(self.root, "b6")
        used = [r for r in everything if r[2]["status"] == "success"]
        self.assertEqual(len(b6.excluded_rows(everything, used)), 1)
        self.assertEqual(sorted(r["state"] for r in b6.lifecycle_rows(everything)), ["failure", "success"])
        self.assertEqual(len(b6.cell_rows(everything)), 4)  # only the successful lifecycle contributes cells

    def test_phase_and_concurrency_rows_cover_the_sweep_and_keep_loops_apart(self):
        run = make_run(self.root, "t5-x", "MlxRing", point="c1")
        rec = json.loads((run / "results.jsonl").read_text())
        inf = rec["measurements"]["extra"]["inference"]
        rep = run / "raw" / "c1" / "rep0"
        for c, mode, kind in ((1, "rounds", "rounds"), (4, "cont", "continuous"), (4, "rounds", "rounds")):
            name = f"conc-r1-c{c}-{mode}"
            recs = [request(i, name, 100.0, 10.0, 8, T0 + i) for i in range(4 * c)]
            (rep / "exo" / f"{name}.requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
            inf["raw_files"][name] = {"requests": f"exo/{name}.requests.jsonl"}
            inf["phases"][name] = {"kind": kind, "phase": name, "conc": c}
        (run / "results.jsonl").write_text(json.dumps(rec) + "\n")
        rows = b6.phase_rows(b6.load(self.root, "b6"))
        conc = b6.concurrency_rows(rows)
        self.assertEqual({(x["conc"], x["loop"]) for x in conc}, {(1, "rounds"), (4, "continuous"), (4, "rounds")})
        self.assertTrue(all(x["output_tokens_per_s_median"] > 0 for x in conc))
        approx(self, conc[0]["tpot_p50_ms_median"], 10.0)
        self.assertTrue(any(x["phase"].startswith("lat-r1") for x in rows))  # the latency cells are phases too

    def test_a_fast_path_variant_never_enters_the_baseline_comparison(self):
        for i in range(2):
            make_run(self.root, f"t9-r{i}", "MlxRing", tpot_ms=10.0, point=f"r{i}")
            make_run(self.root, f"t9-d{i}", "MlxTbccl", tpot_ms=15.0, point=f"d{i}")
            make_run(self.root, f"t9-v{i}", "MlxTbccl", tpot_ms=12.0, point=f"v{i}", variant="spin")
        cells = b6.cell_rows(b6.load(self.root, "b6"))
        ab = b6.ab_rows(cells)
        row = next(r for r in ab if r["metric"] == "tpot" and r["prompt"] == "p16" and r["output_tokens_cfg"] == 16)
        self.assertEqual(row["tbccl_lifecycles"], 2)
        approx(self, row["tbccl_p50_ms"], 15.0)  # the variant lifecycles (12 ms) are not pooled into the baseline
        var = [r for r in b6.variant_rows(cells) if r["metric"] == "tpot" and r["prompt"] == "p16" and r["output_tokens_cfg"] == 16]
        self.assertEqual([r["variant"] for r in var], ["spin"])
        approx(self, var[0]["ratio_variant_over_default"], 12.0 / 15.0)
        approx(self, var[0]["ring_p50_ms"], 10.0)
        self.assertTrue(var[0]["exceeds_variation"])

    def test_a_variant_with_one_lifecycle_is_not_compared(self):
        make_run(self.root, "t9-d0", "MlxTbccl", point="d0")
        make_run(self.root, "t9-d1", "MlxTbccl", point="d1")
        make_run(self.root, "t9-v0", "MlxTbccl", point="v0", variant="async")
        var = b6.variant_rows(b6.cell_rows(b6.load(self.root, "b6")))
        self.assertTrue(var and all("not_compared" in r for r in var))

    def test_gate_rows(self):
        rows = b6.gate_rows(self.campaign())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["passed"] and r["tie_margin"] == 0.251 for r in rows))

    def test_main_writes_tables_and_the_figure_when_matplotlib_exists(self):
        self.campaign()
        out = self.root / "out"
        self.assertEqual(b6.main([str(self.root), "b6", "--out", str(out)]), 0)
        for name in ("lifecycles", "cells", "ab_latency", "resources", "resources_ab", "gates", "phases"):
            self.assertTrue((out / "tables" / f"{name}.csv").exists(), name)
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        self.assertTrue((out / "figures" / "tpot_ratio.png").exists())

    def test_tables_are_written_without_matplotlib(self):
        self.campaign()
        import builtins
        real = builtins.__import__

        def no_mpl(name, *a, **k):
            if name.startswith("matplotlib"):
                raise ImportError(name)
            return real(name, *a, **k)
        builtins.__import__ = no_mpl
        try:
            self.assertEqual(b6.main([str(self.root), "b6", "--out", str(self.root / "o2")]), 0)
        finally:
            builtins.__import__ = real
        self.assertTrue((self.root / "o2" / "tables" / "ab_latency.csv").exists())


if __name__ == "__main__":
    unittest.main()
