"""Parsing of the device-memory benchmarks' real output (saved fixtures) and the command lines the adapters build."""
import json
import unittest

from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import Outcome, ParseError, Point
from benchmarks.suite.adapters.device import GIB_TO_GB
from benchmarks.suite.tests.common import FIXTURES, SuiteTestCase


def point(bench, op, size, backends=("host", "host"), algorithm=None, source=None, **kw):
    return Point(workload="t", benchmark=bench, operation=op, algorithm=algorithm, payload_bytes=size, datatype=None, reduce_op=None, rank_count=len(backends),
                 ranks=[{"rank": i, "host": "local", "memory_backend": b} for i, b in enumerate(backends)], options=kw.pop("options", {}), warmup=2, iterations=8, repetitions=1,
                 timeout_s=30, source_rank=source)


def outcome(*paths):
    return Outcome({i: 0 for i in range(len(paths))}, set(), False, None, {i: str(p) for i, p in enumerate(paths)}, {}, 0.1)


class Ctx:
    def executable(self, name, host):
        return f"/bin/{name}"

    def rank_address(self, rank):
        return f"10.0.0.{rank['rank'] + 1}"

    def providers(self, host):
        return {"host", "cuda"}


class Parsing(SuiteTestCase):
    def test_staging_keeps_each_direction_and_the_native_semantics(self):
        a = REGISTRY["tensor_staging"]
        d2h = a.parse(point("tensor_staging", "staging_d2h", 4096, ("cuda-pinned",)), None, outcome(FIXTURES / "staging_only_cuda_pinned.csv"))
        h2d = a.parse(point("tensor_staging", "staging_h2d", 4096, ("cuda-pinned",)), None, outcome(FIXTURES / "staging_only_cuda_pinned.csv"))
        self.assertEqual(d2h.measurements["timing"]["metric"], "device_to_host_staging_time")
        self.assertEqual(h2d.measurements["timing"]["metric"], "host_to_device_staging_time")
        self.assertNotEqual(d2h.measurements["timing"]["statistics"]["median"], h2d.measurements["timing"]["statistics"]["median"])
        self.assertEqual(d2h.verification["status"], "passed")
        with self.assertRaises(ParseError):
            a.parse(point("tensor_staging", "staging_d2h", 8192, ("cuda-pinned",)), None, outcome(FIXTURES / "staging_only_cuda_pinned.csv"))

    def test_zero_staging_time_gives_no_throughput(self):
        text = (FIXTURES / "staging_only_cuda_pinned.csv").read_text().splitlines()
        head, row = text[0], text[1].split(",")
        row[head.split(",").index("median_us")] = "0"
        row[head.split(",").index("effective_GBps")] = "0"
        path = self.dir / "s.csv"
        path.write_text(head + "\n" + ",".join(row) + "\n")
        r = REGISTRY["tensor_staging"].parse(point("tensor_staging", "staging_d2h", 4096, ("metal-shared",)), None, outcome(path))
        self.assertEqual(r.measurements["throughput"], [])
        self.assertIn("about the cost of reading the clock", " ".join(r.measurements["warnings"]))

    def test_ack_calibration_keeps_the_zero_byte_row_but_does_not_call_it_an_ack_floor(self):
        path = self.dir / "a.csv"
        head = "mode,stage,source_rank,source_backend,destination_backend,payload_bytes,iterations,warmup,median_us,p95_us,p99_us,min_us,max_us,half_rtt_us,source_staging_us,source_sync_us,sender_network_us,receiver_network_us,destination_staging_us,destination_sync_us,completion_confirmed_us,effective_GBps,effective_Gbps,allocation_type,staging_capacity_bytes,allocation_count,reuse_count,timing_scope,producer_enqueue_us"
        path.write_text(head + "\n" + "ack-calibration,NA,0,host,host,0,10,3,7.92,9,9,7,9,NA,0.02,0.02,0.09,0.1,0.03,NA,7.92,0,0,host,0,1,0,ready,NA\n"
                        + "ack-calibration,NA,0,host,host,4096,10,3,15.54,16,16,15,16,NA,0.02,0.02,5.9,13.4,0.05,NA,15.54,0.26,2.1,host,4096,1,0,ready,NA\n")
        r = REGISTRY["tensor_transfer"].parse(point("tensor_transfer", "ack_calibration", 4096), None, outcome(path))
        self.assertEqual(r.measurements["timing"]["statistics"]["median"], 15.54)
        self.assertEqual(r.measurements["extra"]["zero_byte_completion_median_us"]["value"], 7.92)
        self.assertEqual(r.measurements["timing"]["metric"], "host_ack_calibration_completion_time")
        self.assertIn("NOT an ACK floor", r.measurements["extra"]["zero_byte_completion_median_us"]["note"])

    def test_diagnostics_keep_the_raw_completion_samples(self):
        csv_path = FIXTURES / "tensor_latency_floor.csv"
        head = "mode,stage,source_rank,source_backend,destination_backend,payload_bytes,iterations,warmup,median_us,p95_us,p99_us,min_us,max_us,half_rtt_us,source_staging_us,source_sync_us,sender_network_us,receiver_network_us,destination_staging_us,destination_sync_us,completion_confirmed_us,effective_GBps,effective_Gbps,allocation_type,staging_capacity_bytes,allocation_count,reuse_count,timing_scope,producer_enqueue_us"
        out = self.dir / "o.csv"
        out.write_text(head + "\n" + "end-to-end,NA,0,host,host,4096,3,1,10,12,12,9,12,NA,0,0,1,1,0,NA,10,0.4,3,host,4096,1,0,ready,NA\n")
        err = self.dir / "e.txt"
        err.write_text('noise\nTBCCL_DIAGNOSTIC {"kind":"samples","rank":0,"payload_bytes":4096,"metric":"completion_confirmed","values_us":[9,10,12]}\n')
        o = Outcome({0: 0, 1: 0}, set(), False, None, {0: str(out)}, {0: str(err)}, 0.1)
        p = point("tensor_transfer", "end_to_end", 4096, options={"diagnostics": True})
        self.assertEqual(REGISTRY["tensor_transfer"].parse(p, None, o).measurements["timing"]["samples"], [9.0, 10.0, 12.0])
        err.write_text('TBCCL_DIAGNOSTIC {"kind":"samples","rank":0,"payload_bytes":4096,"metric":"completion_confirmed","values_us":[9,10]}\n')
        with self.assertRaises(ParseError):
            REGISTRY["tensor_transfer"].parse(p, None, o)
        self.assertIn("--diagnostics", REGISTRY["tensor_transfer"].plan(p, Ctx(), [31000, 31001], self.dir).invocations[0].argv)

    def test_async_transfer_separates_enqueue_from_completion_and_converts_gib(self):
        r = REGISTRY["async_transfer"].parse(point("async_transfer", "transfer", 1048576, ("cuda-chunked", "host"), source=0), None, outcome(FIXTURES / "async_transfer_sender.json"))
        t, native = r.measurements["timing"], json.loads((FIXTURES / "async_transfer_sender.json").read_text())
        self.assertEqual(t["metric"], "completion_confirmed_time")
        self.assertEqual(len(t["samples"]), t["sample_count"])
        self.assertEqual(t["statistics"]["median"], native["completion_median_us"])
        self.assertLess(r.measurements["extra"]["enqueue_median_us"]["value"], t["statistics"]["median"] / 10)
        self.assertIn("submission only", r.measurements["extra"]["enqueue_median_us"]["note"])
        self.assertAlmostEqual(r.measurements["throughput"][0]["value"], native["gib_per_s"] * GIB_TO_GB)
        self.assertEqual(r.measurements["throughput"][0]["unit"], "GB/s")
        self.assertEqual(r.measurements["throughput"][0]["kind"], "device_to_device_effective")

    def test_async_transfer_rejects_a_failed_verification_or_missing_samples(self):
        native = json.loads((FIXTURES / "async_transfer_sender.json").read_text())
        for change in ({"verify_ok": False}, {"completion_samples_us": []}, {"role": "receiver"}):
            path = self.dir / "o.json"
            path.write_text(json.dumps({**native, **change}))
            with self.assertRaises(ParseError):
                REGISTRY["async_transfer"].parse(point("async_transfer", "transfer", 1048576, ("cuda-chunked", "host"), source=0), None, outcome(path))

    def test_cpu_overlap_is_labelled_and_an_impossible_fraction_is_flagged_not_clipped(self):
        native = json.loads((FIXTURES / "async_overlap_sender.json").read_text())
        path = self.dir / "o.json"
        path.write_text(json.dumps({**native, "hidden_fraction": 1.41388, "hidden_us": 629.9}))
        r = REGISTRY["async_overlap"].parse(point("async_overlap", "overlap", 262144, source=0), None, outcome(path))
        self.assertEqual(r.measurements["extra"]["hidden_fraction"]["value"], 1.41388)
        text = " ".join(r.measurements["warnings"])
        self.assertIn("CPU hash loop", text)
        self.assertIn("outside [0, 1]", text)
        ok = REGISTRY["async_overlap"].parse(point("async_overlap", "overlap", 262144, source=0), None, outcome(FIXTURES / "async_overlap_sender.json"))
        self.assertNotIn("outside [0, 1]", " ".join(ok.measurements["warnings"]))
        self.assertEqual(ok.verification["status"], "not_supported")

    def test_cuda_overlap_schedules(self):
        a = REGISTRY["cuda_overlap"]
        r = a.parse(point("cuda_overlap", "overlap", 1048576, ("cuda-chunked", "host"), source=0), None, outcome(FIXTURES / "cuda_overlap_overlap_sender.json"))
        self.assertEqual(r.verification["status"], "passed")
        self.assertEqual(r.measurements["timing"]["metric"], "overlap_schedule_total_time")
        c = a.parse(point("cuda_overlap", "compute_only", 1048576, ("cuda-chunked", "host"), source=0), None, outcome(FIXTURES / "cuda_overlap_compute_only_sender.json"))
        self.assertEqual(c.verification["status"], "not_supported")
        with self.assertRaises(ParseError):  # an overlap result is not a serial result
            a.parse(point("cuda_overlap", "serial", 1048576, ("cuda-chunked", "host"), source=0), None, outcome(FIXTURES / "cuda_overlap_overlap_sender.json"))

    def test_hetero_takes_the_slower_rank_and_keeps_both(self):
        r = REGISTRY["hetero_allreduce"].parse(point("hetero_allreduce", "all_reduce", 1048576, ("cuda", "host"), algorithm="async"), None,
                                               outcome(FIXTURES / "hetero_async_rank0.json", FIXTURES / "hetero_async_rank1.json"))
        t = r.measurements["timing"]
        self.assertEqual(len(t["per_rank"]), 2)
        self.assertEqual(t["statistics"]["median"], max(p["statistics"]["median"] for p in t["per_rank"]))
        with self.assertRaises(ParseError):  # the sync fields are not in an async run
            REGISTRY["hetero_allreduce"].parse(point("hetero_allreduce", "all_reduce", 1048576, ("cuda", "host"), algorithm="sync"), None,
                                               outcome(FIXTURES / "hetero_async_rank0.json", FIXTURES / "hetero_async_rank1.json"))


class Planning(SuiteTestCase):
    def test_async_transfer_command_lines_use_each_ranks_backend_and_the_source(self):
        p = point("async_transfer", "transfer", 4096, ("metal-shared", "cuda-chunked"), source=1, options={"chunk_bytes": 1024, "depth": 3, "metal_async_path": "direct"})
        g = REGISTRY["async_transfer"].plan(p, Ctx(), [31000, 31001], self.dir)
        a0, a1 = g.invocations[0].argv, g.invocations[1].argv
        self.assertEqual(a0[a0.index("--backend") + 1], "metal-shared")
        self.assertEqual(a1[a1.index("--backend") + 1], "cuda-chunked")
        self.assertEqual(a0[a0.index("--source-rank") + 1], "1")
        self.assertIn("--metal-async-path", a0)
        self.assertNotIn("--metal-async-path", a1)
        self.assertEqual((g.invocations[0].listens_on, g.invocations[1].wait_for_port), (31000, 31000))

    def test_cuda_overlap_needs_the_cuda_rank_as_source_and_a_known_receiver(self):
        a = REGISTRY["cuda_overlap"]
        self.assertIsNone(a.check_point(point("cuda_overlap", "overlap", 4096, ("cuda-chunked", "host"), source=0), Ctx()))
        bad = point("cuda_overlap", "overlap", 4096, ("host", "cuda-chunked"), source=0)
        self.assertIn("cuda-chunked", a.check_point(bad, Ctx()))
        bad2 = point("cuda_overlap", "overlap", 4096, ("cuda-chunked", "cuda-pinned"), source=0)
        self.assertIn("receiver backend", a.check_point(bad2, Ctx()))
        g = a.plan(point("cuda_overlap", "serial", 4096, ("cuda-chunked", "metal-shared"), source=0, options={"compute_rounds": 7}), Ctx(), [31000, 31001], self.dir)
        self.assertIn("--receiver-backend", g.invocations[1].argv)
        self.assertNotIn("--receiver-backend", g.invocations[0].argv)
        self.assertEqual(g.invocations[0].argv[g.invocations[0].argv.index("--schedule") + 1], "serial")

    def test_missing_compute_rounds_and_provider_are_reported(self):
        self.assertTrue(REGISTRY["cuda_overlap"].validate_workload({"options": {}}, ["overlap"]))
        self.assertTrue(REGISTRY["async_overlap"].validate_workload({}, ["overlap"]))
        p = point("tensor_staging", "staging_d2h", 4096, ("metal-shared",))
        self.assertIn("not available", REGISTRY["tensor_staging"].check_point(p, Ctx()))
        self.assertEqual(p.skip_state, "unsupported")

    def test_hetero_passes_float_count_and_algorithm(self):
        g = REGISTRY["hetero_allreduce"].plan(point("hetero_allreduce", "all_reduce", 4096, ("cuda", "metal-shared"), algorithm="sync"), Ctx(), [31000, 31001], self.dir)
        a = g.invocations[0].argv
        self.assertEqual((a[a.index("--count") + 1], a[a.index("--algo") + 1], a[a.index("--backend") + 1]), ("1024", "sync", "cuda"))
        self.assertIn("multiple of 4", REGISTRY["hetero_allreduce"].check_point(point("hetero_allreduce", "all_reduce", 10, algorithm="sync"), Ctx()))

    def test_staging_is_one_process_without_network(self):
        g = REGISTRY["tensor_staging"].plan(point("tensor_staging", "staging_h2d", 4096, ("cuda-pinned",)), Ctx(), [31000], self.dir)
        self.assertEqual(len(g.invocations), 1)
        self.assertNotIn("--peers", g.invocations[0].argv)
        self.assertIsNone(g.invocations[0].listens_on)


if __name__ == "__main__":
    unittest.main()
