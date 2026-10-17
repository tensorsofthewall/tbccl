"""The streaming client against a fake vLLM-style server (offline, loopback only): readiness, rounds, references, correctness classification, row-order corruption,
cancellation, timeouts, partial streams and HTTP errors."""
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.suite.tests.common import approx
from benchmarks.suite.vllm_bench import client

WORKLOAD = {"prompts": {"p16": list(range(1, 17)), "p128": list(range(100, 228))}}


def gen_tokens(prompt, n):
    base = sum(prompt)
    return [(base * 31 + i * 7) % 5000 for i in range(n)]


class FakeServer:
    def __init__(self, **behaviour):
        self.b = {"tie_positions": (), "margin": 0.5, "delay_s": 0.002, "mix_rows": False, "stall_after": None, "close_after": None, "http_error": None, "flip": None, "no_usage": False,
                  "batch_tokens": 1, "finish": "length", "truncate_after": None}
        self.b.update(behaviour)
        self.running = 0
        self.aborted = 0
        self.requests = []
        self.lock = threading.Lock()
        self.serial = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json(200, {"data": [{"id": "m"}]})
                if self.path == "/health":
                    return self._json(200, {})
                if self.path == "/version":
                    return self._json(200, {"version": "0.31.0"})
                if self.path == "/metrics":
                    text = (f'vllm:num_requests_running{{engine="0"}} {float(outer.running)}\nvllm:num_requests_waiting{{engine="0"}} 0.0\n'
                            'vllm:request_queue_time_seconds_sum{engine="0"} 0.5\nvllm:request_queue_time_seconds_count{engine="0"} 2.0\n'
                            'vllm:request_queue_time_seconds_bucket{le="1.0"} 2.0\nvllm:prompt_tokens_created{engine="0"} 1.7e9\n').encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(text)))
                    self.end_headers()
                    self.wfile.write(text)
                    return
                self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer.lock:
                    outer.serial += 1
                    serial = outer.serial
                    outer.requests.append(body)
                if outer.b["http_error"]:
                    return self._json(outer.b["http_error"], {"error": "boom"})
                prompt, n = body["prompt"], body["max_tokens"]
                toks = gen_tokens(prompt, n)
                if outer.b["mix_rows"] and outer.running > 0:  # a concurrent request receives another request's rows
                    toks = gen_tokens([x + 1 for x in prompt], n)
                if outer.b["flip"]:
                    pos, kind = outer.b["flip"]
                    if pos < n:
                        toks[pos] = (toks[pos] + 1) % 5000
                if not body.get("stream"):
                    lp = {"tokens": [str(t) for t in toks], "top_logprobs": [{"a": -0.1, "b": -0.1 - (0.0 if i in outer.b["tie_positions"] else outer.b["margin"])} for i in range(n)]}
                    return self._json(200, {"choices": [{"text": "x", "token_ids": toks, "logprobs": lp}], "usage": {"prompt_tokens": len(prompt), "completion_tokens": n}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(data):
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()

                with outer.lock:
                    outer.running += 1
                try:
                    step = outer.b["batch_tokens"]
                    sent = 0
                    while sent < n:
                        if outer.b["truncate_after"] is not None and sent >= outer.b["truncate_after"]:  # the engine died: the API closes the stream cleanly without a finish reason
                            chunk(b"data: [DONE]\n\n")
                            chunk(b"")
                            return
                        if outer.b["stall_after"] is not None and sent >= outer.b["stall_after"]:
                            time.sleep(5)
                        if outer.b["close_after"] is not None and sent >= outer.b["close_after"]:
                            self.close_connection = True
                            self.wfile.flush()
                            self.connection.shutdown(socket.SHUT_RDWR)
                            return
                        time.sleep(outer.b["delay_s"])
                        take = toks[sent:sent + step]
                        sent += len(take)
                        ev = {"choices": [{"index": 0, "text": "x", "token_ids": take, "finish_reason": (outer.b["finish"] if sent == n else None)}]}
                        if not outer.b["no_usage"]:
                            ev["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": sent}
                        chunk(("data: " + json.dumps(ev) + "\n\n").encode())
                    if not outer.b["no_usage"]:
                        chunk(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": n}}) + "\n\n").encode())
                    chunk(b"data: [DONE]\n\n")
                    chunk(b"")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    with outer.lock:
                        outer.aborted += 1
                finally:
                    with outer.lock:
                        outer.running -= 1

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_client(tmp_path, workload, port, phases, reference=None, **extra):
    spec = {"api": {"host": "127.0.0.1", "port": port}, "model": "m", "workload": workload, "reference": reference, "request_timeout_s": extra.pop("request_timeout_s", 10), "phases": phases, **extra}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    out = io.StringIO()
    import sys
    argv = sys.argv
    sys.argv = ["client.py", "--spec", str(path)]
    try:
        with redirect_stdout(out):
            code = client.main()
    finally:
        sys.argv = argv
    lines = out.getvalue().splitlines()

    def parse(prefix):
        return [json.loads(line[len(prefix) + 1:]) for line in lines if line.startswith(prefix + " ")]
    return code, parse("TBCCL_VLLM_BENCH"), parse("TBCCL_VLLM_RAW"), parse("TBCCL_VLLM_REFERENCE")


def make_reference(tmp_path, workload, port, tie_positions=()):
    code, bench, _, refs = run_client(tmp_path, workload, port, [{"name": "ready", "kind": "ready", "timeout_s": 5},
                                                                 {"name": "ref", "kind": "reference", "cases": [{"name": "p16", "prompt": "p16", "tokens": 32}, {"name": "p128", "prompt": "p128", "tokens": 32}]}])
    assert code == 0
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(refs[0]["reference"]))
    return str(path)


class ClientAgainstAFakeServer(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)
        self.workload = str(self.tmp_path / "workload.json")
        Path(self.workload).write_text(json.dumps(WORKLOAD))

    def test_ready_rounds_and_raw_evidence(self):
        srv = FakeServer()
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [
                {"name": "ready", "kind": "ready", "timeout_s": 5},
                {"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 8, "conc": 3, "rounds": 2}])
        finally:
            srv.close()
        assert code == 0
        ready = bench[0]
        assert ready["ready"] and ready["/metrics"] == 200 and ready["version"]["version"] == "0.31.0"
        r = next(b for b in bench if b["phase"] == "r")
        assert r["loop"] == "closed-rounds" and r["concurrency"] == 3 and r["requests"] == 6 and r["throughput"]["successful"] == 6
        assert r["throughput"]["output_tokens"] == 48  # actual generated tokens, not a configured maximum applied blindly
        assert r["distributions"]["ttft_s"]["n"] == 6 and r["distributions"]["itl_s"]["n"] == 6 * 7
        assert len(r["throughput"]["per_round"]) == 2
        assert len([x for x in raw if x["phase"] == "r"]) == 6
        assert any(b["kind"] == "scrape" and b["samples"] >= 1 for b in bench)


    def test_reference_then_exact_check_passes_and_records_margins(self):
        srv = FakeServer(tie_positions=(5,))
        try:
            ref = make_reference(self.tmp_path, self.workload, srv.port)
            cases = json.load(open(ref))["cases"]
            assert cases[0]["margins"][5] == approx(0.0) and cases[0]["margins"][4] == approx(0.5, abs=1e-6)
            code, bench, _, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "c", "kind": "check", "cases": ["p16", "p128"], "conc": 4, "repeat": 2, "tokens": 32}], reference=ref)
        finally:
            srv.close()
        c = next(b for b in bench if b["phase"] == "c")["check"]
        assert code == 0 and c["passed"] and c["outcome"]["exact"] == 4 and c["outcome"]["diverged"] == 0


    def test_divergence_at_a_near_tie_is_classified_but_a_meaningful_margin_fails(self):
        srv = FakeServer(tie_positions=(9,))
        try:
            ref = make_reference(self.tmp_path, self.workload, srv.port)
            srv.b["flip"] = (9, "tie")
            code, bench, _, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "c", "kind": "check", "cases": ["p16"], "conc": 1, "tokens": 32}], reference=ref)
            tie = next(b for b in bench if b["phase"] == "c")["check"]
            srv.b["flip"] = (4, "real")  # margin 0.5 at position 4: a real divergence
            code2, bench2, _, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "c", "kind": "check", "cases": ["p16"], "conc": 1, "tokens": 32}], reference=ref)
            real = next(b for b in bench2 if b["phase"] == "c")["check"]
        finally:
            srv.close()
        assert code == 0 and tie["outcome"]["near_tie"] == 1 and tie["passed"] and tie["details"][0]["reference_margin"] == approx(0.0)
        assert code2 == 2 and real["outcome"]["diverged"] == 1 and not real["passed"] and real["details"][0]["position"] == 4


    def test_row_mixing_between_concurrent_requests_is_detected(self):
        srv = FakeServer()
        try:
            ref = make_reference(self.tmp_path, self.workload, srv.port)
            srv.b["mix_rows"] = True
            code, bench, _, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "c", "kind": "check", "cases": ["p16", "p128"], "conc": 8, "repeat": 4, "tokens": 32, "stagger_s": 0.01}], reference=ref)
        finally:
            srv.close()
        c = next(b for b in bench if b["phase"] == "c")["check"]
        assert code == 2 and c["outcome"]["diverged"] >= 1 and not c["passed"]


    def test_short_output_is_a_failure(self):
        srv = FakeServer()
        try:
            ref = make_reference(self.tmp_path, self.workload, srv.port)
            srv.b["close_after"] = 10  # the stream ends early, without [DONE]
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "c", "kind": "check", "cases": ["p16"], "conc": 1, "tokens": 32}], reference=ref)
        finally:
            srv.close()
        c = next(b for b in bench if b["phase"] == "c")["check"]
        assert code == 2 and c["outcome"]["request_failed"] == 1
        assert next(r for r in raw if r["phase"] == "c")["request"]["status"] == "partial"


    def test_a_stream_truncated_by_a_dead_engine_is_a_failure_even_though_it_ends_with_done(self):
        srv = FakeServer(truncate_after=3)
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 8, "conc": 1, "rounds": 1}])
            code2, bench2, _, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 8, "conc": 1, "rounds": 1, "allow_failures": True}])
        finally:
            srv.close()
        rec = next(x for x in raw if x["phase"] == "r")["request"]
        assert rec["status"] == "partial" and "no finish_reason" in rec["error"]
        assert code == 8  # a measured phase with a failed request fails the point
        assert code2 == 0 and next(b for b in bench2 if b["phase"] == "r")["throughput"]["successful"] == 0  # the lifecycle tier expects and records it


    def test_an_unexpected_finish_reason_is_an_error(self):
        srv = FakeServer(finish="abort")
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 4, "conc": 1, "rounds": 1}])
        finally:
            srv.close()
        assert next(x for x in raw if x["phase"] == "r")["request"]["status"] == "error" and code == 8


    def test_timeout_is_recorded_and_stops_the_client(self):
        srv = FakeServer(stall_after=2)
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 8, "conc": 1, "rounds": 2, "request_timeout_s": 0.4}],
                                             request_timeout_s=0.4)
        finally:
            srv.close()
        r = next(b for b in bench if b["phase"] == "r")
        assert code == 5 and r["throughput"]["timeouts"] == 1 and r["throughput"]["successful"] == 0
        assert r["throughput"]["error_rate"] == 1.0
        assert r["distributions"]["ttft_s"]["n"] == 0


    def test_http_error_is_a_failed_request_not_a_crash(self):
        srv = FakeServer(http_error=503)
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "continuous", "prompt": "p16", "tokens": 4, "conc": 2, "requests": 3}])
        finally:
            srv.close()
        r = next(b for b in bench if b["phase"] == "r")
        assert r["throughput"]["status_counts"] == {"error": 3} and r["throughput"]["successful"] == 0
        assert "HTTP 503" in next(x for x in raw if x["phase"] == "r")["request"]["error"]


    def test_not_ready_server_fails_bounded(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        t0 = time.monotonic()
        code, bench, _, _ = run_client(self.tmp_path, self.workload, port, [{"name": "ready", "kind": "ready", "timeout_s": 1.5}])
        assert code == 3 and bench[0]["ready"] is False and time.monotonic() - t0 < 6


    def test_cancellation_drains_the_engine_and_followups_match_the_reference(self):
        srv = FakeServer(delay_s=0.01)
        try:
            ref = make_reference(self.tmp_path, self.workload, srv.port)
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [
                {"name": "x", "kind": "cancel", "prompt": "p16", "tokens": 200, "streams": 6, "complete_every": 3, "drop_after_events": 3, "drain_timeout_s": 5,
                 "followup": {"kind": "check", "cases": ["p16", "p128"], "conc": 1, "tokens": 32}}], reference=ref)
            aborted = srv.aborted
        finally:
            srv.close()
        x = next(b for b in bench if b["phase"] == "x")["cancel"]
        assert code == 0 and x["cancelled"] == 4 and x["completed"] == 2 and x["drained"] and x["followup"]["passed"] and x["passed"]
        assert aborted >= 1


    def test_open_loop_records_offered_and_achieved_rates(self):
        srv = FakeServer()
        try:
            code, bench, _, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "o", "kind": "open_loop", "prompt": "p16", "tokens": 4, "rate": 50, "requests": 12}])
        finally:
            srv.close()
        o = next(b for b in bench if b["phase"] == "o")
        assert o["loop"] == "open-poisson" and o["offered_rate_per_s"] == 50 and o["throughput"]["successful"] == 12
        assert o["throughput"]["rates"]["requests_per_s"] > 0


    def test_batched_events_are_kept_explicit(self):
        srv = FakeServer(batch_tokens=4)
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 8, "conc": 1, "rounds": 1}])
        finally:
            srv.close()
        r = next(b for b in bench if b["phase"] == "r")
        assert r["distributions"]["batched_intervals"] == 1 and r["distributions"]["itl_s"]["n"] == 0  # no fictional per-token timestamps
        assert r["throughput"]["output_tokens"] == 8


    def test_missing_usage_is_flagged_but_counted_from_token_ids(self):
        srv = FakeServer(no_usage=True)
        try:
            code, bench, raw, _ = run_client(self.tmp_path, self.workload, srv.port, [{"name": "r", "kind": "rounds", "prompt": "p16", "tokens": 5, "conc": 1, "rounds": 1}])
        finally:
            srv.close()
        assert code == 0
        assert next(b for b in bench if b["phase"] == "r")["throughput"]["output_tokens"] == 5


if __name__ == "__main__":
    unittest.main()
