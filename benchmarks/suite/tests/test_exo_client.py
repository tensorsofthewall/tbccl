"""The Exo lifecycle client against a stub Exo HTTP API (offline: no Exo, no network beyond loopback): node wait, placement read-back, readiness, streaming phases, reference/check/cancel
gates, instance recreation and deletion, and the failure paths."""
import base64
import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.suite.tests.common import FIXTURES

CLIENT = str(Path(__file__).resolve().parents[1] / "exo_bench" / "client.py")
REAL = json.loads((FIXTURES / "exo" / "state_single_ring_ready.json").read_text())
MODEL = "mlx-community/Qwen3-0.6B-8bit"


class Stub:
    """A minimal Exo: two connected nodes, one instance at a time, runners ready after two polls, deterministic greedy streams."""

    def __init__(self, layers=(7, 21), kind="MlxTbcclInstance", connected=True, corrupt=False, drop_finish=False, fail_runner=False):
        self.layers, self.kind, self.connected, self.corrupt, self.drop_finish, self.fail_runner = layers, kind, connected, corrupt, drop_finish, fail_runner
        self.instance = None
        self.polls = 0
        self.placed = 0
        self.deleted = 0
        self.requests = 0
        self.lock = threading.Lock()

    def state(self):
        with self.lock:
            st = {"instances": {}, "runners": {}, "topology": {"nodes": ["n0", "n1"], "connections": {}}}
            if self.connected:
                st["topology"]["connections"] = {"n0": {"n1": [{"x": 1}]}, "n1": {"n0": [{"x": 1}]}}
            if self.instance:
                self.polls += 1
                iid = self.instance
                shard0 = next(iter(next(iter(next(iter(REAL["instances"].values())).values()))["shardAssignments"]["runnerToShard"].values()))["PipelineShardMetadata"]
                r2s, n2r, runners, start = {}, {}, {}, 0
                for rank, n in enumerate(self.layers):
                    shard = copy.deepcopy(shard0)
                    shard.update({"deviceRank": rank, "worldSize": 2, "startLayer": start, "endLayer": start + n, "nLayers": 28})
                    r2s[f"r{rank}"] = {"PipelineShardMetadata": shard}
                    n2r[f"n{rank}"] = f"r{rank}"
                    runners[f"r{rank}"] = {"RunnerReady": {}} if self.polls > 2 else {"RunnerLoading": {}}
                    start += n
                if self.fail_runner and self.polls > 2:
                    runners["r1"] = {"RunnerFailed": {"errorMessage": "boom"}}
                body = {"instanceId": iid, "shardAssignments": {"modelId": MODEL, "runnerToShard": r2s, "nodeToRunner": n2r}, "hostsByNode": {"n0": "192.0.2.1", "n1": "192.0.2.2"}}
                st["instances"] = {iid: {self.kind: body}}
                st["runners"] = runners
            return st

    def tokens(self, content, n):
        seed = sum(map(ord, content)) % 7
        return [f"w{(seed + i) % 5}" for i in range(n)]


def handler_for(stub):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send_json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/state":
                self.send_json(stub.state())
            elif self.path == "/node_id":
                self.send_json("n0")
            else:
                self.send_json({}, 404)

        def do_DELETE(self):
            with stub.lock:
                stub.instance = None
                stub.polls = 0
                stub.deleted += 1
            self.send_json({"message": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/place_instance":
                if not stub.connected:
                    return self.send_json({"error": {"message": "Needs 2 connected nodes"}}, 400)
                with stub.lock:
                    stub.placed += 1
                    stub.instance = f"inst-{stub.placed}"
                    stub.polls = 0
                return self.send_json({"message": "Command received."})
            if self.path != "/v1/chat/completions":
                return self.send_json({}, 404)
            stub.requests += 1
            n = int(body["max_tokens"])
            toks = stub.tokens(body["messages"][0]["content"], n)
            if stub.corrupt:
                toks[1] = "BAD"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(text):
                data = text.encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            try:
                chunk(": prefill_progress {}\n\n")
                for i, t in enumerate(toks):
                    ch = {"index": 0, "delta": {"role": "assistant", "content": t}}
                    if body.get("logprobs"):
                        ch["logprobs"] = {"content": [{"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1}, {"token": "z", "logprob": -5.0}]}]}
                    ev = {"choices": [ch]}
                    if i == n - 1 and not stub.drop_finish:
                        ch["finish_reason"] = "length"
                        ev["usage"] = {"prompt_tokens": 20, "completion_tokens": n}
                    chunk("data: " + json.dumps(ev) + "\n\n")
                chunk("data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # a client that drops a stream on purpose resets the connection
        pass


class Server:
    def __init__(self, stub):
        self.stub = stub
        self.httpd = QuietServer(("127.0.0.1", 0), handler_for(stub))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


PROMPTS = {"short": {"text": "Count from one to five in words.", "approx_tokens": 20}, "p128": {"items": 3, "approx_tokens": 80}}
CASES = [{"prompt": "short", "max_tokens": 6}, {"prompt": "p128", "max_tokens": 8}]


def run_client(port, phases, **kw):
    spec = {"api": {"host": "127.0.0.1", "port": port}, "model": MODEL, "backend": "MlxTbccl", "world_size": 2, "node_count": 2, "prompts": PROMPTS, "phases": phases,
            "peers": [{"name": "host-a", "host": "127.0.0.1", "port": port}, {"name": "host-b", "host": "127.0.0.1", "port": port}], "link_addresses": ["192.0.2.1", "192.0.2.2"],
            "discovery_timeout_s": kw.pop("discovery_timeout_s", 20), "load_timeout_s": 20, "request_timeout_s": 10, "tag": "t", "poll_s": 0.05, "stable_s": 0.1, **kw}
    out = subprocess.run([sys.executable, "-I", CLIENT, "--spec-b64", base64.b64encode(json.dumps(spec).encode()).decode()], capture_output=True, text=True, timeout=120)
    lines = {}
    for line in out.stdout.splitlines():
        prefix, _, payload = line.partition(" ")
        lines.setdefault(prefix, []).append(json.loads(payload))
    return out.returncode, lines, out.stderr


def events(lines, name):
    return [b for b in lines.get("TBCCL_EXO_BENCH", []) if b.get("event") == name]


class Lifecycle(unittest.TestCase):
    def serve(self, **kw):
        s = Stub(**kw)
        srv = Server(s)
        self.addCleanup(srv.close)
        return s, srv

    def test_full_lifecycle_places_loads_measures_and_deletes(self):
        s, srv = self.serve()
        phases = [{"name": "warm", "kind": "warmup", "prompt": "short", "max_tokens": 4, "requests": 1}, {"name": "m", "kind": "rounds", "prompt": "short", "max_tokens": 6, "conc": 2, "requests": 4}]
        code, lines, err = run_client(srv.port, phases, expected_layers=[7, 21])
        self.assertEqual(code, 0, err)
        place = events(lines, "placement")[0]
        self.assertEqual((place["problems"], [x["n_layers"] for x in place["layout"]["stages"]]), ([], [7, 21]))
        self.assertGreater(events(lines, "ready")[0]["load_s"], 0)
        m = next(e for e in events(lines, "phase") if e["phase"] == "m")
        self.assertEqual(m["throughput"]["successful"], 4)
        self.assertEqual(m["accounting_problems"], [])
        self.assertEqual(events(lines, "done")[0]["status"], "ok")
        self.assertEqual((s.placed, s.deleted, s.instance), (1, 1, None))
        raws = lines["TBCCL_EXO_RAW"]
        self.assertEqual(len(raws), 5)
        self.assertTrue(all(r["request"]["status"] == "ok" for r in raws))
        self.assertTrue(any(st["step"] == "ready" for st in lines["TBCCL_EXO_STATE"]))

    def test_unconnected_nodes_time_out_without_placing(self):
        s, srv = self.serve(connected=False)
        code, lines, err = run_client(srv.port, [{"name": "w", "kind": "warmup", "prompt": "short", "max_tokens": 2, "requests": 1}], discovery_timeout_s=3)
        self.assertNotEqual(code, 0)
        self.assertIn("connected in both directions", err)
        self.assertEqual(s.placed, 0)

    def test_wrong_layer_assignment_is_refused_before_any_request_and_the_instance_is_deleted(self):
        s, srv = self.serve(layers=(14, 14))
        code, lines, err = run_client(srv.port, [{"name": "w", "kind": "warmup", "prompt": "short", "max_tokens": 2, "requests": 1}], expected_layers=[7, 21])
        self.assertNotEqual(code, 0)
        self.assertTrue(events(lines, "placement")[0]["problems"])
        self.assertEqual((s.requests, s.deleted), (0, 1))
        self.assertIn("does not match the experiment", events(lines, "error")[0]["error"])

    def test_wrong_instance_type_is_refused(self):
        s, srv = self.serve(kind="MlxRingInstance")
        code, lines, err = run_client(srv.port, [{"name": "w", "kind": "warmup", "prompt": "short", "max_tokens": 2, "requests": 1}])
        self.assertNotEqual(code, 0)
        self.assertTrue(any("MlxTbcclInstance" in p for p in events(lines, "placement")[0]["problems"]))
        self.assertEqual(s.requests, 0)

    def test_a_failed_runner_stops_the_run_and_still_deletes(self):
        s, srv = self.serve(fail_runner=True)
        code, lines, err = run_client(srv.port, [{"name": "w", "kind": "warmup", "prompt": "short", "max_tokens": 2, "requests": 1}])
        self.assertNotEqual(code, 0)
        self.assertEqual((s.requests, s.deleted), (0, 1))

    def test_a_stream_without_finish_reason_is_a_partial_request(self):
        s, srv = self.serve(drop_finish=True)
        code, lines, err = run_client(srv.port, [{"name": "m", "kind": "rounds", "prompt": "short", "max_tokens": 4, "conc": 1, "requests": 1}])
        rec = lines["TBCCL_EXO_RAW"][0]["request"]
        self.assertEqual(rec["status"], "partial")
        self.assertIn("finish_reason", rec["error"])
        phase = events(lines, "phase")[0]
        self.assertEqual(phase["throughput"]["successful"], 0)


class Gates(unittest.TestCase):
    def serve(self, **kw):
        s = Stub(**kw)
        srv = Server(s)
        self.addCleanup(srv.close)
        return s, srv

    def reference_for(self, srv):
        code, lines, err = run_client(srv.port, [{"name": "ref", "kind": "reference", "cases": CASES}])
        self.assertEqual(code, 0, err)
        path = Path(tempfile.mkdtemp()) / "ref.json"
        path.write_text(json.dumps(lines["TBCCL_EXO_REFERENCE"][0]["reference"]))
        self.addCleanup(path.unlink)
        return str(path)

    def test_reference_then_check_passes_and_a_corrupted_stream_fails(self):
        _, srv = self.serve()
        ref = self.reference_for(srv)
        gate = [{"name": "gate", "kind": "check", "cases": CASES, "conc": 2, "repeat": 2}]
        code, lines, err = run_client(srv.port, gate, reference=ref)
        self.assertEqual(code, 0, err)
        chk = events(lines, "phase")[0]["check"]
        self.assertEqual((chk["passed"], chk["outcome"]), (True, {"exact": 4}))
        _, bad = self.serve(corrupt=True)
        code, lines, err = run_client(bad.port, gate, reference=ref)
        chk = events(lines, "phase")[0]["check"]
        self.assertFalse(chk["passed"])
        self.assertEqual(chk["outcome"], {"mismatch": 4})
        self.assertEqual(chk["details"][0]["first_difference"], 1)

    def test_a_reference_made_in_the_same_lifecycle_serves_the_following_checks(self):
        _, srv = self.serve()
        code, lines, err = run_client(srv.port, [{"name": "ref", "kind": "reference", "cases": CASES}, {"name": "self", "kind": "check", "cases": CASES, "repeat": 2}])
        self.assertEqual(code, 0, err)
        self.assertEqual(events(lines, "phase")[1]["check"]["outcome"], {"exact": 4})

    def test_check_without_a_reference_is_an_error(self):
        _, srv = self.serve()
        code, lines, err = run_client(srv.port, [{"name": "gate", "kind": "check", "cases": CASES}])
        self.assertNotEqual(code, 0)
        self.assertIn("needs spec.reference", events(lines, "error")[0]["error"])

    def test_cancel_phase_drops_streams_and_proves_the_follow_up(self):
        _, srv = self.serve()
        ref = self.reference_for(srv)
        phase = {"name": "cancel", "kind": "cancel", "prompt": "short", "max_tokens": 60, "streams": 4, "complete_every": 4, "drop_after_events": 2, "settle_s": 0.1,
                 "followup": {"kind": "check", "cases": CASES, "conc": 1}}
        code, lines, err = run_client(srv.port, [phase], reference=ref)
        self.assertEqual(code, 0, err)
        c = events(lines, "phase")[0]["cancel"]
        self.assertEqual((c["passed"], c["cancelled"], c["completed"]), (True, 3, 1))

    def test_recreate_replaces_the_instance_and_checks_the_new_placement(self):
        s, srv = self.serve()
        ref = self.reference_for(srv)
        phases = [{"name": "before", "kind": "check", "cases": CASES}, {"name": "again", "kind": "recreate"}, {"name": "after", "kind": "check", "cases": CASES}]
        code, lines, err = run_client(srv.port, phases, reference=ref)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(events(lines, "placement")), 2)
        self.assertTrue(all(e["check"]["passed"] for e in events(lines, "phase") if "check" in e))
        self.assertEqual(s.deleted, 2 + 1)  # the reference run, the recreation and the final delete: three instances over this stub's life


if __name__ == "__main__":
    unittest.main()
