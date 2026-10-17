"""Lifecycle and streaming client for a running Exo cluster (standard library only; it runs on the API host and never imports Exo, MLX or exo-tbccl).

    python client.py (--spec SPEC.json | --spec-b64 BASE64) [--out-dir DIR]

The suite starts the Exo nodes; this client drives the one thing that is Exo's own: the application lifecycle through its HTTP API. In order it waits for the expected nodes,
places one instance with Exo's normal placement (``POST /place_instance``, no manual layer boundaries), reads back what was actually placed and checks it against the spec,
waits for every runner to be ready (the model load, timed on its own), runs the ordered phases and deletes the instance (timed). Every step prints a ``TBCCL_EXO_STATE`` line with the
raw ``/state`` evidence; every request prints a ``TBCCL_EXO_RAW`` timeline; each phase prints one ``TBCCL_EXO_BENCH`` summary. Durations use ``time.monotonic_ns``; ``time.time`` stamps
are recorded only to align logs.

Phase kinds
  warmup       untimed sequential requests (kept, flagged)
  rounds       closed loop in rounds of ``conc`` simultaneous requests
  continuous   closed loop with ``conc`` workers that issue their next request when the previous ends, for ``requests`` requests in total
  reference    greedy generations stored with the top-1/top-2 logprob margin of every position (run on a single-device instance)
  check        requests compared with a stored reference (exact, or divergence at a near-tie)
  cancel       streams dropped after a few tokens, then a follow-up request compared with the reference
"""
import argparse
import base64
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
import validation  # noqa: E402

PREFIX = "TBCCL_EXO_BENCH"
LOCK = threading.Lock()


def line(prefix, obj):
    with LOCK:
        print(f"{prefix} {json.dumps(obj)}", flush=True)


def emit(obj):
    line(PREFIX, obj)


def now_ns():
    return time.monotonic_ns()


class Api:
    def __init__(self, spec):
        self.host, self.port, self.model = spec["api"]["host"], spec["api"]["port"], spec["model"]
        self.timeout = float(spec.get("request_timeout_s", 300))

    def conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)

    def call(self, method, path, body=None, timeout=30):
        c = self.conn(timeout)
        try:
            c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            if r.status >= 300:
                raise RuntimeError(f"{method} {path}: HTTP {r.status}: {data[:200]!r}")
            return json.loads(data) if data else None
        finally:
            c.close()

    def state(self):
        return self.call("GET", "/state")


def prompt_text(n_items):
    return "Summarize the following in two sentences. " + " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog number {i * 3}." for i in range(1, n_items + 1))


def messages_for(prompt):
    text = prompt["text"] if "text" in prompt else prompt_text(int(prompt["items"]))
    return [{"role": "user", "content": text}]


def stream_request(api, meta, messages, max_tokens, logprobs=False, top_logprobs=0, drop_after_events=None, timeout=None):
    """One streamed chat completion at temperature 0. Returns the timeline record; never raises (every failure becomes a status and an error string)."""
    tl = metrics.ExoTimeline({**meta, "max_tokens": max_tokens, "wall_submit": time.time()}, now_ns())
    body = {"model": api.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": True, "enable_thinking": False}
    if logprobs:
        body.update({"logprobs": True, "top_logprobs": top_logprobs})
    c = api.conn(timeout)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        tl.http_status = r.status
        tl.t_response_ns = now_ns()
        if r.status != 200:
            tl.finish("error", now_ns(), f"HTTP {r.status}: {r.read()[:200].decode('utf-8', 'replace')}")
            return tl.to_record()
        while True:
            raw = r.readline()
            t = now_ns()
            if not raw:
                break
            kind, obj = metrics.parse_sse_line(raw)
            if kind == "ignore":
                continue
            tl.on_event(kind, obj, t)
            if drop_after_events is not None and len(tl.events) >= drop_after_events:
                tl.finish("cancelled", now_ns())
                return tl.to_record()
            if kind == "done":
                break
        if tl.stream_error:
            tl.finish("error", now_ns(), f"error event: {json.dumps(tl.stream_error)[:200]}")
        elif tl.t_done_ns is None:
            tl.finish("partial", now_ns(), "stream ended without [DONE]")
        elif not tl.events:
            tl.finish("error", now_ns(), "stream completed without any token event")
        elif tl.finish_reason is None:
            tl.finish("partial", now_ns(), "stream ended with [DONE] but no finish_reason")
        elif tl.finish_reason not in ("length", "stop"):
            tl.finish("error", now_ns(), f"finish_reason {tl.finish_reason!r}")
        else:
            tl.finish("ok", now_ns())
    except (TimeoutError, OSError) as e:
        timed_out = isinstance(e, TimeoutError) or "timed out" in str(e)
        tl.finish("timeout" if timed_out else "error", now_ns(), f"{type(e).__name__}: {e}")
    except metrics.StreamError as e:
        tl.finish("error", now_ns(), str(e))
    finally:
        try:
            c.close()
        except OSError:
            pass
    return tl.to_record()


# ---- lifecycle ----
def wait_nodes(api, count, timeout, poll_s=1.0, stable_s=3.0):
    t0 = now_ns()
    last = None
    while (now_ns() - t0) / 1e9 < timeout:
        try:
            st = api.state()
        except (OSError, RuntimeError):
            time.sleep(poll_s)
            continue
        n = len(validation.nodes_of(st))
        last = f"{n} nodes, connected={validation.connected(st, count)}"
        if n == count and validation.connected(st, count):
            time.sleep(stable_s)  # membership must be stable, not just momentarily right
            st2 = api.state()
            if validation.connected(st2, count):
                line("TBCCL_EXO_STATE", {"step": "nodes", "state": st2})
                return (now_ns() - t0) / 1e9
        time.sleep(poll_s)
    raise TimeoutError(f"cluster not ready after {timeout}s (last seen: {last}); expected {count} nodes connected in both directions")


def place(api, spec):
    before = set((api.state().get("instances") or {}))
    t0 = now_ns()
    api.call("POST", "/place_instance", {"model_id": spec["model"], "sharding": "Pipeline", "instance_meta": spec["backend"], "min_nodes": spec["world_size"]})
    for _ in range(300):
        st = api.state()
        new = set(st.get("instances") or {}) - before
        if new:
            return new.pop(), st, (now_ns() - t0) / 1e9
        time.sleep(0.3)
    raise RuntimeError("no instance appeared")


def wait_ready(api, iid, timeout, poll_s=1.0):
    t0 = now_ns()
    seen = []
    while (now_ns() - t0) / 1e9 < timeout:
        st = api.state()
        if iid not in (st.get("instances") or {}):
            raise RuntimeError("instance vanished while loading")
        tags = validation.runner_states(st, iid)
        if not seen or seen[-1][1] != tags:
            seen.append([(now_ns() - t0) / 1e9, dict(tags)])
        problems = validation.check_ready(st, iid)
        if not problems:
            return (now_ns() - t0) / 1e9, st, seen
        if any("failed" in p for p in problems):
            raise RuntimeError("; ".join(problems))
        time.sleep(poll_s)
    raise TimeoutError(f"instance not ready after {timeout}s")


def delete(api, iid, timeout=120):
    t0 = now_ns()
    api.call("DELETE", f"/instance/{iid}")
    while (now_ns() - t0) / 1e9 < timeout:
        if iid not in (api.state().get("instances") or {}):
            return (now_ns() - t0) / 1e9
        time.sleep(0.3)
    raise RuntimeError("instance not deleted")


def peer_views(spec, iid):
    """Every node's own /state (the API node first) and node id; the cluster is only trusted when all nodes agree."""
    views, ids = {}, {}
    for peer in spec.get("peers") or []:
        a = Api({"api": peer, "model": spec["model"], "request_timeout_s": 10})
        ids[peer.get("name", peer["host"])] = a.call("GET", "/node_id")
        views[peer.get("name", peer["host"])] = a.state()
    return ids, views


# ---- phases ----
def summarize_phase(phase, records, base):
    ok = [r for r in records if r.get("status") == "ok"]
    out = {"phase": phase["name"], "kind": phase["kind"], **base, "distributions": metrics.request_distributions(records), "throughput": metrics.phase_throughput(records),
           "accounting_problems": [p for r in ok for p in metrics.accounting(r)["problems"]][:20]}
    return out


def run_requests(api, phase, spec, tag):
    prompts = spec["prompts"]
    conc, total = int(phase.get("conc", 1)), int(phase.get("requests", 1))
    msgs = {k: messages_for(v) for k, v in prompts.items()}
    plan = [phase["prompt"] if isinstance(phase.get("prompt"), str) else phase["prompt"][i % len(phase["prompt"])] for i in range(total)]
    records = [None] * total
    counter = [0]
    t_start = now_ns()

    def worker():
        while True:
            with LOCK:
                i = counter[0]
                counter[0] += 1
            if i >= total:
                return
            records[i] = stream_request(api, {"phase": phase["name"], "index": i, "prompt": plan[i], "conc": conc, "tag": tag, "warmup": phase["kind"] == "warmup"}, msgs[plan[i]],
                                        int(phase["max_tokens"]), logprobs=bool(phase.get("logprobs")), top_logprobs=int(phase.get("top_logprobs", 0)))

    if phase["kind"] == "rounds":
        for start in range(0, total, conc):
            idx = list(range(start, min(start + conc, total)))
            ths = []
            for i in idx:
                def one(i=i):
                    records[i] = stream_request(api, {"phase": phase["name"], "index": i, "prompt": plan[i], "conc": conc, "tag": tag, "warmup": False}, msgs[plan[i]],
                                                int(phase["max_tokens"]), logprobs=bool(phase.get("logprobs")), top_logprobs=int(phase.get("top_logprobs", 0)))
                ths.append(threading.Thread(target=one))
            for t in ths:
                t.start()
            for t in ths:
                t.join()
    else:
        ths = [threading.Thread(target=worker) for _ in range(1 if phase["kind"] == "warmup" else conc)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    wall = (now_ns() - t_start) / 1e9
    return records, wall


# ---- correctness gates ----
def case_key(case):
    return f"{case['prompt']}:{case['max_tokens']}"


def run_cases(api, phase, spec, with_margins):
    """Requests for the cases of a reference/check/cancel phase, ``repeat`` times, ``conc`` at a time; returns [(case, record)]."""
    msgs = {k: messages_for(v) for k, v in spec["prompts"].items()}
    jobs = [(c, r) for r in range(int(phase.get("repeat", 1))) for c in phase["cases"]]
    out = [None] * len(jobs)
    counter = [0]

    def worker():
        while True:
            with LOCK:
                i = counter[0]
                counter[0] += 1
            if i >= len(jobs):
                return
            case, rep = jobs[i]
            out[i] = (case, stream_request(api, {"phase": phase["name"], "index": i, "prompt": case["prompt"], "repeat": rep, "conc": phase.get("conc", 1), "tag": spec.get("tag"), "warmup": False},
                                           msgs[case["prompt"]], int(case["max_tokens"]), logprobs=True, top_logprobs=2))

    ths = [threading.Thread(target=worker) for _ in range(max(1, int(phase.get("conc", 1))))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out


def run_reference(api, phase, spec):
    pairs = run_cases(api, phase, spec, True)
    cases = {}
    for case, rec in pairs:
        if rec["status"] != "ok":
            raise RuntimeError(f"reference request failed: {rec.get('error')}")
        cases[case_key(case)] = {"tokens": rec["token_texts"], "margins": rec["margins"], "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"), "finish_reason": rec["finish_reason"]}
    doc = {"model": spec["model"], "backend": spec["backend"], "cases": cases, "greedy": True}
    line("TBCCL_EXO_REFERENCE", {"name": phase["name"], "reference": doc})
    return [r for _, r in pairs], {"reference_cases": len(cases)}, doc


def run_check(api, phase, spec, ref):
    pairs = run_cases(api, phase, spec, True)
    tie = float(spec.get("tie_margin", 0.251))
    outcomes = []
    for case, rec in pairs:
        if rec["status"] != "ok":
            outcomes.append({"outcome": "failed", "error": rec.get("error"), "case": case_key(case)})
            continue
        want = ref["cases"].get(case_key(case))
        if want is None:
            outcomes.append({"outcome": "failed", "error": "no reference for this case", "case": case_key(case)})
            continue
        res = metrics.compare_tokens(rec["token_texts"], want["tokens"], want["margins"], tie)
        outcomes.append({**res, "case": case_key(case), "finish_reason": rec["finish_reason"], "accounting": metrics.accounting(rec)["problems"]})
    summary = metrics.check_summary(outcomes)
    summary["accounting_problems"] = sum(1 for o in outcomes if o.get("accounting"))
    if summary["accounting_problems"]:
        summary["passed"] = False
    summary["tie_margin"] = tie
    summary["details"] = [o for o in outcomes if o["outcome"] != "exact"][:20]
    return [r for _, r in pairs], {"check": summary}


def run_cancel(api, phase, spec, ref):
    """Streams dropped after a few tokens (Exo cancels the task on disconnect), interleaved with complete streams; afterwards a follow-up check proves the instance still answers correctly."""
    msgs = messages_for(spec["prompts"][phase["prompt"]])
    records, ths = [None] * int(phase.get("streams", 4)), []
    every = int(phase.get("complete_every", 2))

    def one(i):
        drop = None if (i + 1) % every == 0 else int(phase.get("drop_after_events", 4))
        records[i] = stream_request(api, {"phase": phase["name"], "index": i, "prompt": phase["prompt"], "conc": len(records), "tag": spec.get("tag"), "warmup": False}, msgs,
                                    int(phase["max_tokens"]), logprobs=True, top_logprobs=0, drop_after_events=drop)
    for i in range(len(records)):
        ths.append(threading.Thread(target=one, args=(i,)))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    time.sleep(float(phase.get("settle_s", 4)))
    follow = dict(phase["followup"], name=phase["name"] + "-followup")
    recs, extra = run_check(api, follow, spec, ref)
    cancelled = sum(1 for r in records if r["status"] == "cancelled")
    completed = sum(1 for r in records if r["status"] == "ok")
    extra["cancel"] = {"passed": extra["check"]["passed"] and cancelled >= 1, "cancelled": cancelled, "completed": completed, "followup": extra["check"]}
    return records + recs, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--spec-b64", help="the same spec, base64-encoded JSON (so a remote client needs no file transfer)")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if bool(a.spec) == bool(a.spec_b64):
        ap.error("exactly one of --spec and --spec-b64 is required")
    spec = json.load(open(a.spec)) if a.spec else json.loads(base64.b64decode(a.spec_b64))
    api = Api(spec)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    result = {"model": spec["model"], "backend": spec["backend"], "steps": {}}
    nodes_s = wait_nodes(api, int(spec["node_count"]), float(spec.get("discovery_timeout_s", 120)), float(spec.get("poll_s", 1.0)), float(spec.get("stable_s", 3.0)))
    result["steps"]["nodes_s"] = nodes_s
    iid, st, place_s = place(api, spec)
    result["steps"]["place_s"] = place_s
    lay = validation.layout(st, iid)
    problems = validation.check_placement(st, backend=spec["backend"], model_id=spec["model"], world_size=int(spec["world_size"]), layers=spec.get("expected_layers"), instance_id=iid,
                                         link_addresses=spec.get("link_addresses"))
    line("TBCCL_EXO_STATE", {"step": "placed", "layout": lay, "problems": problems, "state": st})
    emit({"event": "placement", "layout": lay, "problems": problems})
    status = "ok"
    try:
        if problems:
            raise RuntimeError("placement does not match the experiment: " + "; ".join(problems))
        load_s, st, seen = wait_ready(api, iid, float(spec.get("load_timeout_s", 900)), float(spec.get("poll_s", 1.0)))
        result["steps"]["load_s"] = load_s
        ids, views = peer_views(spec, iid)
        agreement = validation.check_agreement({ids[n]: v for n, v in views.items()}, iid) if views else []
        line("TBCCL_EXO_STATE", {"step": "ready", "runner_timeline": seen, "state": st, "peer_node_ids": ids, "peer_states": views, "agreement_problems": agreement})
        emit({"event": "ready", "load_s": load_s, "runner_timeline": seen, "node_ids": ids, "agreement_problems": agreement})
        if agreement:
            raise RuntimeError("the nodes do not agree on the cluster state: " + "; ".join(agreement))
        ref = json.load(open(spec["reference"])) if spec.get("reference") else None
        for phase in spec["phases"]:
            extra = {}
            if phase["kind"] == "recreate":
                delete_s = delete(api, iid)
                iid, st, place_s = place(api, spec)
                lay = validation.layout(st, iid)
                problems = validation.check_placement(st, backend=spec["backend"], model_id=spec["model"], world_size=int(spec["world_size"]), layers=spec.get("expected_layers"), instance_id=iid,
                                                      link_addresses=spec.get("link_addresses"))
                emit({"event": "placement", "layout": lay, "problems": problems, "recreated": phase["name"]})
                if problems:
                    raise RuntimeError("placement after recreation does not match the experiment: " + "; ".join(problems))
                load_s, st, seen = wait_ready(api, iid, float(spec.get("load_timeout_s", 900)), float(spec.get("poll_s", 1.0)))
                emit({"event": "phase", "phase": phase["name"], "kind": "recreate", "delete_s": delete_s, "place_s": place_s, "load_s": load_s, "runner_timeline": seen})
                continue
            if phase["kind"] == "reference":
                records, extra, ref = run_reference(api, phase, spec)  # later check phases of this lifecycle compare with it
                wall = 0.0
            elif phase["kind"] in ("check", "cancel"):
                if ref is None:
                    raise RuntimeError(f"phase {phase['name']} needs spec.reference")
                t0 = now_ns()
                records, extra = (run_check if phase["kind"] == "check" else run_cancel)(api, phase, spec, ref)
                wall = (now_ns() - t0) / 1e9
            else:
                records, wall = run_requests(api, phase, spec, spec.get("tag"))
            for r in records:
                line("TBCCL_EXO_RAW", {"phase": phase["name"], "request": r})
            base = {"backend": spec["backend"], "wall_s": wall, "conc": phase.get("conc", 1), "max_tokens": phase.get("max_tokens")}
            emit({"event": "phase", **summarize_phase(phase, records, base), **extra})
    except Exception as e:  # noqa: BLE001
        status = "error"
        emit({"event": "error", "error": f"{type(e).__name__}: {e}"})
    finally:
        try:
            result["steps"]["delete_s"] = delete(api, iid)
        except Exception as e:  # noqa: BLE001
            status = "error"
            emit({"event": "error", "error": f"delete: {e}"})
    emit({"event": "done", "status": status, "steps": result["steps"]})
    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
