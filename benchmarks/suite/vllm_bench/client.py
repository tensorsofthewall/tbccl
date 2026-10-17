"""Streaming benchmark and validation client for a running vLLM OpenAI-compatible server (standard library only).

    python client.py --spec SPEC.json --out-dir DIR

The spec (written by the suite) names the API, the model, the fixed prompt file and an ordered list of phases. Each phase writes its raw per-request timelines to
``DIR/<phase>.requests.jsonl`` and prints one ``TBCCL_VLLM_BENCH <json>`` summary line; the suite parses only those lines. The summaries are derived from the raw timelines
by ``metrics.py`` and the raw files are what reports use.

Phase kinds
  ready        wait for the API (and report which metrics endpoints exist)
  warmup       untimed sequential requests (their timelines are kept but flagged)
  rounds       closed loop in rounds: ``conc`` simultaneous requests per round, the next round starts when every request of the round has ended
  continuous   closed loop with ``conc`` workers that each issue their next request when the previous one ends
  open_loop    Poisson arrivals at ``rate`` requests/s (offered), recorded against the achieved rate
  reference    greedy generations at one request at a time, stored as a reference with the top-1/top-2 log-probability margin of every position
  check        requests compared with a stored reference (exact, or divergence at a near-tie position)
  cancel       streams dropped after a few events, the engine drained, then deterministic follow-up requests compared with the reference
Client placement, arrival pattern and loop type are recorded in every summary: closed-loop and open-loop results are different workloads.
"""
import argparse
import base64
import hashlib
import http.client
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

PREFIX = "TBCCL_VLLM_BENCH"
SCRAPE_KEYS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
               "vllm:num_preemptions_total", "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds",
               "vllm:request_inference_time_seconds", "vllm:e2e_request_latency_seconds", "vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
               "vllm:request_time_per_output_token_seconds", "vllm:request_success_total")


def emit(obj):
    print(f"{PREFIX} {json.dumps(obj)}", flush=True)


class Sink:
    """Raw evidence leaves the client on stdout (``TBCCL_VLLM_RAW`` per request, ``TBCCL_VLLM_SCRAPE`` per metrics sample, ``TBCCL_VLLM_REFERENCE``), so a client on a remote host needs no
    file transfer; with ``--out-dir`` the same records are also written to files."""

    def __init__(self, out_dir=None):
        self.out_dir = out_dir
        self.lock = threading.Lock()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def line(self, prefix, obj):
        with self.lock:
            print(f"{prefix} {json.dumps(obj)}", flush=True)

    def raw(self, phase, records):
        for r in records:
            self.line("TBCCL_VLLM_RAW", {"phase": phase, "request": r})
        if self.out_dir:
            write_raw(os.path.join(self.out_dir, phase + ".requests.jsonl"), records)

    def scrape(self, phase, sample):
        self.line("TBCCL_VLLM_SCRAPE", {"phase": phase, **sample})

    def reference(self, name, doc):
        self.line("TBCCL_VLLM_REFERENCE", {"name": name, "reference": doc})
        if self.out_dir:
            json.dump(doc, open(os.path.join(self.out_dir, name + ".reference.json"), "w"))


SINK = Sink()


def now_ns():
    return time.monotonic_ns()


class Api:
    def __init__(self, spec):
        self.host, self.port, self.model = spec["api"]["host"], spec["api"]["port"], spec["model"]
        self.timeout = float(spec.get("request_timeout_s", 120))

    def conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)

    def get(self, path, timeout=5):
        c = self.conn(timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def body(self, ids, max_tokens, extra=None):
        b = {"model": self.model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True,
             "stream_options": {"include_usage": True, "continuous_usage_stats": True}, "return_token_ids": True}
        b.update(extra or {})
        return b


def stream_request(api, meta, ids, max_tokens, extra=None, drop_after_events=None, timeout=None):
    """One streamed completion. Returns the timeline record. Never raises: every failure becomes a status and an error string."""
    tl = metrics.Timeline({**meta, "prompt_tokens_planned": len(ids), "max_tokens": max_tokens}, now_ns())
    c = api.conn(timeout)
    try:
        c.request("POST", "/v1/completions", json.dumps(api.body(ids, max_tokens, extra)), {"Content-Type": "application/json"})
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
        if tl.t_done_ns is None:
            tl.finish("partial", now_ns(), "stream ended without [DONE]")
        elif not tl.events:
            tl.finish("error", now_ns(), "stream completed without any output event")
        elif tl.finish_reason is None:  # a well-formed but truncated stream: the engine died mid-request and the API closed it
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


def plain_request(api, ids, max_tokens, extra=None, timeout=None):
    """Non-streaming completion (references need the log-probabilities)."""
    b = {"model": api.model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    b.update(extra or {})
    c = api.conn(timeout)
    try:
        c.request("POST", "/v1/completions", json.dumps(b), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        c.close()


def parse_metrics_text(text):
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#" or "_bucket" in line.split("{")[0].split(" ")[0]:
            continue
        name = line.split("{")[0].split(" ")[0]
        if name.endswith("_created") or not any(name.startswith(k) for k in SCRAPE_KEYS):
            continue
        try:
            head, value = line.rsplit(" ", 1)
            out[head] = float(value)
        except ValueError:
            continue
    return out


class Scraper(threading.Thread):
    """Samples the server's /metrics endpoint during a phase; the snapshots are raw evidence (queue depth, running requests, server-side histograms' sum and count)."""

    def __init__(self, api, phase, interval):
        super().__init__(daemon=True)
        self.api, self.phase, self.interval = api, phase, interval
        self.stop_event = threading.Event()
        self.samples = 0
        self.errors = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                status, body = self.api.get("/metrics", timeout=3)
                if status == 200:
                    SINK.scrape(self.phase, {"t_ns": now_ns(), "values": parse_metrics_text(body.decode("utf-8", "replace"))})
                    self.samples += 1
                else:
                    self.errors += 1
            except OSError:
                self.errors += 1
            self.stop_event.wait(self.interval)

    def finish(self):
        self.stop_event.set()
        self.join(timeout=5)


def load_prompts(spec):
    return json.load(open(spec["workload"]))["prompts"]


def write_raw(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def summarize_phase(phase, records, base):
    d = metrics.request_distributions(records)
    t = metrics.phase_throughput(records)
    return {**base, "phase": phase["name"], "kind": phase["kind"], "requests": t["requests"], "throughput": t, "distributions": d,
            "problems": sorted({p for r in records if r.get("status") == "ok" for p in metrics.derive(r)["problems"]})}


def slot_plan(phase, prompts):
    """The (prompt name, max_tokens) of every concurrent slot: one entry broadcast to ``conc`` slots, or an explicit mixed list."""
    mixed = phase.get("mix")
    if mixed:
        return [(m["prompt"], m["tokens"]) for m in mixed]
    return [(phase["prompt"], phase["tokens"])] * phase.get("conc", 1)


def run_rounds(api, phase, prompts, tag, extra_meta=None):
    plan = slot_plan(phase, prompts)
    conc = len(plan)
    stagger = float(phase.get("stagger_s", 0.0))
    rng = random.Random(phase.get("seed", 7))
    records = []
    lock = threading.Lock()
    walls = []
    for rnd in range(phase["rounds"]):
        delays = [rng.random() * stagger for _ in range(conc)] if stagger else [0.0] * conc
        barrier = threading.Barrier(conc + 1)
        results = [None] * conc

        def worker(i):
            barrier.wait()
            if delays[i]:
                time.sleep(delays[i])
            name, ntok = plan[i]
            results[i] = stream_request(api, {"phase": phase["name"], "round": rnd, "slot": i, "request_index": rnd * conc + i, "prompt": name, **(extra_meta or {})}, prompts[name], ntok,
                                        timeout=phase.get("request_timeout_s"))

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conc)]
        for t in threads:
            t.start()
        barrier.wait()
        t0 = now_ns()
        for t in threads:
            t.join()
        walls.append((now_ns() - t0) / 1e9)
        with lock:
            records.extend(results)
        if any(r["status"] in ("timeout",) for r in results):
            break
    return records, {"round_wall_s": walls}


def run_continuous(api, phase, prompts):
    plan = slot_plan(phase, prompts)
    conc = len(plan)
    total = phase["requests"]
    counter = {"next": 0}
    lock = threading.Lock()
    records = []

    def worker(slot):
        name, ntok = plan[slot]
        while True:
            with lock:
                i = counter["next"]
                if i >= total:
                    return
                counter["next"] += 1
            rec = stream_request(api, {"phase": phase["name"], "round": 0, "slot": slot, "request_index": i, "prompt": name}, prompts[name], ntok, timeout=phase.get("request_timeout_s"))
            with lock:
                records.append(rec)
            if rec["status"] == "timeout":
                return

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(conc)]
    t0 = now_ns()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(records, key=lambda r: r["request_index"]), {"wall_s": (now_ns() - t0) / 1e9}


def run_open_loop(api, phase, prompts):
    rng = random.Random(phase.get("seed", 11))
    rate, total = float(phase["rate"]), phase["requests"]
    plan = slot_plan({**phase, "conc": 1}, prompts)
    records, threads, lock = [], [], threading.Lock()
    scheduled = []
    t = 0.0
    for _ in range(total):
        t += rng.expovariate(rate)
        scheduled.append(t)

    def one(i, planned_ns):
        name, ntok = plan[i % len(plan)]
        rec = stream_request(api, {"phase": phase["name"], "round": 0, "slot": 0, "request_index": i, "prompt": name, "scheduled_offset_s": scheduled[i]}, prompts[name], ntok,
                             timeout=phase.get("request_timeout_s"))
        rec["scheduled_ns"] = planned_ns
        with lock:
            records.append(rec)

    start = now_ns()
    for i, off in enumerate(scheduled):
        target = start + int(off * 1e9)
        while True:
            lag = (target - now_ns()) / 1e9
            if lag <= 0:
                break
            time.sleep(min(lag, 0.05))
        th = threading.Thread(target=one, args=(i, target), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = (now_ns() - start) / 1e9
    return sorted(records, key=lambda r: r["request_index"]), {"offered_rate_per_s": rate, "arrival_span_s": scheduled[-1] if scheduled else 0.0, "wall_s": wall}


def margin_of(top):
    v = sorted(top.values(), reverse=True)
    return (v[0] - v[1]) if len(v) > 1 else None


def make_reference(api, phase, prompts):
    cases = []
    for case in phase["cases"]:
        ids = prompts[case["prompt"]]
        d = plain_request(api, ids, case["tokens"], {"logprobs": 5}, timeout=phase.get("request_timeout_s"))
        ch = d["choices"][0]
        lp = ch["logprobs"]
        cases.append({"name": case["name"], "prompt": case["prompt"], "prompt_tokens": len(ids), "token_ids": ch["token_ids"], "text": ch["text"],
                      "margins": [margin_of(t) for t in lp["top_logprobs"]], "usage": d.get("usage")})
    return cases


def compare(tokens, ref, tie_margin):
    """Token ids against a reference: ('exact'|'near_tie'|'diverged'|'short', first divergent position, reference margin there)."""
    want = ref["token_ids"]
    n = min(len(tokens), len(want))
    for i in range(n):
        if tokens[i] != want[i]:
            m = ref["margins"][i] if i < len(ref["margins"]) else None
            return ("near_tie" if m is not None and m <= tie_margin else "diverged"), i, m
    if len(tokens) < len(want):
        return "short", len(tokens), None
    return "exact", None, None


def run_check(api, phase, prompts, refs, tie_margin, out_dir=None):
    """Requests (sequential, concurrent, mixed-length, staggered, repeated) compared with the reference; any 'diverged' or 'short' outcome fails the gate."""
    by_name = {c["name"]: c for c in refs["cases"]}
    jobs = []
    for rep in range(phase.get("repeat", 1)):
        for name in phase["cases"]:
            ref = by_name[name]
            ntok = min(phase.get("tokens") or len(ref["token_ids"]), len(ref["token_ids"]))
            jobs.append((name, ntok, rep))
    conc = phase.get("conc", 1)
    stagger = float(phase.get("stagger_s", 0.0))
    rng = random.Random(phase.get("seed", 5))
    if phase.get("random_len"):
        jobs = [(n, rng.randint(max(3, ntok // 4), ntok), rep) for n, ntok, rep in jobs]
    delays = [rng.random() * stagger for _ in jobs] if stagger else [0.0] * len(jobs)
    if conc > len(jobs):
        reps = (conc + len(jobs) - 1) // len(jobs)
        jobs = (jobs * reps)[:conc]
        delays = (delays * reps)[:conc]
    results = [None] * len(jobs)
    sem = threading.Semaphore(conc)

    def work(i):
        name, ntok, rep = jobs[i]
        with sem:
            if delays[i]:
                time.sleep(delays[i])
            results[i] = stream_request(api, {"phase": phase["name"], "round": rep, "slot": i, "request_index": i, "prompt": by_name[name]["prompt"], "case": name},
                                        prompts[by_name[name]["prompt"]], ntok, timeout=phase.get("request_timeout_s"))

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    outcome = {"exact": 0, "near_tie": 0, "diverged": 0, "short": 0, "request_failed": 0}
    details = []
    for (name, ntok, rep), rec in zip(jobs, results):
        if rec["status"] != "ok":
            outcome["request_failed"] += 1
            details.append({"case": name, "result": "request_failed", "error": rec.get("error")})
            continue
        got = rec["token_ids"] or []
        ref = by_name[name]
        res, pos, m = compare(got, {"token_ids": ref["token_ids"][:ntok], "margins": ref["margins"][:ntok]}, tie_margin)
        outcome[res] += 1
        if res != "exact":
            details.append({"case": name, "result": res, "position": pos, "reference_margin": m, "generated": len(got), "expected": ntok})
    SINK.raw(phase["name"], results)
    return results, {"outcome": outcome, "details": details, "tie_margin": tie_margin, "passed": outcome["diverged"] == 0 and outcome["short"] == 0 and outcome["request_failed"] == 0,
                     "concurrency": conc, "staggered": bool(stagger)}


def run_cancel(api, phase, prompts, refs, tie_margin, out_dir=None):
    """Start streams with a long output, drop most after a few events, wait until the server reports it idle, then check deterministic follow-up requests."""
    n, keep = phase["streams"], phase.get("complete_every", 0)
    long_prompt = prompts[phase["prompt"]]
    results = [None] * n

    def one(i):
        drop = None if (keep and i % keep == keep - 1) else phase.get("drop_after_events", 4)
        results[i] = stream_request(api, {"phase": phase["name"], "round": 0, "slot": i, "request_index": i, "prompt": phase["prompt"]}, long_prompt, phase["tokens"],
                                    drop_after_events=drop, timeout=phase.get("request_timeout_s"))

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    drained, idle_after_s, last = False, None, None
    t0 = now_ns()
    while (now_ns() - t0) / 1e9 < phase.get("drain_timeout_s", 30):
        try:
            status, body = api.get("/metrics", timeout=3)
        except OSError:
            break
        if status != 200:
            break
        vals = parse_metrics_text(body.decode("utf-8", "replace"))
        run = sum(v for k, v in vals.items() if k.startswith("vllm:num_requests_running"))
        wait = sum(v for k, v in vals.items() if k.startswith("vllm:num_requests_waiting"))
        last = {"running": run, "waiting": wait}
        if run == 0 and wait == 0:
            drained, idle_after_s = True, (now_ns() - t0) / 1e9
            break
        time.sleep(0.25)
    SINK.raw(phase["name"], results)
    follow = run_check(api, {**phase["followup"], "name": phase["name"] + "-followup"}, prompts, refs, tie_margin, out_dir)
    cancelled = sum(1 for r in results if r["status"] == "cancelled")
    return results, {"streams": n, "cancelled": cancelled, "completed": sum(1 for r in results if r["status"] == "ok"), "drained": drained, "idle_after_s": idle_after_s,
                     "last_metrics": last, "metrics_available": last is not None, "followup": follow[1],
                     "passed": follow[1]["passed"] and cancelled >= 1 and (drained or last is None)}


def wait_ready(api, phase):
    t0 = now_ns()
    deadline = t0 + int(phase.get("timeout_s", 600) * 1e9)
    last = None
    while now_ns() < deadline:
        try:
            status, _ = api.get("/v1/models", timeout=3)
            if status == 200:
                break
            last = f"HTTP {status}"
        except OSError as e:
            last = str(e)
        time.sleep(1.0)
    else:
        return None, {"ready": False, "waited_s": (now_ns() - t0) / 1e9, "last_error": last}
    info = {"ready": True, "waited_s": (now_ns() - t0) / 1e9}
    for path in ("/metrics", "/health", "/version"):
        try:
            status, body = api.get(path, timeout=3)
            info[path] = status
            if path == "/version" and status == 200:
                info["version"] = json.loads(body)
            if path == "/metrics" and status == 200:
                info["metric_names"] = sorted({ln.split("{")[0].split(" ")[0] for ln in body.decode().splitlines() if ln and ln[0] != "#" and "_bucket" not in ln.split("{")[0]})
        except (OSError, ValueError):
            info[path] = None
    return None, info


def main():
    global SINK
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help="spec JSON file")
    ap.add_argument("--spec-b64", default=None, help="the spec as base64 JSON (used when the suite starts the client on a remote host)")
    ap.add_argument("--out-dir", default=None, help="also write the raw records to files here (they are always printed)")
    a = ap.parse_args()
    if bool(a.spec) == bool(a.spec_b64):
        raise SystemExit("exactly one of --spec and --spec-b64 is required")
    spec = json.load(open(a.spec)) if a.spec else json.loads(base64.b64decode(a.spec_b64))
    SINK = Sink(a.out_dir)
    api = Api(spec)
    for key, path in (("workload_sha256", spec.get("workload")), ("reference_sha256", spec.get("reference"))):
        want = spec.get(key)
        if want and path:
            have = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if have != want:
                print(f"{path} does not match the expected {key} ({have[:12]} != {want[:12]})", file=sys.stderr)
                return 6
    prompts = load_prompts(spec)
    inputs = {k: hashlib.sha256(open(spec[k], "rb").read()).hexdigest() for k in ("workload", "reference") if spec.get(k) and os.path.exists(spec[k])}
    refs = json.load(open(spec["reference"])) if spec.get("reference") and os.path.exists(spec["reference"]) else None
    tie = float(spec.get("tie_margin", 0.251))
    placement = {"client_host_role": spec.get("client_role", "api-local"), "api": f"{api.host}"}
    gate_failed = False
    for phase in spec["phases"]:
        kind = phase["kind"]
        base = {"client": placement}
        scraper = None
        if kind in ("rounds", "continuous", "open_loop", "check", "cancel") and phase.get("scrape", True):
            scraper = Scraper(api, phase["name"], float(phase.get("scrape_interval_s", 0.5)))
            scraper.start()
        started = time.time()
        try:
            if kind == "ready":
                _, info = wait_ready(api, phase)
                emit({**base, "phase": phase["name"], "kind": kind, "input_sha256": inputs, **info})
                if not info["ready"]:
                    return 3
                continue
            if kind == "warmup":
                recs = [stream_request(api, {"phase": phase["name"], "round": 0, "slot": 0, "request_index": i, "prompt": phase["prompt"], "warmup": True}, prompts[phase["prompt"]], phase["tokens"])
                        for i in range(phase["requests"])]
                SINK.raw(phase["name"], recs)
                emit({**base, "phase": phase["name"], "kind": kind, "requests": len(recs), "ok": sum(1 for r in recs if r["status"] == "ok")})
                if any(r["status"] != "ok" for r in recs):
                    print("warm-up request failed", file=sys.stderr)
                    return 4
                continue
            if kind == "reference":
                cases = make_reference(api, phase, prompts)
                refs = {"schema": "tbccl-vllm-reference/1", "cases": cases, "workload": os.path.basename(spec["workload"])}
                SINK.reference(phase["name"], refs)
                emit({**base, "phase": phase["name"], "kind": kind, "cases": len(cases), "min_margin": min((m for c in cases for m in c["margins"] if m is not None), default=None)})
                continue
            if kind == "check":
                recs, res = run_check(api, phase, prompts, refs, tie, a.out_dir)
                emit({**summarize_phase(phase, recs, base), "check": res})
                gate_failed |= not res["passed"]
                continue
            if kind == "cancel":
                recs, res = run_cancel(api, phase, prompts, refs, tie, a.out_dir)
                emit({**summarize_phase(phase, recs, base), "cancel": res})
                gate_failed |= not res["passed"]
                continue
            if kind == "rounds":
                recs, extra = run_rounds(api, phase, prompts, phase["name"])
                loop = {"loop": "closed-rounds", "concurrency": len(slot_plan(phase, prompts)), "rounds": phase["rounds"], "arrival": "simultaneous" if not phase.get("stagger_s") else f"staggered<= {phase['stagger_s']}s"}
            elif kind == "continuous":
                recs, extra = run_continuous(api, phase, prompts)
                loop = {"loop": "closed-continuous", "concurrency": len(slot_plan(phase, prompts)), "arrival": "on-completion"}
            elif kind == "open_loop":
                recs, extra = run_open_loop(api, phase, prompts)
                loop = {"loop": "open-poisson", "arrival": "poisson"}
            else:
                raise SystemExit(f"unknown phase kind {kind}")
            SINK.raw(phase["name"], recs)
            wall = extra.get("wall_s")
            summary = summarize_phase(phase, recs, {**base, **loop, **{k: v for k, v in extra.items() if k != "wall_s"}})
            if kind == "rounds":
                summary["throughput"]["per_round"] = [metrics.phase_throughput([r for r in recs if r["round"] == i], w)["rates"] for i, w in enumerate(extra["round_wall_s"])]
            summary["prompt"] = phase.get("prompt")
            summary["max_tokens"] = phase.get("tokens")
            emit(summary)
            if not phase.get("allow_failures"):
                if any(r["status"] == "timeout" for r in recs):
                    return 5
                if any(r["status"] != "ok" for r in recs):
                    print(f"phase {phase['name']}: {sum(1 for r in recs if r['status'] != 'ok')} request(s) did not complete normally", file=sys.stderr)
                    return 8
        finally:
            if scraper:
                scraper.finish()
                emit({"phase": phase["name"], "kind": "scrape", "samples": scraper.samples, "errors": scraper.errors, "duration_s": round(time.time() - started, 3)})
    emit({"phase": "end", "kind": "end", "gate_failed": gate_failed})
    if gate_failed:
        print("correctness gate failed: see the check/cancel summaries", file=sys.stderr)
    return 2 if gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())
