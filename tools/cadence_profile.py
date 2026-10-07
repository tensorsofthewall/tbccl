#!/usr/bin/env python3
"""Communication-gap statistics and replay profiles from the call-timestamp dumps of exo-tbccl's benchmarks/cadence_recorder.py.

    cadence_profile.py --out DIR --tag TAG rank0.json rank1.json ...

Per rank and phase (prefill / decode) it reports the gap distribution: from the completion of one communication call to the submission of the next
(everything in between is the application's compute, wherever it ran), plus the repeating operation sequence. For the decode phase it writes
DIR/TAG.rank<r>_decode_profile.json, the input of `small_message_latency --profile`:  {"rank":R,"ops":[{"op":"send","bytes":2048,"gap_us":4300},...]}
holding the steady-state decode steps only (a gap longer than 3x the median gap of the same operation kind (and 2 ms) is the driver's own KV-cache digest, a
measurement artifact of benchmarks/real_model_loopback.py, so it is replaced by that median and counted) (the first two decode steps, which include kernel-compile warm-up, are dropped). No model data is stored: only call
kinds, payload sizes and durations.
"""
import argparse
import json
import statistics

COMM = {"send": "send", "recv_like": "recv", "all_gather": "all_gather", "any_true": "all_gather"}  # the end-of-run barrier is not decode traffic


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    summary = {}
    for path in a.files:
        d = json.load(open(path))
        rank = d["rank"]
        comm = [(t0, t1, COMM[n], nb, ph) for t0, t1, n, nb, ph in d["events"] if n in COMM]
        gaps = {"prefill": [], "decode": []}
        seq = {"prefill": [], "decode": []}
        for prev, cur in zip(comm, comm[1:]):
            gaps[cur[4]].append((cur[0] - prev[1]) / 1000.0)
            seq[cur[4]].append(cur[2])
        rec = {}
        for ph in ("prefill", "decode"):
            g = gaps[ph]
            if not g:
                continue
            rec[ph] = {"ops": len(g), "gap_us": {"median": round(statistics.median(g), 1), "p25": round(pct(g, .25), 1), "p75": round(pct(g, .75), 1),
                                                  "p90": round(pct(g, .9), 1), "p95": round(pct(g, .95), 1), "max": round(max(g), 1)}}
            by_kind = {}
            for kind in sorted(set(seq[ph])):
                kg = [x for x, k in zip(g, seq[ph]) if k == kind]
                by_kind[kind] = {"n": len(kg), "median_us": round(statistics.median(kg), 1), "p95_us": round(pct(kg, .95), 1)}
            rec[ph]["by_kind"] = by_kind
            unit = []
            for k in seq[ph]:
                if unit and k == unit[0] and len(unit) > 1:
                    break
                unit.append(k)
            rec[ph]["repeating_unit"] = unit
        dec = [c for c in comm if c[4] == "decode"]
        # steady state: drop the first two decode steps (each ends at its all_gather)
        ends = [i for i, c in enumerate(dec) if c[2] == "all_gather"]
        first_end = ends[1] if len(ends) > 1 else ends[0]
        steady = dec[first_end + 1:]
        origin = dec[first_end]
        ops, last_exit, replaced = [], origin[1], 0
        raw = [(t0 - prev_exit) / 1000.0 for (t0, _, _, _, _), prev_exit in zip(steady, [origin[1]] + [c[1] for c in steady])]
        kind_median = {k: statistics.median(g for g, c in zip(raw, steady) if c[2] == k) for k in {c[2] for c in steady}}
        for (t0, t1, kind, nbytes, _), gap in zip(steady, raw):
            if gap > 3 * kind_median[kind] and gap > 2000:
                gap, replaced = kind_median[kind], replaced + 1
            ops.append({"op": kind, "bytes": nbytes, "gap_us": max(0, int(round(gap)))})
        with open(f"{a.out}/{a.tag}.rank{rank}_decode_profile.json", "w") as f:
            json.dump({"rank": rank, "ops": ops}, f, separators=(",", ":"))
        rec["profile_ops"] = len(ops)
        rec["artifact_gaps_replaced"] = replaced
        summary[f"rank{rank}"] = rec
    with open(f"{a.out}/{a.tag}.gap_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for r, rec in summary.items():
        for ph in ("prefill", "decode"):
            if ph in rec:
                g = rec[ph]["gap_us"]
                print(f"{r} {ph:7s} n={rec[ph]['ops']:4d} gap us: median {g['median']} p25 {g['p25']} p75 {g['p75']} p90 {g['p90']} p95 {g['p95']} max {g['max']}  unit={rec[ph]['repeating_unit']}")
                for k, v in rec[ph]["by_kind"].items():
                    print(f"        before {k:10s} n={v['n']:3d} median {v['median_us']} us  p95 {v['p95_us']} us")


if __name__ == "__main__":
    main()
