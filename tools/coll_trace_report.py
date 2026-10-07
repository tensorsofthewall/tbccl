#!/usr/bin/env python3
"""Per-stage latency of N=2 collectives (AllGather / the decode chain) from TBCCL_LATENCY_TRACE files, one process at a time.

    coll_trace_report.py <trace file> [--skip N] [--json]

Each collective is a parent Work (sites 11..15, 8, 9, 10) whose children are the point-to-point transfers the CollectiveExecutor runs (correlated by
the aux field of sites 14/15). Stages are medians over the collectives after `--skip` warm-up ones, microseconds, all on this process's clock:

  caller submit -> executor notified        site 11 -> 12      (admission into the executor queue)
  executor notified -> executor running     site 12 -> 13      (the executor thread's wake: a cold futex wake when it has been idle)
  per child k (0 = first transfer, 1 = second):
    executor free -> child posted           (13 or previous observe) -> 14
    child posted -> lane worker dequeued    child site 1 -> 2    (the lane thread's wake)
    child dequeued -> terminal              child site 2 -> 8    (socket work; for a receive this includes waiting for the peer)
    child terminal -> executor observed     child site 8 -> 15   (the executor's wake from its wait)
  last child observed -> parent terminal    15 -> 8
  parent terminal -> caller awake           8 -> 9               (the caller's wake)
  whole collective                          11 -> 10
"""
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    ops = defaultdict(dict)
    meta = {}
    posted = defaultdict(list)  # parent id -> [(child id, ns)] in posting order
    observed = defaultdict(dict)  # parent id -> {child id: ns}
    for line in open(path):
        ns, oid, site, aux = (int(x) for x in line.strip().split(","))
        if site == 14:
            posted[oid].append((aux, ns))
        elif site == 15:
            observed[oid][aux] = ns
        else:
            ops[oid][site] = ns
            if site == 0:
                meta[oid] = "recv" if aux >> 31 else "send"
    return ops, meta, posted, observed


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def report(path, skip):
    ops, meta, posted, observed = load(path)
    parents = [p for p in sorted(ops) if 11 in ops[p] and posted.get(p)][skip:]
    rows = defaultdict(list)
    for p in parents:
        e = ops[p]
        d = lambda a, b: (b - a) / 1000.0
        if 12 in e:
            rows["caller submit -> executor notified"].append(d(e[11], e[12]))
        if 13 in e:
            rows["submit -> executor running (whole executor wake)"].append(d(e[11], e[13]))
        prev = e.get(13)
        for k, (child, posted_ns) in enumerate(posted[p][:2]):
            c = ops.get(child, {})
            if child not in observed[p] or prev is None:
                continue
            label = f"child {k} ({meta.get(child, '?')})"
            rows[f"{label}: executor free -> child posted"].append(d(prev, posted_ns))
            if 1 in c and 2 in c:
                rows[f"{label}: posted -> lane worker dequeued (lane wake)"].append(d(c[1], c[2]))
            if 2 in c and 8 in c:
                rows[f"{label}: dequeued -> terminal (socket; recv includes the peer wait)"].append(d(c[2], c[8]))
            if 8 in c:
                rows[f"{label}: child terminal -> executor observed (executor wake)"].append(d(c[8], observed[p][child]))
            prev = observed[p][child]
        if prev is not None and 8 in e:
            rows["last child observed -> collective terminal"].append(d(prev, e[8]))
        if 8 in e and 9 in e:
            rows["collective terminal -> caller awake (caller wake)"].append(d(e[8], e[9]))
        if 10 in e:
            rows["whole collective: submit -> wait returned"].append(d(e[11], e[10]))
    out = {k: round(med(v), 2) for k, v in rows.items()}
    if "--json" in sys.argv:
        print(json.dumps({"n": len(parents), "stages_us": out}))
        return
    print(f"{path}: {len(parents)} collectives after skipping {skip}")
    for k, v in out.items():
        print(f"  {k:78s} {v:9.1f} us")


def main():
    skip = 0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--skip" in sys.argv:
        skip = int(sys.argv[sys.argv.index("--skip") + 1])
        args = [a for a in args if a != str(skip)]
    report(args[0], skip)


if __name__ == "__main__":
    main()
