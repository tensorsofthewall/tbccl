#!/usr/bin/env python3
"""Phase 55: stage-by-stage latency report from TBCCL_LATENCY_TRACE files (loopback: both ranks share one clock).

    latency_trace_report.py <rank0 trace file> <rank1 trace file> [--skip N]

Operations are correlated by FIFO order per direction: the k-th send of rank 0 pairs with the k-th receive of rank 1 and the k-th send of rank 1 with
the k-th receive of rank 0. Prints medians (microseconds) of each stage of a one-way transfer and of the receive-side completion, then the
ping-pong round-trip composition.  Sites: 0 submit, 1 enqueued, 2 dequeued, 3 send enter, 4 send return, 5 recv first bytes, 6 header ok,
7 payload done, 8 terminal, 9 waiter awake, 10 wait return.
"""
import statistics
import sys
from collections import defaultdict


def load(path):
    ops = defaultdict(dict)
    meta = {}
    for line in open(path):
        ns, oid, site, aux = line.strip().split(",")
        oid, site, aux = int(oid), int(site), int(aux)
        ops[oid][site] = int(ns)
        if site == 0:
            meta[oid] = ("recv" if aux >> 31 else "send", aux & 0x7FFFFFFF)
    sends = [(o, ops[o]) for o in sorted(ops) if meta.get(o, ("",))[0] == "send"]
    recvs = [(o, ops[o]) for o in sorted(ops) if meta.get(o, ("",))[0] == "recv"]
    return sends, recvs


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stage(values, a, b, ops):
    return [(op[b] - op[a]) / 1000.0 for _, op in ops if a in op and b in op]


def solo(path, skip):
    """One process's own view (real link: the two hosts' clocks are unrelated, so only local stage durations are meaningful)."""
    sends, recvs = load(path)
    sends, recvs = sends[skip:], recvs[skip:]
    print(f"{path}: sends={len(sends)} recvs={len(recvs)}")
    print("  SEND ops (this process)")
    for lab, a, b in (("submit entered -> enqueued", 0, 1), ("enqueued -> worker dequeued (caller->TX worker wake)", 1, 2),
                      ("dequeued -> sendmsg entered", 2, 3), ("sendmsg syscall", 3, 4), ("sendmsg returned -> Work terminal", 4, 8),
                      ("Work terminal -> waiter awake (completion wake)", 8, 9), ("whole op: submit -> wait returned", 0, 10)):
        print(f"    {lab:58s} {med(stage(None, a, b, sends)):8.1f} us")
    print("  RECV ops (this process)")
    for lab, a, b in (("submit entered -> enqueued", 0, 1), ("enqueued -> worker dequeued (RX worker wake)", 1, 2),
                      ("dequeued -> first bytes (waiting for the peer)", 2, 5), ("first bytes -> destination ready", 5, 7),
                      ("destination ready -> Work terminal", 7, 8), ("Work terminal -> waiter awake (completion wake)", 8, 9),
                      ("whole op: submit -> wait returned", 0, 10)):
        print(f"    {lab:58s} {med(stage(None, a, b, recvs)):8.1f} us")


def main():
    skip = 0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--skip" in sys.argv:
        skip = int(sys.argv[sys.argv.index("--skip") + 1])
        args = [a for a in args if a != str(skip)]
    if len(args) == 1:
        solo(args[0], skip)
        return
    s0, r0 = load(args[0])
    s1, r1 = load(args[1])
    s0, r0, s1, r1 = (x[skip:] for x in (s0, r0, s1, r1))
    print(f"ops after skipping {skip}: rank0 send={len(s0)} recv={len(r0)}  rank1 send={len(s1)} recv={len(r1)}")
    rows = []
    for name, sends in (("rank0->rank1", list(zip(s0, r1))), ("rank1->rank0", list(zip(s1, r0)))):
        send_ops = [s for s, _ in sends]
        recv_ops = [r for _, r in sends]
        cross = [(r[1][5] - s[1][3]) / 1000.0 for s, r in sends if 5 in r[1] and 3 in s[1]]  # send entered -> receiver first bytes
        cross_done = [(r[1][7] - s[1][3]) / 1000.0 for s, r in sends if 7 in r[1] and 3 in s[1]]  # send entered -> destination ready
        print(f"\n== {name}")
        print("  SENDER side (per send op)")
        for lab, a, b in (("submit entered -> enqueued (post_p2p + admission)", 0, 1), ("enqueued -> worker dequeued (caller->TX worker wake)", 1, 2),
                          ("dequeued -> sendmsg entered", 2, 3), ("sendmsg syscall", 3, 4), ("sendmsg returned -> Work terminal", 4, 8),
                          ("Work terminal -> waiter awake (completion wake)", 8, 9), ("submit entered -> wait returned (whole op)", 0, 10)):
            print(f"    {lab:58s} {med(stage(None, a, b, send_ops)):8.1f} us")
        print("  RECEIVER side (per recv op)")
        for lab, a, b in (("submit entered -> enqueued", 0, 1), ("enqueued -> worker dequeued (RX worker wake)", 1, 2),
                          ("recv first bytes -> destination ready", 5, 7), ("destination ready -> Work terminal", 7, 8),
                          ("Work terminal -> waiter awake (completion wake)", 8, 9), ("dequeued -> destination ready (incl. waiting for the peer)", 2, 7)):
            print(f"    {lab:58s} {med(stage(None, a, b, recv_ops)):8.1f} us")
        print(f"  CROSS: send entered -> receiver first bytes (syscall + kernel + RX wake)  {med(cross):8.1f} us")
        print(f"  CROSS: send entered -> destination ready                                  {med(cross_done):8.1f} us")
        rows.append((name, med(cross_done)))


if __name__ == "__main__":
    main()
