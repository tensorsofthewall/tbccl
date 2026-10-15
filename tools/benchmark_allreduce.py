#!/usr/bin/env python3
"""A reproducible all_reduce benchmark harness for torch-tbccl: methodology first, numbers second.

    python tools/benchmark_allreduce.py                     # two ranks on 127.0.0.1, prints a table and writes tbccl-benchmark.json
    python tools/benchmark_allreduce.py --backend gloo      # the same measurement through gloo, for a like-for-like loopback comparison
    python tools/benchmark_allreduce.py --rank 0 --world-size 2 --master-addr 192.168.3.1 --endpoint 192.168.3.1:0 --out host0.json   # one rank per host

It follows docs/development/benchmark-methodology.md: untimed warmup, a timed loop with no correctness checking inside, then one untimed verified
all_reduce per size. Every result file records the versions, platform, transport and settings it was produced with and says whether the ranks shared a
machine. Loopback numbers measure software overhead on one machine, not a network, and must not be quoted as TBCCL's performance on a link. The script
never configures a network, never starts anything on another host and downloads nothing; two-host runs are started by whoever owns both hosts.
"""
import argparse
import datetime
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

SCHEMA = "tbccl-benchmark/1"
DEFAULT_SIZES = "1024,65536,1048576,16777216"
NOTE = ("Transport is TCP over IPv4 (over Thunderbolt IP on a link), not RDMA. Algorithmic bandwidth is payload bytes divided by the time of one all_reduce "
        "as seen by rank 0, with no bus-bandwidth correction. Loopback results measure one machine's software and memory overhead, not a network.")


def summarize(per_iter_seconds, nbytes):
    """Statistics over the repeats, each repeat being the mean seconds per all_reduce of one timed loop."""
    med = statistics.median(per_iter_seconds)
    return {"repeats": len(per_iter_seconds), "seconds_per_op_median": med, "seconds_per_op_min": min(per_iter_seconds), "seconds_per_op_max": max(per_iter_seconds),
            "algbw_gbps_median": nbytes / med / 1e9 if med > 0 else None}


def is_loopback(addr):
    return addr in ("127.0.0.1", "localhost", "::1") or addr.startswith("127.")


def environment(backend, master_addr, world_size):
    env = {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}", "cpu_count": os.cpu_count(), "backend": backend,
           "world_size": world_size, "master_addr_is_loopback": is_loopback(master_addr)}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    if backend == "tbccl":
        try:
            import torch_tbccl
            env["torch_tbccl"] = getattr(torch_tbccl, "__version__", None)
            env["libtbccl"] = torch_tbccl.runtime_version()
        except Exception as e:  # report, do not hide
            env["torch_tbccl_error"] = repr(e)
    for var in ("OMP_NUM_THREADS", "TBCCL_LOCAL_ENDPOINT", "TBCCL_ALGORITHM"):
        if os.environ.get(var) is not None:
            env[var] = os.environ[var]
    return env


def build_report(env, settings, results, label):
    return {"schema": SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "label": label, "environment": env, "settings": settings,
            "results": results, "note": NOTE}


def rank_main(args):
    import torch
    import torch.distributed as dist

    if args.backend == "tbccl":
        import torch_tbccl  # noqa: F401
    dist.init_process_group(args.backend, timeout=datetime.timedelta(seconds=args.timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    results = []
    for nbytes in (int(s) for s in args.sizes.split(",")):
        n = max(nbytes // 4, 1)
        x = torch.ones(n, dtype=torch.float32)
        for _ in range(args.warmup):
            dist.all_reduce(x)
        per_repeat = []
        for _ in range(args.repeats):
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                dist.all_reduce(x)
            dist.barrier()
            per_repeat.append((time.perf_counter() - t0) / args.iters)
        y = torch.full((n,), float(rank + 1), dtype=torch.float32)  # untimed, verified once
        dist.all_reduce(y)
        verified = bool(torch.all(y == sum(range(1, world + 1))).item())
        results.append({"bytes": n * 4, **summarize(per_repeat, n * 4), "verified": verified})
        if rank == 0 and not args.quiet:
            r = results[-1]
            print(f"{r['bytes']:>10} B  {r['seconds_per_op_median'] * 1e3:>10.3f} ms  {r['algbw_gbps_median']:>7.3f} GB/s  verified={verified}", flush=True)
    dist.barrier()
    code = 0
    if rank == 0:
        settings = {"sizes": args.sizes, "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats, "dtype": "float32", "op": "all_reduce SUM", "device": "cpu"}
        report = build_report(environment(args.backend, os.environ["MASTER_ADDR"], world), settings, results, args.label)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")
        code = 0 if all(r["verified"] for r in results) else 1
    dist.destroy_process_group()
    return code


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launcher(args):
    base = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port()), "WORLD_SIZE": str(args.world_size), "TBCCL_LOCAL_ENDPOINT": "127.0.0.1:0"}
    base.setdefault("OMP_NUM_THREADS", "1")
    cmd = [sys.executable, __file__, *sys.argv[1:]]
    procs = [subprocess.Popen(cmd, env={**base, "RANK": str(r), "LOCAL_RANK": str(r)}) for r in range(args.world_size)]
    codes = []
    for p in procs:
        try:
            codes.append(p.wait(timeout=args.launch_timeout))
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            sys.exit(f"FAIL: the ranks did not finish within {args.launch_timeout} s")
    return 0 if codes == [0] * args.world_size else 1


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("tbccl", "gloo"), default="tbccl")
    ap.add_argument("--sizes", default=DEFAULT_SIZES, help="payload bytes, comma separated (float32 elements, rounded down to a multiple of 4)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50, help="all_reduce calls inside one timed loop")
    ap.add_argument("--repeats", type=int, default=5, help="timed loops per size")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--label", default="", help="free text stored in the result, e.g. what you changed since the last run")
    ap.add_argument("--out", default="tbccl-benchmark.json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--rank", type=int, help="run as one rank of a group you started yourself (use with --master-addr, --master-port, --endpoint)")
    ap.add_argument("--master-addr")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--endpoint", help="this rank's TBCCL_LOCAL_ENDPOINT host:port (port 0 picks a free one)")
    ap.add_argument("--launch-timeout", type=float, default=600.0)
    return ap.parse_args(argv)


def main():
    args = parse()
    if args.rank is not None:
        if not args.master_addr:
            sys.exit("--rank needs --master-addr")
        os.environ.update({"RANK": str(args.rank), "WORLD_SIZE": str(args.world_size), "MASTER_ADDR": args.master_addr, "MASTER_PORT": str(args.master_port)})
        if args.endpoint:
            os.environ["TBCCL_LOCAL_ENDPOINT"] = args.endpoint
    if "RANK" in os.environ:
        return rank_main(args)
    return launcher(args)


if __name__ == "__main__":
    sys.exit(main())
