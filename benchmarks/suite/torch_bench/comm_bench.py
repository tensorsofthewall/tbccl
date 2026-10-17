"""PyTorch communication micro-benchmark for the suite (one process per rank, world size 2, float32, dense contiguous tensors).

Operations (--op):
  all_reduce        blocking dist.all_reduce SUM
  all_reduce_async  dist.all_reduce(async_op=True): submission, Work.wait() and the Work's completion callback are timed separately
  broadcast         from --source to the other rank
  all_gather        one tensor of --bytes from each rank
  p2p_ack           --source sends --bytes with dist.send, the other rank receives and answers with a 1-byte CPU ACK; the source's time is
                    "completion confirmed" (the same semantics as tbccl_tensor_transfer_bench end-to-end: includes the ACK's return trip)
  isend_irecv_ack   as p2p_ack with isend/irecv and Work.wait(); submission and wait are timed separately
  p2p_rtt           ping-pong of --bytes both ways (the semantics of small_message_latency p2p)

Per iteration, outside the timer: refill the source tensor, synchronize the device (so the tensor is "ready", as the native tools' ready scope), barrier (collectives).
Timed: the call(s). Then, for CUDA and MPS tensors, a measurement-only device synchronize whose duration is recorded separately (dev_sync_us), so
"device_ready_us" = completion + that synchronization. Raw per-iteration samples (microseconds) are printed for every metric. Verification runs once after the loop.
"""
import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--backend", default="tbccl", choices=["tbccl", "gloo"])
p.add_argument("--devices", default="cpu,cpu", help="device of rank 0 and rank 1 (cpu | cuda | mps)")
p.add_argument("--op", required=True)
p.add_argument("--bytes", type=int, required=True)
p.add_argument("--source", type=int, default=0, help="the sending / root rank")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--timeout", type=float, default=120.0)
p.add_argument("--gloo-address", default=None, help="gloo: bind the transport to this IPv4 address (an interface name can resolve to an IPv6 link-local address)")
p.add_argument("--expect-addresses", default="", help="comma separated data-link addresses of the two ranks; the adapter rejects a run whose TCP peers are anywhere else")
a = p.parse_args()

OPS = ("all_reduce", "all_reduce_async", "broadcast", "all_gather", "p2p_ack", "isend_irecv_ack", "p2p_rtt")
if a.op not in OPS:
    raise SystemExit(f"unknown --op {a.op!r}")
if a.bytes < 4 or a.bytes % 4:
    raise SystemExit("--bytes must be a positive multiple of 4 (float32)")
if a.backend == "tbccl":
    import torch_tbccl  # noqa: F401

if a.backend == "gloo" and a.gloo_address:
    gopts = dist.ProcessGroupGloo._Options()
    gopts._devices = [dist.ProcessGroupGloo.create_device(hostname=a.gloo_address)]
    dist.init_process_group("cpu:gloo", timeout=datetime.timedelta(seconds=a.timeout), pg_options=gopts)  # "cpu:gloo", not "gloo": plain "gloo" also registers CUDA on a CUDA host, and the two ranks' store prefixes then differ (a silent deadlock)
else:
    dist.init_process_group(a.backend, timeout=datetime.timedelta(seconds=a.timeout))
rank, world = dist.get_rank(), dist.get_world_size()
if world != 2:
    raise SystemExit("this benchmark is a two-rank benchmark")
names = a.devices.split(",")
dev = C.make_device(names[rank])
peer = 1 - rank
n = a.bytes // 4
src, dst = a.source, 1 - a.source
ns = C.now_ns
us = lambda d: d / 1e3  # noqa: E731
samples = {}
flags = {"done_at_submit": 0, "callback_seen": 0}
callbacks = []  # (timestamp holder, t0): the completion callback runs on the backend's worker, possibly after wait() returned; read after the loop


def rec(name, value):
    samples.setdefault(name, []).append(us(value))


x = torch.zeros(n, dtype=torch.float32, device=dev)
ack = torch.zeros(1, dtype=torch.uint8)
outs = [torch.zeros(n, dtype=torch.float32, device=dev) for _ in range(2)]


def fill(seed):
    x.copy_(C.pattern(n, seed).to(dev))
    C.device_sync(dev)


def iteration(k, measure):
    if a.op in ("all_reduce", "all_reduce_async"):
        fill(rank + 1)
        dist.barrier()
        t0 = ns()
        if a.op == "all_reduce":
            dist.all_reduce(x)
            t1 = ns()
            C.device_sync(dev)
            t2 = ns()
            if measure:
                rec("completion_us", t1 - t0); rec("dev_sync_us", t2 - t1); rec("device_ready_us", t2 - t0)
        else:
            w = dist.all_reduce(x, async_op=True)
            t1 = ns()
            cb = []
            try:
                w.get_future().add_done_callback(lambda f, cb=cb: cb.append(ns()))
            except Exception:  # noqa: BLE001 - callbacks are optional
                pass
            if measure:
                callbacks.append((cb, t0))
            done = w.is_completed()
            w.wait()
            t2 = ns()
            C.device_sync(dev)
            t3 = ns()
            if measure:
                rec("submit_us", t1 - t0); rec("wait_us", t2 - t1); rec("lifetime_us", t2 - t0); rec("dev_sync_us", t3 - t2); rec("device_ready_us", t3 - t0)
                flags["done_at_submit"] += bool(done)
    elif a.op == "broadcast":
        if rank == src:
            fill(9)
        else:
            x.zero_(); C.device_sync(dev)
        dist.barrier()
        t0 = ns()
        dist.broadcast(x, src=src)
        t1 = ns()
        C.device_sync(dev)
        t2 = ns()
        if measure:
            rec("completion_us", t1 - t0); rec("dev_sync_us", t2 - t1); rec("device_ready_us", t2 - t0)
    elif a.op == "all_gather":
        fill(rank + 21)
        dist.barrier()
        t0 = ns()
        dist.all_gather(outs, x)
        t1 = ns()
        C.device_sync(dev)
        t2 = ns()
        if measure:
            rec("completion_us", t1 - t0); rec("dev_sync_us", t2 - t1); rec("device_ready_us", t2 - t0)
    elif a.op in ("p2p_ack", "isend_irecv_ack"):
        if rank == src:
            fill(33)
            t0 = ns()
            if a.op == "p2p_ack":
                dist.send(x, dst)
                t1 = ns()
            else:
                w = dist.isend(x, dst)
                t_sub = ns()
                w.wait()
                t1 = ns()
            dist.recv(ack, dst)
            t2 = ns()
            if measure:
                rec("completion_confirmed_us", t2 - t0)
                rec("send_us" if a.op == "p2p_ack" else "send_lifetime_us", t1 - t0)
                if a.op == "isend_irecv_ack":
                    rec("submit_us", t_sub - t0); rec("wait_us", t1 - t_sub)
        else:
            x.zero_(); C.device_sync(dev)
            t0 = ns()
            if a.op == "p2p_ack":
                dist.recv(x, src)
                t1 = ns()
            else:
                w = dist.irecv(x, src)
                t_sub = ns()
                w.wait()
                t1 = ns()
            C.device_sync(dev)
            t2 = ns()
            if measure:
                rec("recv_us", t1 - t0); rec("dev_sync_us", t2 - t1)
                if a.op == "isend_irecv_ack":
                    rec("submit_us", t_sub - t0); rec("wait_us", t1 - t_sub)
            dist.send(ack, src)
    else:  # p2p_rtt
        if rank == src:
            fill(41)
            t0 = ns()
            dist.send(x, dst)
            dist.recv(outs[0], dst)
            t1 = ns()
            C.device_sync(dev)
            t2 = ns()
            if measure:
                rec("round_trip_us", t1 - t0); rec("dev_sync_us", t2 - t1); rec("device_ready_us", t2 - t0)
        else:
            dist.recv(outs[0], src)
            dist.send(outs[0], src)


for k in range(a.warmup):
    iteration(k, False)
connections = C.own_tcp_connections()
dist.barrier()
cpu0, wall0 = time.process_time(), time.perf_counter()
for k in range(a.iterations):
    iteration(k, True)
cpu1, wall1 = time.process_time(), time.perf_counter()
time.sleep(0.2)
for holder, t0 in callbacks:
    if holder:
        flags["callback_seen"] += 1
        rec("callback_us", holder[0] - t0)

# ---- verification, outside the timed loop
ok = True
detail = ""
fill(rank + 1)
if a.op in ("all_reduce", "all_reduce_async"):
    w = dist.all_reduce(x, async_op=(a.op == "all_reduce_async"))
    if w is not None:
        w.wait()
    want = C.pattern(n, 1) + C.pattern(n, 2)
    ok = C.bits_equal(x, want)
elif a.op == "broadcast":
    if rank == src:
        fill(9)
    else:
        x.zero_(); C.device_sync(dev)
    dist.broadcast(x, src=src)
    ok = C.bits_equal(x, C.pattern(n, 9))
elif a.op == "all_gather":
    fill(rank + 21)
    dist.all_gather(outs, x)
    ok = C.bits_equal(outs[0], C.pattern(n, 21)) and C.bits_equal(outs[1], C.pattern(n, 22))
elif a.op in ("p2p_ack", "isend_irecv_ack"):
    if rank == src:
        fill(34)
        dist.send(x, dst)
        dist.recv(ack, dst)
    else:
        x.zero_(); C.device_sync(dev)
        dist.recv(x, src)
        ok = C.bits_equal(x, C.pattern(n, 34)) and x.device.type == dev.type
        dist.send(ack, src)
else:
    if rank == src:
        fill(42)
        dist.send(x, dst)
        dist.recv(outs[0], dst)
        ok = C.bits_equal(outs[0], C.pattern(n, 42))
    else:
        dist.recv(outs[0], src)
        dist.send(outs[0], src)
if dev.type != names[rank] and dev.type != torch.device(names[rank]).type:
    ok, detail = False, f"tensor ended on {dev}"
dist.barrier()
env = C.environment(a.backend)
C.emit("TBCCL_TORCH_BENCH", {"schema": "tbccl-torch-bench/1", "kind": "comm", "rank": rank, "op": a.op, "backend": a.backend, "bytes": a.bytes, "dtype": "float32", "shape": [n],
                             "device": str(dev), "peer_device": names[peer], "source": src, "warmup": a.warmup, "iterations": a.iterations, "samples_us": samples, "flags": flags,
                             "cpu_seconds": cpu1 - cpu0, "wall_seconds": wall1 - wall0, "verify_ok": bool(ok), "verify_detail": detail, "connections": connections,
                             "expect_addresses": a.expect_addresses, "environment": env, "pid": os.getpid()})
dist.destroy_process_group()
sys.exit(0 if ok else 3)
