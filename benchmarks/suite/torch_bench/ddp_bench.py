"""DistributedDataParallel training benchmark for the suite (one process per rank, world size 2, float32, synthetic deterministic data).

Model: `--layers` x [Linear(width, width) + tanh] followed by an MSE loss against a fixed random target; SGD, no momentum. Inputs are generated
once on the CPU from fixed seeds and moved to the rank's device before any timing. Parameters start from seed 1234 + rank (DDP's constructor broadcast makes
rank 0's authoritative).

Phases, each measured on this rank's own monotonic clock:
  correctness   --correctness-steps steps compared with a CPU reference that never touches the backend (grads, parameters, finite values, cross-rank agreement); not timed
  free          --steps steps with ONE device synchronization at the end of each step (the realistic step time; basis of steps/s and samples/s)
  phased        --steps steps with a measurement synchronization after forward, backward and the optimizer step (forward / backward / optimizer times)
  nosync        --steps steps under DistributedDataParallel.no_sync(): forward + backward + optimizer with NO communication: this rank's compute-only time
  profile       (only with --profile and TORCH_TBCCL_TRACE=1) --profile-steps steps with gradient-ready hooks and the adapter's collective trace; reported as a summary
Exposed communication per step = backward(phased) - backward(nosync) is an inference from two loops of the same process, not a measured interval.
Nothing here subtracts clocks of different hosts. Raw per-step samples (microseconds) are printed.
"""
import argparse
import datetime
import math
import os
import resource
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--backend", default="tbccl", choices=["tbccl", "gloo"])
p.add_argument("--devices", default="cpu,cpu")
p.add_argument("--layers", type=int, default=4)
p.add_argument("--width", type=int, default=512)
p.add_argument("--batch", type=int, default=64, help="per-rank batch size")
p.add_argument("--bucket-cap-mb", type=float, default=25.0)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--correctness-steps", type=int, default=3)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--profile", action="store_true", help="trace phase (needs TORCH_TBCCL_TRACE=1) and GPU utilization sampling; kept out of the timed phases")
p.add_argument("--profile-steps", type=int, default=20)
p.add_argument("--timeout", type=float, default=300.0)
p.add_argument("--gloo-address", default=None, help="gloo: bind the transport to this IPv4 address (an interface name can resolve to an IPv6 link-local address)")
p.add_argument("--expect-addresses", default="")
a = p.parse_args()

if a.backend == "tbccl":
    import torch_tbccl
if a.backend == "gloo" and a.gloo_address:
    gopts = dist.ProcessGroupGloo._Options()
    gopts._devices = [dist.ProcessGroupGloo.create_device(hostname=a.gloo_address)]
    dist.init_process_group("cpu:gloo", timeout=datetime.timedelta(seconds=a.timeout), pg_options=gopts)  # "cpu:gloo", not "gloo": plain "gloo" also registers CUDA on a CUDA host, and the two ranks' store prefixes then differ (a silent deadlock)
else:
    dist.init_process_group(a.backend, timeout=datetime.timedelta(seconds=a.timeout))
rank, world = dist.get_rank(), dist.get_world_size()
if world != 2:
    raise SystemExit("two ranks only")
names = a.devices.split(",")
dev = C.make_device(names[rank])
ns = C.now_ns
sync = lambda: C.device_sync(dev)  # noqa: E731
W, L, B = a.width, a.layers, a.batch
NB = 8


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(W, W) for _ in range(L))

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


def build(seed):
    torch.manual_seed(seed)
    return DeepMLP()


def batch_cpu(k, r):
    g = torch.Generator().manual_seed(5000 + k * 10 + r)
    return torch.randn(B, W, generator=g), torch.randn(B, W, generator=g)


model = build(1234 + rank).to(dev)
ddp = DDP(model, device_ids=[0] if dev.type == "cuda" else None, bucket_cap_mb=a.bucket_cap_mb)
opt = torch.optim.SGD(ddp.parameters(), lr=a.lr)
nparams = sum(q.numel() for q in model.parameters())
data = [tuple(t.to(dev) for t in batch_cpu(k, rank)) for k in range(NB)]
sync()
mse = nn.functional.mse_loss
warnings = []


def flat_cpu(m, grads=False):
    return torch.cat([(q.grad if grads else q).detach().cpu().reshape(-1) for q in m.parameters()])


# ---------------------------------------------------------------- correctness (not timed)
checks = {"finite": True, "max_grad_rel_err": 0.0, "max_param_rel_err": 0.0, "constructor_sync_ok": None, "cross_rank_max_abs_diff": None}
ref = build(1234)  # rank 0's initial weights on the CPU
checks["constructor_sync_ok"] = bool(torch.equal(flat_cpu(model), flat_cpu(ref)))
ref_opt = torch.optim.SGD(ref.parameters(), lr=a.lr)
for k in range(a.correctness_steps):
    x, y = data[k % NB]
    opt.zero_grad()
    loss = mse(ddp(x), y)
    loss.backward()
    sync()
    got = flat_cpu(model, grads=True)
    checks["finite"] &= bool(torch.isfinite(loss).item() and torch.isfinite(got).all().item())
    opt.step()
    sync()
    gs = []
    for r in (0, 1):
        ref.zero_grad()
        xr, yr = batch_cpu(k % NB, r)
        mse(ref(xr), yr).backward()
        gs.append(flat_cpu(ref, grads=True))
    want = (gs[0] + gs[1]) / 2
    off = 0
    for q in ref.parameters():
        q.grad = want[off:off + q.numel()].reshape(q.shape).clone()
        off += q.numel()
    ref_opt.step()
    checks["max_grad_rel_err"] = max(checks["max_grad_rel_err"], float((got - want).abs().max() / (want.abs().max() + 1e-12)))
    checks["max_param_rel_err"] = max(checks["max_param_rel_err"], float((flat_cpu(model) - flat_cpu(ref)).abs().max() / (flat_cpu(ref).abs().max() + 1e-12)))
mine = torch.cat([q.detach().reshape(-1) for q in model.parameters()])
gathered = [torch.zeros_like(mine) for _ in range(2)]
dist.all_gather(gathered, mine)
checks["cross_rank_max_abs_diff"] = float((gathered[0] - gathered[1]).abs().max().cpu())
same_type = names[0] == names[1]
tol_x = 1e-5 if not same_type else 1e-6
correct = bool(checks["finite"] and checks["constructor_sync_ok"] and checks["max_grad_rel_err"] <= 1e-3 and checks["max_param_rel_err"] <= 1e-4 and checks["cross_rank_max_abs_diff"] <= tol_x)
checks.update({"tolerances": {"grad_rel": 1e-3, "param_rel": 1e-4, "cross_rank_abs": tol_x}, "passed": correct})
if not correct:
    C.emit("TBCCL_TORCH_DDP", {"schema": "tbccl-torch-bench/1", "kind": "ddp", "rank": rank, "correctness": checks, "failed": "correctness"})
    dist.destroy_process_group()
    sys.exit(3)


# ---------------------------------------------------------------- timed phases
def step_free(k):
    x, y = data[k % NB]
    opt.zero_grad(set_to_none=True)
    loss = mse(ddp(x), y)
    loss.backward()
    opt.step()
    sync()


def step_phased(k):
    x, y = data[k % NB]
    t0 = ns()
    opt.zero_grad(set_to_none=True)
    loss = mse(ddp(x), y)
    sync(); t1 = ns()
    loss.backward()
    sync(); t2 = ns()
    opt.step()
    sync(); t3 = ns()
    return t1 - t0, t2 - t1, t3 - t2, t3 - t0


def step_nosync(k):
    x, y = data[k % NB]
    with ddp.no_sync():
        t0 = ns()
        opt.zero_grad(set_to_none=True)
        loss = mse(ddp(x), y)
        sync(); t1 = ns()
        loss.backward()
        sync(); t2 = ns()
        opt.step()
        sync(); t3 = ns()
    return t1 - t0, t2 - t1, t3 - t2, t3 - t0


us = lambda d: d / 1e3  # noqa: E731
S = {}
for k in range(a.warmup):
    step_free(k)
dist.barrier()
connections = C.own_tcp_connections()
cpu0, wall0 = time.process_time(), time.perf_counter()
for k in range(a.steps):
    dist.barrier()
    t0 = ns()
    step_free(k)
    S.setdefault("step_us", []).append(us(ns() - t0))
cpu1, wall1 = time.process_time(), time.perf_counter()

for k in range(a.warmup):
    step_phased(k)
dist.barrier()
for k in range(a.steps):
    dist.barrier()
    f, b, o, t = step_phased(k)
    for name, v in (("forward_us", f), ("backward_us", b), ("optimizer_us", o), ("phased_step_us", t)):
        S.setdefault(name, []).append(us(v))

for k in range(a.warmup):
    step_nosync(k)
for k in range(a.steps):
    f, b, o, t = step_nosync(k)
    for name, v in (("nosync_forward_us", f), ("nosync_backward_us", b), ("nosync_optimizer_us", o), ("nosync_step_us", t)):
        S.setdefault(name, []).append(us(v))

# ---------------------------------------------------------------- profile (separate, optional)
profile = None
if a.profile:
    gpu_samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,clocks.sm,clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                gpu_samples.append([float(v) if v.replace(".", "", 1).isdigit() else v for v in (x.strip() for x in out.split(","))])
            except (OSError, subprocess.TimeoutExpired, ValueError):
                break
            stop.wait(0.5)

    if dev.type == "cuda":
        threading.Thread(target=sampler, daemon=True).start()
    trace_on = a.backend == "tbccl" and torch_tbccl.trace_enabled()
    if trace_on:
        torch_tbccl.trace_reset()
        now_t = torch_tbccl.trace_now_ns
        ready, steps_rec, cur = [], [], [0]
        for i, q in enumerate(model.parameters()):
            q.register_post_accumulate_grad_hook(lambda _q, i=i: ready.append((cur[0], i, now_t())))
    for k in range(a.profile_steps):
        x, y = data[k % NB]
        dist.barrier()
        sync()
        cur[0] = k if trace_on else 0
        t_start = now_t() if trace_on else ns()
        opt.zero_grad(set_to_none=True)
        loss = mse(ddp(x), y)
        t_f = now_t() if trace_on else ns()
        loss.backward()
        t_b = now_t() if trace_on else ns()
        opt.step()
        sync()
        t_e = now_t() if trace_on else ns()
        if trace_on:
            steps_rec.append({"step": k, "start_ns": t_start, "fwd_end_ns": t_f, "bwd_end_ns": t_b, "end_ns": t_e})
    stop.set()
    profile = {"steps": a.profile_steps, "trace": trace_on}
    if gpu_samples:
        profile["gpu_utilization_percent"] = [s[0] for s in gpu_samples if isinstance(s[0], float)]
        profile["gpu_temperature_c"] = [s[1] for s in gpu_samples if isinstance(s[1], float)]
        profile["gpu_throttle_reasons"] = sorted({str(s[3]) for s in gpu_samples})
    if trace_on:
        ev = [e for e in torch_tbccl.trace_events() if e["op"] == "allreduce"]
        rows, per = [], []
        layouts = []
        for st in steps_rec:
            es = sorted((e for e in ev if st["start_ns"] <= e["entry_ns"] <= st["end_ns"]), key=lambda e: e["seq"])
            layouts.append(tuple(e["bytes"] for e in es))
        final = layouts[-1]
        first = next(i for i in range(len(layouts)) if all(l == final for l in layouts[i:]))
        first = max(first, 3)
        for st, es in list(zip(steps_rec, [sorted((e for e in ev if s["start_ns"] <= e["entry_ns"] <= s["end_ns"]), key=lambda e: e["seq"]) for s in steps_rec]))[first:]:
            gr = [r for r in ready if r[0] == st["step"]]
            last_ready = max(r[2] for r in gr) if gr else None
            ov, chained = [], 0
            for i, e in enumerate(es):
                nxt = es[i + 1]["entry_ns"] if i + 1 < len(es) else None
                rows.append({"submit_us": (e["return_ns"] - e["entry_ns"]) / 1e3, "total_us": (e["complete_ns"] - e["entry_ns"]) / 1e3, "bytes": e["bytes"],
                             "entry_rel_ms": (e["entry_ns"] - st["start_ns"]) / 1e6})
                if nxt is not None:
                    ov.append(max(0, min(nxt, e["complete_ns"]) - e["return_ns"]) / 1e3)
                    chained += e["return_ns"] < nxt < e["complete_ns"]
            per.append({"iter_ms": (st["end_ns"] - st["start_ns"]) / 1e6, "fwd_ms": (st["fwd_end_ns"] - st["start_ns"]) / 1e6, "bwd_ms": (st["bwd_end_ns"] - st["fwd_end_ns"]) / 1e6,
                        "tail_after_last_grad_ms": (st["bwd_end_ns"] - last_ready) / 1e6 if last_ready else None, "buckets": len(es), "overlap_lb_us": sum(ov), "chained": chained,
                        "first_grad_rel_ms": (min(r[2] for r in gr) - st["start_ns"]) / 1e6 if gr else None, "last_grad_rel_ms": (last_ready - st["start_ns"]) / 1e6 if last_ready else None})
        med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
        profile["trace_summary"] = {"measured_steps": len(per), "layout_bytes": list(final), "buckets_per_step": med([s["buckets"] for s in per]),
                                    "submit_us_median": med([r["submit_us"] for r in rows]), "total_work_us_median": med([r["total_us"] for r in rows]),
                                    "fwd_ms_median": med([s["fwd_ms"] for s in per]), "bwd_ms_median": med([s["bwd_ms"] for s in per]),
                                    "tail_after_last_grad_ms_median": med([s["tail_after_last_grad_ms"] for s in per if s["tail_after_last_grad_ms"] is not None]),
                                    "first_grad_ready_ms_median": med([s["first_grad_rel_ms"] for s in per if s["first_grad_rel_ms"] is not None]),
                                    "last_grad_ready_ms_median": med([s["last_grad_rel_ms"] for s in per if s["last_grad_rel_ms"] is not None]),
                                    "overlap_lower_bound_us_median": med([s["overlap_lb_us"] for s in per]), "chained_buckets_median": med([s["chained"] for s in per]),
                                    "bucket_entry_rel_ms_first": med([r["entry_rel_ms"] for r in rows][:1]), "note": "application-visible events from this process only; overlap figures are lower bounds"}

dist.barrier()
if dev.type == "cuda":
    peak = torch.cuda.max_memory_allocated()
elif dev.type == "mps":
    peak = torch.mps.driver_allocated_memory()
else:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
logging = {}
try:
    logging = {k: v for k, v in ddp._get_ddp_logging_data().items() if k in ("bucket_sizes", "num_buckets_reduced", "has_rebuilt_buckets", "rebuilt_bucket_sizes")}
except Exception as error:  # noqa: BLE001
    warnings.append(f"ddp logging data unavailable: {type(error).__name__}")
C.emit("TBCCL_TORCH_DDP", {"schema": "tbccl-torch-bench/1", "kind": "ddp", "rank": rank, "backend": a.backend, "device": str(dev), "peer_device": names[1 - rank], "model": f"deep-mlp-l{L}-w{W}",
                           "layers": L, "width": W, "batch": B, "global_batch": 2 * B, "bucket_cap_mb": a.bucket_cap_mb, "parameters": nparams, "grad_bytes_per_step": nparams * 4, "steps": a.steps, "warmup": a.warmup,
                           "lr": a.lr, "seed": 1234, "samples_us": S, "correctness": checks, "cpu_seconds": cpu1 - cpu0, "wall_seconds": wall1 - wall0, "peak_memory_bytes": int(peak),
                           "ddp_logging": logging, "profile": profile, "connections": connections, "expect_addresses": a.expect_addresses, "environment": C.environment(a.backend), "warnings": warnings, "pid": os.getpid()})
dist.destroy_process_group()
