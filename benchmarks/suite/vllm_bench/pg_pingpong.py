"""Pipeline-boundary ping-pong through vllm-tbccl's own transport, at vLLM's GroupCoordinator level (no model, no scheduler).

    VLLM_TBCCL_ENABLE=1 VLLM_TBCCL_PLATFORM=cuda|cpu TBCCL_LOCAL_ENDPOINT=<link ip>:0 python pg_pingpong.py --rank R --init tcp://HOST:PORT --tokens 1,4,12,128,512,2000

Rank 0 sends a tensor dict {hidden_states, residual} (bf16, [tokens, hidden]) with vLLM's own ``GroupCoordinator.send_tensor_dict``; rank 1 receives it and sends the same dict back; rank 0
times the round trip with its own monotonic clock. That is the exact exchange vLLM performs between two pipeline stages (a pickled metadata message on the control group, then the tensors on
the device group), without any model compute. One line ``TBCCL_VLLM_PG <json>`` per rank is printed; the payload is verified on the receiving side on every size (first iteration).
Hosts' clocks are never compared; a one-way time is not claimed and the round trip is never halved: the round trip of the whole exchange is the measurement.
"""
import argparse
import json
import os
import time

import torch

from vllm_tbccl._backend import op_stats, register_backend, reset_op_stats

register_backend()
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import parallel_state as ps  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--rank", type=int, required=True)
p.add_argument("--hidden", type=int, default=1024)
p.add_argument("--tokens", default="1,4,12,128,512,2000")
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--iterations", type=int, default=300)
a = p.parse_args()
rank = a.rank


def pattern(tokens, k):
    idx = torch.arange(tokens * a.hidden, dtype=torch.int64)
    return (((idx * 31 + k * 101) % 997).to(torch.float32) / 7.0).reshape(tokens, a.hidden).to(torch.bfloat16)


def pct(v, f):
    v = sorted(v)
    pos = f * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


with set_current_vllm_config(VllmConfig()):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=a.init, local_rank=0, backend=current_platform.dist_backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp = ps.get_pp_group()
    dev = pp.device
    reset_op_stats()
    out = {"rank": rank, "platform": type(current_platform).__name__, "device": str(dev), "communicator": type(pp.device_communicator).__name__, "torch": torch.__version__,
           "dist_backend": current_platform.dist_backend, "hidden": a.hidden, "sizes": []}
    for tokens in (int(t) for t in a.tokens.split(",")):
        ref = {"hidden_states": pattern(tokens, 0), "residual": pattern(tokens, 1)}
        payload = {k: v.to(dev) for k, v in ref.items()}
        samples, verified = [], None
        for i in range(a.warmup + a.iterations):
            if rank == 0:
                t0 = time.perf_counter_ns()
                pp.send_tensor_dict(payload)
                got = pp.recv_tensor_dict()
                t1 = time.perf_counter_ns()
                if i >= a.warmup:
                    samples.append((t1 - t0) / 1e3)
            else:
                got = pp.recv_tensor_dict()
                if verified is None:
                    verified = all(torch.equal(got[k].cpu(), ref[k]) for k in ref)
                pp.send_tensor_dict({k: v.to(dev) for k, v in got.items()})
        if rank == 0:
            got_ok = all(torch.equal(got[k].cpu(), ref[k]) for k in ref)
            out["sizes"].append({"tokens": tokens, "bytes_per_direction": 2 * tokens * a.hidden * 2, "rtt_us": {"p50": pct(samples, .5), "p95": pct(samples, .95), "min": min(samples), "max": max(samples),
                                                                                                                   "mean": sum(samples) / len(samples), "n": len(samples)},
                                 "samples_us": samples, "echo_verified": got_ok})
        else:
            out["sizes"].append({"tokens": tokens, "peer_received_verified": bool(verified)})
    stats = op_stats()
    out["tbccl_ops"] = {k: v["count"] for k, v in stats.items()}
    out["tbccl_bytes"] = sum(v["bytes"] for v in stats.values())
    out["connections"] = None
    try:
        import psutil
        conns = psutil.Process().net_connections(kind="tcp")
        out["connections"] = sorted({f"{c.laddr.ip}>{c.raddr.ip}" for c in conns if c.raddr and c.status == "ESTABLISHED"})
    except Exception:
        pass
    print("TBCCL_VLLM_PG " + json.dumps(out), flush=True)
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
