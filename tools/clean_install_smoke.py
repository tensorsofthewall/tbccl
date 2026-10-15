#!/usr/bin/env python3
"""Two-rank loopback check for an installed torch-tbccl: run it under torchrun from an environment the install helper filled, for example

    TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 .venv/bin/torchrun --standalone --nproc-per-node 2 tools/clean_install_smoke.py

It moves a few bytes over loopback TCP on the CPU, loads no model and needs no GPU. It fails unless the group really uses the "tbccl" backend and the sum is right.
"""
import sys

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401  (registers the "tbccl" backend)

dist.init_process_group("tbccl")
rank = dist.get_rank()
backend = dist.get_backend()
x = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
dist.all_reduce(x)
ok = backend == "tbccl" and x.tolist() == [3.0, 30.0]
print(f"rank {rank}: backend={backend} result={x.tolist()} {'OK' if ok else 'FAIL'}", flush=True)
dist.barrier()
dist.destroy_process_group()
sys.exit(0 if ok else 1)
