#!/usr/bin/env python3
"""Two-rank loopback check for an installed torch-tbccl: run it with the interpreter of an environment the install helper filled, for example

    .venv/bin/python tools/clean_install_smoke.py

It starts both ranks itself on 127.0.0.1 (no torchrun, so nothing depends on how the machine resolves its own hostname), moves a few bytes over loopback
TCP on the CPU, loads no model and needs no GPU. It fails unless the group really uses the "tbccl" backend and the sum is right.
"""
import os
import socket
import subprocess
import sys

WORLD = 2


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rank_main():
    import torch
    import torch.distributed as dist

    import torch_tbccl  # noqa: F401  (registers the "tbccl" backend)

    dist.init_process_group("tbccl")
    rank = dist.get_rank()
    backend = dist.get_backend()
    x = torch.tensor([rank + 1.0, 10.0 * (rank + 1)])
    dist.all_reduce(x)
    ok = backend == "tbccl" and x.tolist() == [3.0, 30.0]
    sys.stdout.write(f"rank {rank}: backend={backend} result={x.tolist()} {'OK' if ok else 'FAIL'}\n")
    sys.stdout.flush()
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


def launcher():
    base = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port()), "WORLD_SIZE": str(WORLD), "TBCCL_LOCAL_ENDPOINT": "127.0.0.1:0", "OMP_NUM_THREADS": "1"}
    procs = [subprocess.Popen([sys.executable, __file__], env={**base, "RANK": str(r), "LOCAL_RANK": str(r)}) for r in range(WORLD)]
    codes = []
    for p in procs:
        try:
            codes.append(p.wait(timeout=120))
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            sys.exit("FAIL: the ranks did not finish within 120 s")
    return 0 if codes == [0] * WORLD else 1


if __name__ == "__main__":
    sys.exit(rank_main() if "RANK" in os.environ else launcher())
