"""Passive per-host sampler for the Exo benchmarks: the vLLM sampler (``vllm_bench/sampler.py``: process tree CPU and memory, link counters, TCP census, GPU and thermal reads) with Exo's process
selection. It is read-only; nothing is configured and no temperature ever stops a run.

    python sampler.py --namespace NS --role NAME --interval 1.0 --link-interface IFACE --expect-addresses A,B [--gpu] [--thermal] [--liveness]

An Exo process is any process of the current user whose environment carries ``EXO_ZENOH_NAMESPACE=NS``: the node and the runner processes it spawns inherit the unique namespace the suite
gave the node, so no process is ever matched by a name pattern. Prints ``TBCCL_EXO_SAMPLE <json>`` lines. With ``--liveness`` the sampler exits with status 7 when the Exo node process it saw
disappears (runners legitimately exit when an instance is deleted, so only the node counts; a failed runner shows up in the cluster state the client reads).
"""
import importlib.util
import os
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / "vllm_bench" / "sampler.py"
_spec = importlib.util.spec_from_file_location("_tbccl_base_sampler", _path)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

NAMESPACE = {"value": None}


def exo_processes(psutil):
    found = {}
    me = os.getpid()
    for p in psutil.process_iter(["pid"]):
        if p.pid == me:
            continue
        try:
            if p.environ().get("EXO_ZENOH_NAMESPACE") == NAMESPACE["value"]:
                found[p.pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


def role_of(p):
    try:
        text = " ".join(p.cmdline())
    except Exception:  # noqa: BLE001
        return "unknown"
    return "runner" if "multiprocessing" in text else "exo-node"


def lost_nodes(seen, alive):
    lost = {pid: r for pid, r in seen.items() if pid not in alive and r == "exo-node"}
    return lost or None


def main():
    args = sys.argv[1:]
    if "--namespace" not in args:
        raise SystemExit("--namespace is required")
    i = args.index("--namespace")
    NAMESPACE["value"] = args[i + 1]
    del sys.argv[i + 1], sys.argv[i]
    base.PREFIX = "TBCCL_EXO_SAMPLE"
    base.engine_processes = exo_processes
    base.role_of = role_of
    base.lost_engines = lost_nodes
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
