"""Diagnostic-only capture of exo-tbccl's own trace inside Exo's runner processes (put this directory on PYTHONPATH of a traced node; nothing in Exo or exo-tbccl is modified).

Two things are missing from a normal run: exo-tbccl logs each pipeline operation on the stdlib logger ``exo_tbccl`` at INFO (``EXO_TBCCL_TRACE=1``), but Exo's runner configures no handler for it, so
the lines are dropped; and the adapter-level ``CopyStats`` (copies made by exo-tbccl itself, not staging done inside TBCCL's providers) live only in memory. With ``TBCCL_B6_TRACE_DIR`` set
this hook (1) attaches a file handler with wall-clock timestamps to ``exo_tbccl`` and (2) records every ``TbcclPipelineComm`` created in the process and rewrites a JSON snapshot of its
CopyStats every two seconds (a runner can be terminated without running exit handlers). Without the variable it does nothing. Traced runs are never mixed with the performance runs.
"""
import os
import sys

_DIR = os.environ.get("TBCCL_B6_TRACE_DIR")


def _install():
    import atexit
    import dataclasses
    import importlib.abc
    import importlib.util
    import json
    import logging
    import threading
    import time

    os.makedirs(_DIR, exist_ok=True)
    pid = os.getpid()
    handler = logging.FileHandler(os.path.join(_DIR, f"exo_tbccl-{pid}.log"))
    handler.setFormatter(logging.Formatter("%(created).6f %(message)s"))
    logger = logging.getLogger("exo_tbccl")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    comms = []
    lock = threading.Lock()

    def snapshot():
        with lock:
            items = list(comms)
        data = {"pid": pid, "written": time.time(), "comms": []}
        for c in items:
            try:
                data["comms"].append({"rank": getattr(c, "_rank", None), "stats": dataclasses.asdict(c.stats)})
            except Exception as e:  # noqa: BLE001
                data["comms"].append({"error": repr(e)})
        tmp = os.path.join(_DIR, f".copystats-{pid}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(_DIR, f"copystats-{pid}.json"))

    def loop():
        while True:
            time.sleep(2.0)
            if comms:
                snapshot()

    def wrap(module):
        cls = module.TbcclPipelineComm
        original = cls.__init__

        def init(self, *a, **k):
            original(self, *a, **k)
            with lock:
                comms.append(self)
        cls.__init__ = init
        threading.Thread(target=loop, daemon=True, name="b6-copystats").start()
        atexit.register(lambda: comms and snapshot())

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "exo_tbccl.group":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            real_exec = spec.loader.exec_module

            def exec_module(module):
                real_exec(module)
                wrap(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())


if _DIR:
    _install()
