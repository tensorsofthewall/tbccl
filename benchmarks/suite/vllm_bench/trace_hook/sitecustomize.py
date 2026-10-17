"""Diagnostic-only timing hook for the CUDA / CPU stage of a vLLM pipeline (the stage vllm-tbccl itself does not time). Active only when TBCCL_B5_TRACE_DIR is set.

It wraps three methods when vLLM imports them, without modifying any package source, and appends same-process ``time.monotonic_ns()`` records to
``$TBCCL_B5_TRACE_DIR/hook.<pid>.jsonl`` from a daemon thread (no I/O on the request path):

  step      Worker.execute_model entry and exit, the number of scheduled tokens and requests of the step (a step with one token per request is a decode step)
  isend     GroupCoordinator.isend_tensor_dict submission (the call returning is NOT the transfer completing), bytes of the tensors handed over
  irecv     GroupCoordinator.irecv_tensor_dict submission

Timestamps of different hosts are never compared; every figure derived from this file is a duration measured inside one process. The wrappers add a few microseconds of
Python per call, so traced runs belong to the diagnostic campaign and are never mixed into the uninstrumented performance runs.
"""
import json
import os
import sys
import threading
import time

_dir = os.environ.get("TBCCL_B5_TRACE_DIR")
if _dir:
    try:
        os.makedirs(_dir, exist_ok=True)
        _events = []
        _lock = threading.Lock()
        _path = os.path.join(_dir, f"hook.{os.getpid()}.jsonl")

        def _flush():
            with _lock:
                batch = list(_events)
                _events.clear()
            if batch:
                with open(_path, "a") as fh:
                    fh.write("".join(json.dumps(e) + "\n" for e in batch))

        def _loop():
            while True:
                time.sleep(2.0)
                _flush()

        threading.Thread(target=_loop, daemon=True, name="b5-trace-hook").start()
        import atexit
        atexit.register(_flush)

        def _rec(kind, t0, t1, **extra):
            with _lock:
                _events.append({"kind": kind, "entry_ns": t0, "done_ns": t1, "pid": os.getpid(), **extra})

        def _nbytes(tensors):
            total = 0
            try:
                for v in (tensors or {}).values():
                    total += v.numel() * v.element_size() if hasattr(v, "numel") else 0
            except Exception:
                pass
            return total

        def _wrap_worker(mod):
            cls = mod.Worker
            if getattr(cls, "_b5_hooked", False):
                return
            orig = cls.execute_model

            def execute_model(self, scheduler_output, *a, **k):
                t0 = time.monotonic_ns()
                try:
                    return orig(self, scheduler_output, *a, **k)
                finally:
                    try:
                        _rec("step", t0, time.monotonic_ns(), tokens=int(scheduler_output.total_num_scheduled_tokens), requests=len(scheduler_output.num_scheduled_tokens))
                    except Exception:
                        _rec("step", t0, time.monotonic_ns())

            cls.execute_model = execute_model
            cls._b5_hooked = True

        def _wrap_group(mod):
            cls = mod.GroupCoordinator
            if getattr(cls, "_b5_hooked", False):
                return
            for name, kind in (("isend_tensor_dict", "isend"), ("irecv_tensor_dict", "irecv")):
                if not hasattr(cls, name):
                    continue

                def make(orig=getattr(cls, name), kind=kind):
                    def wrapped(self, tensor_dict=None, *a, **k):
                        t0 = time.monotonic_ns()
                        try:
                            return orig(self, tensor_dict, *a, **k)
                        finally:
                            _rec(kind, t0, time.monotonic_ns(), bytes=_nbytes(tensor_dict) if kind == "isend" else None)
                    return wrapped

                setattr(cls, name, make())
            cls._b5_hooked = True

        import importlib.abc
        import importlib.util

        class _Finder(importlib.abc.MetaPathFinder):
            targets = {"vllm.v1.worker.gpu_worker": _wrap_worker, "vllm.distributed.parallel_state": _wrap_group}

            def find_spec(self, name, path, target=None):
                if name not in self.targets:
                    return None
                sys.meta_path.remove(self)
                try:
                    spec = importlib.util.find_spec(name)
                finally:
                    sys.meta_path.insert(0, self)
                if spec is None or spec.loader is None:
                    return None
                loader, wrap = spec.loader, self.targets[name]
                orig_exec = loader.exec_module

                def exec_module(mod, orig_exec=orig_exec, wrap=wrap):
                    orig_exec(mod)
                    try:
                        wrap(mod)
                    except Exception as error:
                        sys.stderr.write(f"b5 trace hook not installed for {mod.__name__}: {error}\n")

                loader.exec_module = exec_module
                return spec

        sys.meta_path.insert(0, _Finder())
    except Exception as error:  # a diagnostic must never break the process it observes
        sys.stderr.write(f"b5 trace hook disabled: {error}\n")
