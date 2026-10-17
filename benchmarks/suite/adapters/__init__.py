"""Registered benchmark adapters. Importing this package registers the integrated and the inventoried benchmarks."""
from __future__ import annotations

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, Outcome, ParseError, Parsed, Point  # noqa: F401
from .exo import SCENARIOS as EXO_SCENARIOS, ExoPipeline
from .core import CollectiveBench, SmallMessageLatency, TensorTransfer
from .device import AsyncOverlap, AsyncTransfer, CudaOverlap, HeteroAllReduce, TensorStaging
from .torch import TorchAsync, TorchCollectives, TorchDDP, TorchP2P
from .vllm import SCENARIOS as VLLM_SCENARIOS, VllmBackend, VllmPipeline
from .network import Iperf3, TcpPingPong, inventoried_only

REGISTRY = {a.caps.name: a for a in (TcpPingPong(), Iperf3(), SmallMessageLatency(), TensorTransfer(), CollectiveBench(),
                                  TensorStaging(), AsyncTransfer(), AsyncOverlap(), CudaOverlap(), HeteroAllReduce(),
                                  TorchCollectives(), TorchP2P(), TorchAsync(), TorchDDP(), *(ExoPipeline(n) for n in EXO_SCENARIOS), *((VllmBackend if n == "vllm_backend" else VllmPipeline)(n) for n in VLLM_SCENARIOS), *inventoried_only())}
