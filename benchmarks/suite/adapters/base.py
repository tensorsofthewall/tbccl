"""Adapter interface: what the runner needs to know about one native benchmark.

An adapter never implements communication or timing. It describes the executable (operations, algorithms, memory backends, rank counts, what it can verify),
builds the command line for every rank and parses the executable's existing output. Adding a benchmark means adding one subclass and registering it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ParseError(Exception):
    """The executable ran but its output did not have the expected shape; the measurement is invalid."""


@dataclass
class Capabilities:
    name: str
    category: str
    description: str
    executable: str
    operations: tuple
    algorithms: tuple = ()
    memory_backends: tuple = ("host",)
    min_ranks: int = 2
    max_ranks: int | None = 2
    datatypes: tuple = ()
    integrated: bool = True
    source_selectable: bool = False  # the data source rank can be chosen independently of the rank orientation
    executable_in_build: bool = True  # False when the command is found on PATH (an interpreter, a script) rather than in a build directory
    supports_warmup: bool = True
    supports_iterations: bool = True
    verification: dict = field(default_factory=dict)
    source_files: tuple = ()
    notes: str = ""


@dataclass
class Point:
    """One workload point before repetition: everything needed to build commands and describe the result."""
    workload: str
    benchmark: str
    operation: str
    algorithm: str | None
    payload_bytes: int
    datatype: str | None
    reduce_op: str | None
    rank_count: int
    ranks: list
    options: dict
    warmup: int
    iterations: int
    repetitions: int
    timeout_s: float
    verification_policy: str = "preferred"
    skip_reason: str | None = None
    skip_state: str = "skipped"
    orientation: str = "forward"
    source_rank: int | None = None
    estimated_bytes: int | None = None

    @property
    def point_id(self) -> str:
        parts = [self.benchmark, self.operation]
        if self.algorithm:
            parts.append(self.algorithm)
        if self.datatype:
            parts.append(self.datatype)
        if any(b != "host" for b in self.backends):
            parts.append("+".join(self.backends))
        parts += [f"{self.rank_count}r", f"{self.payload_bytes}B"]
        if self.source_rank is not None:
            parts.append(f"src{self.source_rank}")
        if self.orientation != "forward":
            parts.append(self.orientation)
        for key in ("backend", "tag"):  # configurations that differ only here must not share a raw-output directory
            if self.options.get(key):
                parts.append(str(self.options[key]))
        return "-".join(p.replace("_", "-") for p in parts)

    @property
    def backends(self) -> list:
        return [r["memory_backend"] for r in self.ranks]


@dataclass
class Invocation:
    rank: int
    role: str
    host_id: str
    argv: list
    env: dict = field(default_factory=dict)
    persistent: bool = False
    wait_for_port: int | None = None
    wait_host: str | None = None
    output_file: str | None = None
    listens_on: int | None = None
    stop_first: bool = False  # passive helpers (samplers) are stopped before the processes they watch, so they never report the orderly shutdown as a loss


@dataclass
class Group:
    """The processes that together make one measurement; any failure invalidates the whole group."""
    invocations: list
    ports: list
    notes: list = field(default_factory=list)


@dataclass
class Outcome:
    exit_codes: dict
    terminated_by_runner: set
    timed_out: bool
    failed_rank: int | None
    stdout: dict
    stderr: dict
    wall_s: float
    reason: str | None = None
    rank_reports: dict = field(default_factory=dict)
    cleanup_unconfirmed: list = field(default_factory=list)


@dataclass
class Parsed:
    measurements: dict
    verification: dict
    algorithm_resolved: str | None = None
    selection_reason: str | None = None
    configuration_extra: dict = field(default_factory=dict)


def timing(metric, statistics, provenance, primary, *, percentile_method=None, sample_count=None, samples=None, per_rank=None):
    return {"metric": metric, "unit": "us", "primary_statistic": primary, "statistics": statistics,
            "statistics_provenance": {k: provenance for k in statistics} if isinstance(provenance, str) else provenance,
            "percentile_method": percentile_method, "sample_count": sample_count, "samples": samples, "per_rank": per_rank}


def memory_domain_label(backends) -> str:
    """cpu-cpu, cuda-cpu, cuda-metal, ...: the device kinds taking part, sorted so the label does not depend on rank order."""
    kinds = []
    for backend in backends:
        kinds.append("cpu" if backend == "host" else backend.split("-")[0])
    return "-".join(sorted(kinds))


class BenchmarkAdapter:
    caps: Capabilities

    def check_point(self, point: Point, ctx) -> str | None:
        """Reason this point cannot run (a property of the point, not a configuration error), or None."""
        return None

    def validate_workload(self, workload: dict, operations: list) -> list:
        """Configuration-level errors specific to this benchmark (missing options, bad option values)."""
        return []

    def estimate_bytes(self, point: Point) -> int | None:
        """An upper-ish estimate of the payload bytes one repetition moves over the link (both directions summed), or None when unknown. Used against the volume limits."""
        return None

    def data_direction(self, point: Point) -> dict:
        """Which way the payload flows: {'mode': 'directional', 'source_rank': r} or {'mode': 'bidirectional'}. Independent of the rank orientation."""
        return {"mode": "bidirectional", "source_rank": None}

    def describe(self, point: Point) -> dict:
        """Static facts about the measurement: payload_semantics, timing_method, completion_semantics, backend."""
        raise NotImplementedError

    def plan(self, point: Point, ctx, ports: list, rep_dir) -> Group:
        raise NotImplementedError

    def parse(self, point: Point, group: Group, outcome: Outcome) -> Parsed:
        raise NotImplementedError

    def verification_support(self, point: Point) -> str:
        return self.caps.verification.get(point.operation, "not_supported")
