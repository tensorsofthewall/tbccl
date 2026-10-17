"""Declarative benchmark configuration: loading, local overlays, validation and workload-matrix expansion.

A configuration separates physical hosts (`hosts`) from rank identity (`ranks`, a list mapping each rank to a host and a memory backend), so several ranks can
share a host and the topology is data, not code. Tracked example files carry no private addresses or paths; machine-specific values go into an ignored
local overlay merged on top.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .adapters import REGISTRY
from .adapters.base import Point

CONFIG_VERSION = "tbccl-bench-config/1"
SMOKE_SIZES = [4, 4096, 65536, 1048576]
STANDARD_SIZES = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
PRESETS = {"smoke": SMOKE_SIZES, "standard": STANDARD_SIZES}
VERIFICATION_POLICIES = ("required", "preferred", "off")
TOPOLOGY_KINDS = ("loopback", "direct-link", "switched", "unknown")
PLACEHOLDER = re.compile(r"<[A-Za-z0-9_ .:/-]+>")
LOOPBACK_PREFIXES = ("127.", "localhost", "::1")

TOP_KEYS = {"schema", "name", "description", "output_dir", "build_dirs", "limits", "defaults", "topology", "hosts", "ranks", "workloads"}
DEFAULT_KEYS = {"warmup", "iterations", "repetitions", "timeout_s", "verification", "startup_wait_s", "port_range", "lease_s", "term_grace_s", "failure_policy", "provenance",
                "verify_remote_cleanup", "health"}
WORKLOAD_KEYS = {"rank_hosts", "name", "benchmark", "operation", "operations", "algorithm", "algorithms", "sizes", "datatype", "reduce_op", "rank_counts", "warmup", "iterations",
                 "repetitions", "timeout_s", "verification", "options", "memory_backend", "orientations", "source_ranks"}
HOST_KEYS = {"os", "arch", "cpu", "gpu", "driver", "address", "interface", "mtu", "launcher", "executables_dir", "health_command", "health_snapshot", "python", "env", "interfaces",
             "executables", "build", "health_script", "expect", "management_address"}
BUILTIN_DEFAULTS = {"warmup": 10, "iterations": 50, "repetitions": 3, "timeout_s": 120.0, "verification": "preferred", "startup_wait_s": 10.0,
                    "port_range": [30000, 40000], "lease_s": 30.0, "term_grace_s": 3.0,
                    "failure_policy": {"on_failure": "continue", "max_failures": None}, "provenance": {"require_same_version": True, "require_stamped": False},
                    "verify_remote_cleanup": True, "health": {"strict": True, "snapshot_each_point": True, "max_load_fraction": 0.75}}


class ConfigError(Exception):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("invalid configuration:\n  - " + "\n  - ".join(self.errors))


@dataclass
class HostSpec:
    id: str
    address: str = "127.0.0.1"
    os: str | None = None
    arch: str | None = None
    cpu: str | None = None
    gpu: str | None = None
    driver: str | None = None
    interface: str | None = None
    mtu: int | None = None
    launcher: dict = field(default_factory=lambda: {"kind": "local"})
    executables_dir: str | None = None
    health_command: list | None = None
    health_snapshot: bool = False
    python: str | None = None
    env: dict = field(default_factory=dict)
    interfaces: dict = field(default_factory=dict)
    executables: dict = field(default_factory=dict)
    build: dict = field(default_factory=dict)
    health_script: str | None = None
    expect: dict = field(default_factory=dict)
    management_address: str | None = None

    @property
    def is_local(self) -> bool:
        return self.launcher.get("kind", "local") == "local"

    @property
    def is_loopback(self) -> bool:
        return self.address.startswith(LOOPBACK_PREFIXES)


@dataclass
class RunConfig:
    name: str
    description: str
    output_dir: str
    build_dirs: list
    limits: dict
    defaults: dict
    topology: dict
    hosts: dict
    ranks: list
    workloads: list
    source: str | None = None
    overlays: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def physical(self) -> bool:
        return bool(self.topology.get("physical"))


def parse_size(value) -> int:
    """'4', 4, '4B', '4K', '4KiB' (binary, as the native tools read K/M/G) or '4KB' (decimal)."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"invalid payload size {value!r}")
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        match = re.fullmatch(r"(\d+)\s*(?:(B)|([KMG])(iB|B)?)?", str(value).strip(), re.I)
        if not match:
            raise ValueError(f"invalid payload size {value!r}")
        digits, _, prefix, suffix = match.groups()
        number = int(digits)
        if prefix:
            power = {"K": 1, "M": 2, "G": 3}[prefix.upper()]
            number *= (1000 if suffix and suffix.lower() == "b" else 1024) ** power
    if number < 1:
        raise ValueError(f"payload size must be at least 1 byte, got {value!r}")
    return number


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_file(path: Path) -> dict:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as error:
        try:
            return json.loads(text)
        except ValueError:
            raise ConfigError([f"{path}: reading YAML needs PyYAML (pip install pyyaml); a JSON file works without it"]) from error
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError([f"{path}: the top level must be a mapping"])
    return data


def default_overlay(path: Path) -> Path | None:
    candidate = path.with_name(path.stem + ".local" + path.suffix)
    return candidate if candidate.exists() and ".local" not in path.stem else None


def load_config(path, overlays=None, auto_overlay=True) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError([f"configuration file not found: {path}"])
    raw = _read_file(path)
    used = []
    overlays = list(overlays or [])
    paths = [Path(p) for p in overlays]
    if auto_overlay and not overlays:
        auto = default_overlay(path)
        if auto:
            paths.append(auto)
    for overlay in paths:
        if not overlay.exists():
            raise ConfigError([f"overlay file not found: {overlay}"])
        raw = _deep_merge(raw, _read_file(overlay))
        used.append(str(overlay))
    config = build_config(raw)
    config.source = str(path)
    config.overlays = used
    return config


def build_config(raw: dict) -> RunConfig:
    errors = []
    for key in raw:
        if key not in TOP_KEYS:
            errors.append(f"unknown top-level key '{key}'")
    if raw.get("schema", CONFIG_VERSION) != CONFIG_VERSION:
        errors.append(f"unsupported configuration schema {raw.get('schema')!r}; expected {CONFIG_VERSION!r}")
    defaults = {**BUILTIN_DEFAULTS, **(raw.get("defaults") or {})}
    for nested in ("failure_policy", "provenance", "health"):
        if isinstance((raw.get("defaults") or {}).get(nested), dict):
            defaults[nested] = {**BUILTIN_DEFAULTS[nested], **raw["defaults"][nested]}
    for key in (raw.get("defaults") or {}):
        if key not in DEFAULT_KEYS:
            errors.append(f"defaults: unknown key '{key}'")
    _check_defaults(defaults, "defaults", errors)

    topology = {"kind": "loopback", "interconnect": "loopback", "transport": "tcp", "physical": False, **(raw.get("topology") or {})}
    if topology["kind"] not in TOPOLOGY_KINDS:
        errors.append(f"topology.kind must be one of {TOPOLOGY_KINDS}, got {topology['kind']!r}")

    hosts = {}
    for host_id, spec in (raw.get("hosts") or {}).items():
        spec = dict(spec or {})
        for key in spec:
            if key not in HOST_KEYS:
                errors.append(f"hosts.{host_id}: unknown key '{key}'")
        launcher = spec.pop("launcher", None) or {"kind": "local"}
        if launcher.get("kind") not in ("local", "prefix"):
            errors.append(f"hosts.{host_id}.launcher.kind must be 'local' or 'prefix'")
        if launcher.get("kind") == "prefix" and not launcher.get("argv"):
            errors.append(f"hosts.{host_id}.launcher: kind 'prefix' needs a non-empty argv list")
        if spec.get("env") is not None and not isinstance(spec["env"], dict):
            errors.append(f"hosts.{host_id}.env must be a mapping")
            spec["env"] = {}
        for key in ("env", "interfaces", "executables", "build", "expect"):
            spec[key] = dict(spec.get(key) or {})
        hosts[host_id] = HostSpec(id=host_id, launcher=launcher, **{k: v for k, v in spec.items() if k in HostSpec.__dataclass_fields__})
    if not hosts:
        errors.append("hosts: at least one host is required")

    ranks = []
    for index, entry in enumerate(raw.get("ranks") or []):
        if not isinstance(entry, dict) or "host" not in entry:
            errors.append(f"ranks[{index}]: needs a 'host'")
            continue
        rank = entry.get("rank", index)
        for key in entry:
            if key not in ("rank", "host", "memory_backend", "interface", "address"):
                errors.append(f"ranks[{index}]: unknown key '{key}'")
        rank_spec = {"rank": rank, "host": entry["host"], "memory_backend": entry.get("memory_backend", "host")}
        for key in ("interface", "address"):
            if entry.get(key):
                rank_spec[key] = entry[key]
        ranks.append(rank_spec)
        if entry["host"] not in hosts:
            errors.append(f"ranks[{index}]: unknown host '{entry['host']}'")
        elif entry.get("interface") and entry["interface"] not in hosts[entry["host"]].interfaces and entry["interface"] != hosts[entry["host"]].interface:
            errors.append(f"ranks[{index}]: host '{entry['host']}' has no interface '{entry['interface']}' (declared: {sorted(hosts[entry['host']].interfaces) or [hosts[entry['host']].interface]})")
    ranks.sort(key=lambda r: r["rank"])
    if [r["rank"] for r in ranks] != list(range(len(ranks))):
        errors.append("ranks: rank numbers must be 0..N-1 without gaps or repeats")
    if not ranks:
        errors.append("ranks: at least one rank is required")

    limits = {"max_processes": 8, "max_point_bytes": None, "max_total_bytes": None, "max_campaign_s": None, **(raw.get("limits") or {})}
    if not isinstance(limits["max_processes"], int) or limits["max_processes"] < 1:
        errors.append("limits.max_processes must be a positive integer")
    for key in ("max_point_bytes", "max_total_bytes", "max_campaign_s"):
        if limits[key] is not None and (not isinstance(limits[key], (int, float)) or isinstance(limits[key], bool) or limits[key] <= 0):
            errors.append(f"limits.{key} must be a positive number or omitted")

    non_loopback = [h.id for h in hosts.values() if not h.is_loopback]
    if non_loopback and not topology.get("physical"):
        errors.append(f"hosts {non_loopback} have non-loopback addresses; set topology.physical: true (physical runs are opt-in and need --allow-physical)")
    if topology.get("physical") and topology["kind"] == "loopback":
        errors.append("topology.physical is true but topology.kind is 'loopback'")
    if len({r["host"] for r in ranks}) > 1 and topology["kind"] == "loopback":
        errors.append("ranks on several hosts need a topology.kind other than 'loopback'")

    workloads = []
    for index, wl in enumerate(raw.get("workloads") or []):
        label = wl.get("name") or f"workloads[{index}]"
        for key in wl:
            if key not in WORKLOAD_KEYS:
                errors.append(f"{label}: unknown key '{key}'")
        workloads.append(wl)
    if not workloads:
        errors.append("workloads: at least one workload is required")

    config = RunConfig(name=raw.get("name", "unnamed"), description=raw.get("description", ""), output_dir=raw.get("output_dir", "results/bench-suite-local"),
                       build_dirs=list(raw.get("build_dirs") or ["build-release", "build-bench", "build"]), limits=limits, defaults=defaults, topology=topology,
                       hosts=hosts, ranks=ranks, workloads=workloads, raw=raw)
    if not errors:
        errors.extend(_validate_workloads(config))
    if errors:
        raise ConfigError(errors)
    return config


def _check_defaults(values: dict, where: str, errors: list) -> None:
    for key in ("warmup", "iterations", "repetitions"):
        if key in values and (not isinstance(values[key], int) or isinstance(values[key], bool) or values[key] < (0 if key == "warmup" else 1)):
            errors.append(f"{where}.{key} must be {'a non-negative' if key == 'warmup' else 'a positive'} integer, got {values[key]!r}")
    for key in ("timeout_s", "startup_wait_s"):
        if key in values and (not isinstance(values[key], (int, float)) or isinstance(values[key], bool) or values[key] <= 0):
            errors.append(f"{where}.{key} must be a positive number, got {values[key]!r}")
    pr = values.get("port_range")
    if pr is not None:
        placeholders = isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, str) and PLACEHOLDER.search(x) for x in pr)
        valid = isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in pr) and 1024 <= pr[0] < pr[1] <= 64999
        if not (valid or placeholders):
            errors.append(f"{where}.port_range must be [low, high] with 1024 <= low < high <= 64999, got {pr!r}")
    for key in ("lease_s", "term_grace_s"):
        if key in values and (not isinstance(values[key], (int, float)) or isinstance(values[key], bool) or values[key] <= 0):
            errors.append(f"{where}.{key} must be a positive number, got {values[key]!r}")
    policy = values.get("failure_policy")
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - {"on_failure", "max_failures"} or policy.get("on_failure", "continue") not in ("continue", "abort_run"):
            errors.append(f"{where}.failure_policy must be a mapping with on_failure: continue|abort_run and an optional max_failures")
    if "verification" in values and values["verification"] not in VERIFICATION_POLICIES:
        errors.append(f"{where}.verification must be one of {VERIFICATION_POLICIES}, got {values['verification']!r}")


def _as_list(wl, singular, plural):
    if plural in wl:
        value = wl[plural]
        return value if isinstance(value, list) else [value]
    if singular in wl:
        return [wl[singular]]
    return None


def resolve_sizes(value) -> list:
    if isinstance(value, str) and value in PRESETS:
        return list(PRESETS[value])
    items = value if isinstance(value, list) else [value]
    return [parse_size(v) for v in items]


ORIENTATIONS = ("forward", "reverse")


def effective_ranks(config, wl, count, orientation="forward"):
    """The first `count` ranks, with the workload's own memory_backend (one name for all ranks, one per rank, or a {host: backend} map) replacing the configured ones.

    'reverse' keeps rank numbers but swaps which host (and address, interface, backend) holds each rank: rank 0 moves to what was the last of the first `count`."""
    ranks = [dict(r) for r in config.ranks[:count]]
    if wl.get("rank_hosts"):  # explicit hosts for the first ranks (a single-node run on the second machine); the rank entries are those configured for the hosts
        by_host = {r["host"]: r for r in config.ranks}
        ranks = [{**by_host[h], "rank": i} for i, h in enumerate(wl["rank_hosts"][:count])]
    if orientation == "reverse":
        placements = [{k: v for k, v in r.items() if k != "rank"} for r in reversed(ranks)]
        ranks = [{"rank": r["rank"], **placement} for r, placement in zip(ranks, placements)]
    override = wl.get("memory_backend")
    if isinstance(override, dict):  # by host: the backend stays with its machine when the orientation swaps the ranks
        for rank in ranks:
            if rank["host"] in override:
                rank["memory_backend"] = override[rank["host"]]
    elif override is not None:
        names = override if isinstance(override, list) else [override] * count
        for rank, name in zip(ranks, names):
            rank["memory_backend"] = name
    return ranks


def _validate_workloads(config: RunConfig) -> list:
    errors = []
    for index, wl in enumerate(config.workloads):
        label = wl.get("name") or f"workloads[{index}]"
        name = wl.get("benchmark")
        if not name:
            errors.append(f"{label}: 'benchmark' is required")
            continue
        adapter = REGISTRY.get(name)
        if adapter is None:
            errors.append(f"{label}: unknown benchmark '{name}' (known: {', '.join(sorted(REGISTRY))})")
            continue
        caps = adapter.caps
        if not caps.integrated:
            errors.append(f"{label}: benchmark '{name}' is inventoried but not integrated yet")
            continue
        if "source_ranks" in wl:
            if not REGISTRY[wl["benchmark"]].caps.source_selectable if wl.get("benchmark") in REGISTRY else True:
                errors.append(f"{label}: {wl.get('benchmark')} has a fixed data direction; remove 'source_ranks'")
            elif not isinstance(wl["source_ranks"], list) or not wl["source_ranks"] or any(not isinstance(r, int) or isinstance(r, bool) or r < 0 for r in wl["source_ranks"]):
                errors.append(f"{label}: 'source_ranks' must be a non-empty list of rank numbers")
        for orientation in wl.get("orientations") or []:
            if orientation not in ORIENTATIONS:
                errors.append(f"{label}: orientation '{orientation}' must be one of {ORIENTATIONS}")
        ops = _as_list(wl, "operation", "operations")
        if not ops:
            errors.append(f"{label}: 'operation' is required (one of {', '.join(caps.operations)})")
            continue
        for op in ops:
            if op not in caps.operations:
                errors.append(f"{label}: operation '{op}' is not supported by {name} (supported: {', '.join(caps.operations)})")
        algorithms = _as_list(wl, "algorithm", "algorithms")
        if caps.algorithms:
            for algo in algorithms or []:
                if algo not in caps.algorithms:
                    errors.append(f"{label}: algorithm '{algo}' is not supported by {name} (supported: {', '.join(caps.algorithms)})")
        elif algorithms:
            errors.append(f"{label}: {name} has no selectable algorithm; remove 'algorithms'")
        if "sizes" not in wl:
            errors.append(f"{label}: 'sizes' is required (a list, or a preset: {', '.join(PRESETS)})")
        else:
            try:
                sizes = resolve_sizes(wl["sizes"])
                if not sizes:
                    errors.append(f"{label}: 'sizes' is empty")
            except ValueError as error:
                errors.append(f"{label}: {error}")
        if wl.get("rank_hosts") is not None:
            hosts_ok = isinstance(wl["rank_hosts"], list) and wl["rank_hosts"] and all(h in {r["host"] for r in config.ranks} for h in wl["rank_hosts"])
            if not hosts_ok:
                errors.append(f"{label}: rank_hosts must list hosts that have a configured rank")
        if isinstance(wl.get("memory_backend"), dict):
            errors.extend(f"{label}: memory_backend names host '{h}', which is not configured" for h in wl["memory_backend"] if h not in config.hosts)
        counts = wl.get("rank_counts") or [min(len(config.ranks), caps.max_ranks or len(config.ranks))]
        for count in counts:
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                errors.append(f"{label}: invalid rank count {count!r}")
            elif count > len(config.ranks):
                errors.append(f"{label}: rank_counts has {count} but only {len(config.ranks)} ranks are configured")
            elif count < caps.min_ranks or (caps.max_ranks is not None and count > caps.max_ranks):
                errors.append(f"{label}: {name} supports {caps.min_ranks}..{caps.max_ranks or 'many'} ranks, not {count}")
            elif count > config.limits["max_processes"]:
                errors.append(f"{label}: {count} ranks exceed limits.max_processes ({config.limits['max_processes']})")
            else:
                for rank in effective_ranks(config, wl, count, "forward"):
                    if rank["memory_backend"] not in caps.memory_backends:
                        errors.append(f"{label}: memory backend '{rank['memory_backend']}' (rank {rank['rank']}) is not supported by {name} "
                                      f"(supported: {', '.join(caps.memory_backends)})")
        if caps.datatypes and any(op in ("all_reduce", "reduce_scatter") for op in ops):
            dtype = wl.get("datatype", "float32")
            if dtype not in caps.datatypes:
                errors.append(f"{label}: datatype '{dtype}' is not supported by {name} (supported: {', '.join(caps.datatypes)})")
        elif "datatype" in wl and not caps.datatypes:
            errors.append(f"{label}: {name} has no datatype option")
        own = {k: wl[k] for k in ("warmup", "iterations", "repetitions", "timeout_s", "verification") if k in wl}
        _check_defaults(own, label, errors)
        errors.extend(f"{label}: {msg}" for msg in adapter.validate_workload(wl, ops))
    return errors


def expand_matrix(config: RunConfig, ctx=None) -> list:
    """The exact list of points the run would execute; points a benchmark cannot run are kept, with a reason, so a dry run shows them."""
    points = []
    for index, wl in enumerate(config.workloads):
        adapter = REGISTRY[wl["benchmark"]]
        caps = adapter.caps
        ops = _as_list(wl, "operation", "operations")
        algorithms = _as_list(wl, "algorithm", "algorithms") or [None]
        counts = wl.get("rank_counts") or [min(len(config.ranks), caps.max_ranks or len(config.ranks))]
        sizes = resolve_sizes(wl["sizes"])
        dtype_default = "float32" if caps.datatypes else None
        for op in ops:
            op_dtype = wl.get("datatype", dtype_default) if op in ("all_reduce", "reduce_scatter") else None
            sources = wl.get("source_ranks") if caps.source_selectable else [None]
            for count, orientation, source in [(c, o, src) for c in counts for o in (wl.get("orientations") or ["forward"]) for src in (sources or [None])]:
                for algo in algorithms:
                    for size in sizes:
                        merged = {**config.defaults, **{k: wl[k] for k in ("warmup", "iterations", "repetitions", "timeout_s", "verification") if k in wl}}
                        point = Point(workload=wl.get("name") or f"workload-{index}", benchmark=wl["benchmark"], operation=op, algorithm=algo, payload_bytes=size,
                                      datatype=op_dtype, reduce_op=(wl.get("reduce_op", "sum") if op in ("all_reduce", "reduce_scatter") else None),
                                      rank_count=count, ranks=effective_ranks(config, wl, count, orientation), options=dict(wl.get("options") or {}),
                                      warmup=merged["warmup"], iterations=merged["iterations"], repetitions=merged["repetitions"],
                                      timeout_s=float(merged["timeout_s"]), verification_policy=merged["verification"])
                        reason = adapter.check_point(point, ctx)
                        if reason is None and point.verification_policy == "required" and adapter.verification_support(point) == "not_supported":
                            reason = f"verification is required but {caps.name} cannot verify '{op}'"
                            point.skip_state = "unsupported"
                        point.orientation = orientation
                        point.source_rank = source
                        if reason is None and source is not None and source >= count:
                            reason = f"source rank {source} does not exist with {count} ranks"
                        point.estimated_bytes = adapter.estimate_bytes(point)
                        cap = config.limits.get("max_point_bytes")
                        if reason is None and cap and point.estimated_bytes and point.estimated_bytes > cap:
                            reason = f"estimated {point.estimated_bytes / 2**30:.2f} GiB per repetition exceeds limits.max_point_bytes ({cap / 2**30:.2f} GiB)"
                        point.skip_reason = reason
                        points.append(point)
    return points


def unresolved_placeholders(config: RunConfig) -> list:
    found = []

    def walk(value, where):
        if isinstance(value, str) and PLACEHOLDER.search(value):
            found.append(f"{where}: {value}")
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{where}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{where}[{i}]")

    walk(config.raw.get("hosts") or {}, "hosts")
    walk(config.raw.get("defaults") or {}, "defaults")
    return found


def port_range(config):
    """The configured range, or the built-in one while it still holds placeholders (a dry run); a real run refuses unresolved placeholders first."""
    pr = config.defaults.get("port_range")
    if isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, int) for x in pr):
        return tuple(pr)
    return tuple(BUILTIN_DEFAULTS["port_range"])
