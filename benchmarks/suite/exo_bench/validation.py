"""Validation of an Exo cluster state document (``GET /state``) against what an experiment intends (standard library only).

Exo reports the instance kind as the single key of an instance object (``MlxTbcclInstance``, ``MlxRingInstance``, ...), its pipeline split in ``shardAssignments`` and the state of each
runner in ``runners``. Choosing ``instance_meta`` does not guarantee an otherwise identical experiment (Exo chooses the layer split), so the placement that actually happened is read
back from this document and compared; every function returns a list of problems (empty = consistent) and never raises on a malformed document.
"""
from __future__ import annotations

import re

KINDS = {"MlxTbccl": "MlxTbcclInstance", "MlxRing": "MlxRingInstance", "MlxJaccl": "MlxJacclInstance"}
READY = "RunnerReady"
FAILED_PREFIX = "RunnerFailed"


def nodes_of(state):
    """Node ids of the topology snapshot (``topology.nodes`` is a list of ids or of objects with ``nodeId``)."""
    out = []
    for n in (state.get("topology") or {}).get("nodes") or []:
        out.append(n.get("nodeId") if isinstance(n, dict) else n)
    return [n for n in out if n]


def edges(state):
    """Directed (source, sink) node pairs that have at least one connection in the topology snapshot."""
    out = set()
    for a, sinks in ((state.get("topology") or {}).get("connections") or {}).items():
        for b, conns in (sinks or {}).items():
            if conns:
                out.add((a, b))
    return out


def connected(state, node_count=2):
    """True when every node pair is connected in both directions (Exo refuses to place a pipeline otherwise: 'Needs 2 connected nodes')."""
    nodes = nodes_of(state)
    if len(nodes) != node_count:
        return False
    e = edges(state)
    return all((a, b) in e and (b, a) in e for a in nodes for b in nodes if a != b)


def instance_of(state, instance_id=None):
    """(instance_id, kind, body) of the instance with this id, or of the only instance when none is named. Raises LookupError when it is not unambiguous."""
    instances = state.get("instances") or {}
    if instance_id is None:
        if len(instances) != 1:
            raise LookupError(f"expected exactly one instance, found {len(instances)}")
        instance_id = next(iter(instances))
    inst = instances.get(instance_id)
    if not isinstance(inst, dict) or len(inst) != 1:
        raise LookupError(f"instance {instance_id!r} missing or malformed")
    kind = next(iter(inst))
    return instance_id, kind, inst[kind]


def layout(state, instance_id=None):
    """The placement that happened: kind, world size and one record per pipeline stage ordered by device rank."""
    iid, kind, body = instance_of(state, instance_id)
    assign = body.get("shardAssignments") or {}
    runner_to_node = {r: n for n, r in (assign.get("nodeToRunner") or {}).items()}
    stages = []
    for runner, shard in (assign.get("runnerToShard") or {}).items():
        meta = next(iter(shard.values())) if isinstance(shard, dict) and len(shard) == 1 else shard
        stages.append({"runner": runner, "node": runner_to_node.get(runner), "device_rank": meta.get("deviceRank"), "start_layer": meta.get("startLayer"),
                       "end_layer": meta.get("endLayer"), "n_layers": meta.get("endLayer", 0) - meta.get("startLayer", 0), "total_layers": meta.get("nLayers"),
                       "world_size": meta.get("worldSize"), "model_id": ((meta.get("modelCard") or {}).get("modelId"))})
    stages.sort(key=lambda s: (s["device_rank"] is None, s["device_rank"]))
    hosts = {}
    for node, value in (body.get("hostsByNode") or {}).items():
        items = value if isinstance(value, list) else [value]
        hosts[node] = sorted({(h.get("ip") if isinstance(h, dict) else h) for h in items if h})
    return {"instance_id": iid, "kind": kind, "model_id": (stages[0]["model_id"] if stages else None), "world_size": len(stages), "stages": stages, "hosts_by_node": hosts}


def runner_states(state, instance_id=None):
    """Runner id -> status tag of the runners of the instance."""
    lay = layout(state, instance_id)
    out = {}
    for s in lay["stages"]:
        status = (state.get("runners") or {}).get(s["runner"])
        out[s["runner"]] = next(iter(status)) if isinstance(status, dict) and status else None
    return out


def check_placement(state, *, backend, model_id=None, world_size=2, layers=None, instance_id=None, link_addresses=None):
    """Problems between the intended experiment and the placement in ``state``. ``layers`` is the intended layer count per stage in rank order (None = not constrained)."""
    problems = []
    try:
        lay = layout(state, instance_id)
    except LookupError as e:
        return [str(e)]
    want = KINDS.get(backend)
    if want is None:
        problems.append(f"unknown backend {backend!r}")
    elif lay["kind"] != want:
        problems.append(f"instance kind {lay['kind']} is not {want}")
    if lay["world_size"] != world_size:
        problems.append(f"{lay['world_size']} runners placed, expected {world_size}")
    if any(s["world_size"] != world_size for s in lay["stages"]):
        problems.append("a shard reports a world size different from the intended one")
    nodes = {s["node"] for s in lay["stages"]}
    if None in nodes:
        problems.append("a pipeline stage has no node assignment")
    if len(nodes) != len(lay["stages"]):
        problems.append("two pipeline stages share a node (unexpected single-node placement)")
    if sorted(s["device_rank"] for s in lay["stages"] if s["device_rank"] is not None) != list(range(len(lay["stages"]))):
        problems.append("device ranks are not 0..n-1")
    if model_id is not None and lay["model_id"] not in (None, model_id):
        problems.append(f"model {lay['model_id']} is not {model_id}")
    spans = [(s["start_layer"], s["end_layer"]) for s in lay["stages"]]
    if any(a is None or b is None for a, b in spans):
        problems.append("a shard has no layer range")
    else:
        for (_, e1), (s2, _) in zip(spans, spans[1:]):
            if e1 != s2:
                problems.append("layer ranges are not contiguous")
                break
        if spans and spans[0][0] != 0:
            problems.append("the first stage does not start at layer 0")
        totals = {s["total_layers"] for s in lay["stages"] if s["total_layers"] is not None}
        if totals and spans[-1][1] not in totals:
            problems.append("the last stage does not end at the model's last layer")
    if link_addresses is not None and lay["world_size"] > 1:
        used = {ip for ips in lay["hosts_by_node"].values() for ip in ips if ip not in ("0.0.0.0", "::")}
        outside = sorted(used - set(link_addresses))
        if outside or not used:
            problems.append(f"the instance addresses {sorted(used)} are not exactly the data-link addresses" if not used else f"the instance uses addresses outside the data link: {outside}")
    if layers is not None and [s["n_layers"] for s in lay["stages"]] != list(layers):
        problems.append(f"layer counts {[s['n_layers'] for s in lay['stages']]} differ from the intended {list(layers)}")
    return problems


def check_ready(state, instance_id=None):
    """Problems that stop an instance from being benchmarked: any failed runner, any runner not ready."""
    try:
        states = runner_states(state, instance_id)
    except LookupError as e:
        return [str(e)]
    problems = []
    for runner, tag in states.items():
        if tag is None:
            problems.append(f"runner {runner} has no status")
        elif tag.startswith(FAILED_PREFIX):
            problems.append(f"runner {runner} failed")
        elif tag != READY:
            problems.append(f"runner {runner} is {tag}, not {READY}")
    return problems


def check_cluster(state, *, node_count=2, expected_nodes=None):
    """Membership problems: the number of nodes seen in the topology and, optionally, that exactly the expected ones are present."""
    nodes = nodes_of(state)
    problems = []
    if len(nodes) != node_count:
        problems.append(f"{len(nodes)} nodes in the topology, expected {node_count}")
    if expected_nodes is not None and set(nodes) != set(expected_nodes):
        problems.append("node set differs from the expected one")
    return problems


def check_masters(masters_by_node):
    """``masters_by_node``: node id -> the master that node reports. Exactly one master must be agreed by every node."""
    values = {m for m in masters_by_node.values() if m}
    problems = []
    if len(values) > 1:
        problems.append(f"nodes disagree on the master ({len(values)} different masters)")
    if not values:
        problems.append("no master reported")
    if any(not m for m in masters_by_node.values()):
        problems.append("a node reports no master")
    return problems


ELECTION = re.compile(r"Node elected Master - (?:maintaining|promoting) self|Node (\S+) elected master")


def last_election(log_text):
    """The last election outcome in one node's log: ('self', None) when this node is master, ('other', id) when it follows another node, None when no election was logged."""
    last = None
    for m in ELECTION.finditer(log_text or ""):
        last = ("other", m.group(1)) if m.group(1) else ("self", None)
    return last


def masters_by_node(logs_by_node):
    """node id -> the master that node's log last reported (its own id when it is master). Logs of nodes with no election line map to None."""
    out = {}
    for node, text in logs_by_node.items():
        e = last_election(text)
        out[node] = None if e is None else (node if e[0] == "self" else e[1])
    return out


def check_agreement(states_by_node, instance_id=None):
    """Problems when the nodes' own views of the cluster differ: the node set, and the instance with its runner states."""
    problems = []
    views = {}
    for node, st in states_by_node.items():
        try:
            lay = layout(st, instance_id) if st.get("instances") else None
        except LookupError as e:
            problems.append(f"{node}: {e}")
            continue
        views[node] = (frozenset(nodes_of(st)), None if lay is None else (lay["instance_id"], lay["kind"], tuple((s["runner"], s["start_layer"], s["end_layer"]) for s in lay["stages"])),
                       None if lay is None else tuple(sorted(runner_states(st, lay["instance_id"]).items())))
    if len({v[0] for v in views.values()}) > 1:
        problems.append("the nodes see different memberships")
    if len({v[1] for v in views.values()}) > 1:
        problems.append("the nodes see different instances or placements")
    if len({v[2] for v in views.values()}) > 1:
        problems.append("the nodes see different runner states")
    return problems
