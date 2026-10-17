"""Which measurements may be compared directly. A comparison is allowed only when every dimension that defines the experiment agrees; everything else is
rejected with an explanation, and the contextual view never computes a ratio or speedup.

Purposes:
  regression    the same case measured again: every identity dimension, including the backend, must be equal (software versions may differ and are reported)
  ab_transport  two communication backends of the same experiment (for example MlxRing against MlxTbccl): only `backend` may differ. The difference is an
                application-level difference of the two integrations, not the overhead of the transport protocol
  ab_variant    one declared variant of a backend against its default (for example an experimental policy): only `variant` may differ
  ab_placement  the same experiment with a different pipeline placement (stage order and/or layer split). Whatever differs between the placements (which device
                computes which layers, where the API host and the sampler sit, which direction the traffic flows) is confounded: the difference is not attributable to
                the transport or to any single cause
  native_vs_framework
                a native tool against the framework binding of the same operation (p2p, collective or async pairs only). The timing scope, completion semantics and
                every experiment dimension must still agree; the tool, metric name and datatype labels may differ. The result is an observed difference between two
                measurements, not an isolated adapter overhead
"""
from __future__ import annotations

INFERENCE_FAMILIES = ("vllm_pipeline", "exo_pipeline")
NATIVE_FAMILIES = ("raw_network", "native_p2p", "native_staging", "native_async", "native_collective", "native_overlap")
TORCH_FAMILIES = ("torch_p2p", "torch_collective", "torch_async", "torch_ddp")

NATIVE = ("operation", "benchmark_mode", "device_pairing", "memory_kind", "data_direction", "rank_count", "payload_bytes")
INFERENCE = ("model_id", "model_hash", "quantization", "workload", "request_concurrency", "stage_order", "layer_split", "scheduling_mode", "client_placement", "sampling", "runtime",
             "device_pairing")
BASE = ("family", "benchmark_adapter", "metric_name", "aggregation", "hardware_profile", "transport", "backend", "variant", "timing_scope", "completion_semantics",
        "instrumentation")

FAMILY_DIMENSIONS = {
    "raw_network": NATIVE,
    "native_p2p": NATIVE,
    "native_staging": NATIVE,
    "native_async": NATIVE,
    "native_overlap": NATIVE + ("workload",),
    "native_collective": NATIVE + ("algorithm", "datatype"),
    "torch_p2p": NATIVE + ("datatype",),
    "torch_collective": NATIVE + ("datatype",),
    "torch_async": NATIVE + ("datatype",),
    "torch_ddp": ("operation", "device_pairing", "rank_count", "payload_bytes", "datatype", "model_id", "workload"),
    "vllm_pipeline": INFERENCE,
    "exo_pipeline": INFERENCE,
}
# dimensions that must be set on both sides: an unset value means equivalence cannot be established
MUST_BE_SET = {
    **{f: ("operation", "timing_scope", "completion_semantics", "device_pairing", "payload_bytes") for f in NATIVE_FAMILIES + TORCH_FAMILIES if f != "torch_ddp"},
    "torch_ddp": ("operation", "timing_scope", "completion_semantics", "device_pairing", "model_id", "workload"),
    "vllm_pipeline": ("model_id", "quantization", "workload", "stage_order", "layer_split", "sampling", "runtime", "timing_scope", "completion_semantics"),
    "exo_pipeline": ("model_id", "quantization", "workload", "stage_order", "layer_split", "sampling", "runtime", "timing_scope", "completion_semantics"),
}
PURPOSES = {"regression": (), "ab_transport": ("backend",), "ab_variant": ("variant",),
            "ab_placement": ("stage_order", "layer_split"),
            "native_vs_framework": ("family", "benchmark_adapter", "metric_name", "backend", "datatype")}
FRAMEWORK_PAIRS = {frozenset(("native_p2p", "torch_p2p")), frozenset(("native_collective", "torch_collective")), frozenset(("native_async", "torch_async"))}


def required_dimensions(family: str) -> tuple:
    return BASE + FAMILY_DIMENSIONS.get(family, ())


def _unset(value) -> bool:
    return value is None


def _extra(case):
    return case.get("extra_dimensions") or {}


def mismatches(a: dict, b: dict, allowed=()) -> list:
    """[{dimension, baseline, candidate, kind}] for every identity dimension on which the two cases disagree. kind is 'differs', or 'missing' when one side is unset and
    the other is not (equivalence then cannot be established)."""
    out = []
    dims = list(dict.fromkeys(required_dimensions(a["family"]) + required_dimensions(b["family"])))
    for dim in dims:
        if dim in allowed:
            continue
        va, vb = a.get(dim), b.get(dim)
        must = dim in MUST_BE_SET.get(a["family"], ()) or dim in MUST_BE_SET.get(b["family"], ())
        if va == vb and not (must and _unset(va)):
            continue
        out.append({"dimension": dim, "baseline": va, "candidate": vb, "kind": "missing" if (_unset(va) or _unset(vb)) else "differs"})
    ea, eb = _extra(a), _extra(b)
    for key in sorted(set(ea) | set(eb)):
        if key in allowed:
            continue
        if ea.get(key) != eb.get(key):
            out.append({"dimension": f"extra_dimensions.{key}", "baseline": ea.get(key), "candidate": eb.get(key), "kind": "differs"})
    return out


def explanations(a: dict, b: dict, found: list) -> list:
    """Why the pair is misleading, for the combinations that have caused wrong conclusions."""
    names = {m["dimension"] for m in found}
    fa, fb = a["family"], b["family"]
    out = []
    if fa != fb:
        if (fa in NATIVE_FAMILIES and fb in TORCH_FAMILIES) or (fb in NATIVE_FAMILIES and fa in TORCH_FAMILIES):
            out.append("a native measurement and a framework measurement have different timing and completion scopes (for example a native round trip against a framework "
                       "send plus acknowledgement); they are two measurements, not an overhead figure")
        elif {fa, fb} == set(INFERENCE_FAMILIES):
            out.append("these are different inference runtimes with their own models, quantization, schedulers and clients; they are not a transport benchmark")
        else:
            out.append(f"different measurement families ({fa} and {fb}) answer different questions")
    elif "timing_scope" in names or "completion_semantics" in names:
        out.append("the timing scope or completion semantics differ: the two numbers do not measure the same interval")
    if a["family"] in INFERENCE_FAMILIES or b["family"] in INFERENCE_FAMILIES:
        if names & {"model_id", "model_hash", "quantization"}:
            out.append("the model, its files or its quantization differ: the compute per step differs, so this is not a transport comparison")
        if names & {"stage_order", "layer_split"}:
            out.append("the pipeline placement (stage order or layer split) differs: an end-to-end difference cannot be attributed to the transport")
        if ("model_id" in names or "quantization" in names) and names & {"workload", "request_concurrency"}:
            out.append("model and workload both differ: model-size scaling requires a fixed workload")
        if "client_placement" in names:
            out.append("the client sits on a different host: client-observed latencies include a different path")
    if "instrumentation" in names:
        out.append("one side is traced and the other is not: tracing adds overhead that must be labelled, not hidden in a comparison")
    if "backend" in names and not names - {"backend"}:
        out.append("only the communication backend differs: the difference is an application-level difference of the two integrations, not the overhead of the protocol")
    if "variant" in names and not names - {"variant"}:
        out.append("only the declared variant differs: an experimental policy against the default; it does not generalise beyond the measured model, hosts and conditions")
    if any(m["kind"] == "missing" for m in found):
        out.append("a dimension that defines the experiment is not recorded on at least one side, so equivalence cannot be established")
    return out


def check(a: dict, b: dict, purpose: str = "regression") -> dict:
    """{compatible, purpose, mismatches, explanations, labels}. `labels` are statements any report of the comparison must carry."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; expected one of {sorted(PURPOSES)}")
    allowed = PURPOSES[purpose]
    found = mismatches(a, b, allowed)
    labels = []
    if purpose == "native_vs_framework":
        if a["family"] != b["family"] and frozenset((a["family"], b["family"])) not in FRAMEWORK_PAIRS:
            found.insert(0, {"dimension": "family", "baseline": a["family"], "candidate": b["family"], "kind": "differs"})
        elif a["family"] == b["family"]:
            found.insert(0, {"dimension": "family", "baseline": a["family"], "candidate": b["family"], "kind": "differs"})
            return {"compatible": False, "purpose": purpose, "mismatches": found, "explanations": ["native_vs_framework compares a native tool with a framework binding; both cases are the same family"], "labels": []}
        labels.append("observed difference between a native tool and its framework binding; not an isolated adapter overhead")
    if purpose == "ab_transport":
        labels.append("application-level difference between two backends, not protocol overhead")
    if purpose == "ab_variant":
        labels.append("experimental variant against its default")
    if purpose == "ab_placement":
        labels.append("placement contrast: stage order, API host, sampler location and traffic direction are confounded; not attributable to the transport")
    return {"compatible": not found, "purpose": purpose, "mismatches": found, "explanations": explanations(a, b, found), "labels": labels}
