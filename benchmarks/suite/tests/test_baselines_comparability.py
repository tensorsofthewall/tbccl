"""Comparability rules against the shipped case definitions and synthetic variants: which pairs may be compared, and why the others may not."""
import copy
import unittest

from benchmarks.suite.baselines import catalog, comparability


def get(case_id, **over):
    case = copy.deepcopy(catalog.case_by_id(case_id))
    case.update(over)
    return case


def reasons(result):
    return " ".join(result["explanations"])


class Equivalent(unittest.TestCase):
    def test_a_case_is_comparable_with_itself_and_its_re_measurement(self):
        a = get("exo.tpot.p128-o32.c1.MlxRing.mac>linux")
        self.assertTrue(comparability.check(a, copy.deepcopy(a))["compatible"])

    def test_software_versions_may_differ(self):
        a = get("native.transfer.e2e.16MiB.host-host.a-to-b", software_versions={"tbccl": "0.6.0"})
        b = get("native.transfer.e2e.16MiB.host-host.a-to-b", software_versions={"tbccl": "0.7.0"})
        self.assertTrue(comparability.check(a, b)["compatible"])

    def test_unknown_purpose_is_an_error(self):
        a = get("native.tcp-rtt.4KiB")
        with self.assertRaises(ValueError):
            comparability.check(a, a, "speedup")


class NativeMismatches(unittest.TestCase):
    def test_payload_direction_memory_and_mode_must_match(self):
        base = get("native.transfer.e2e.16MiB.host-host.a-to-b")
        for field, value in (("payload_bytes", 4096), ("data_direction", "b-to-a"), ("memory_kind", "cuda-pinned"), ("benchmark_mode", "network_only"), ("rank_count", 3),
                             ("completion_semantics", "send() has returned on the source"), ("device_pairing", "cuda-metal")):
            result = comparability.check(base, get("native.transfer.e2e.16MiB.host-host.a-to-b", **{field: value}))
            self.assertFalse(result["compatible"], field)
            self.assertIn(field, [m["dimension"] for m in result["mismatches"]])

    def test_native_round_trip_against_framework_send_ack_is_rejected(self):
        native_rtt = get("native.p2p-rtt.4KiB")
        framework_ack = get("torch.p2p.send-recv-ack.cpu-cpu.4KiB")
        for purpose in ("regression", "native_vs_framework"):
            result = comparability.check(native_rtt, framework_ack, purpose)
            self.assertFalse(result["compatible"], purpose)
            self.assertIn("timing_scope", [m["dimension"] for m in result["mismatches"]])
        self.assertIn("timing and completion scopes", reasons(comparability.check(native_rtt, framework_ack)))

    def test_native_against_framework_with_the_same_scope_is_a_labelled_observation(self):
        native = get("native.transfer.e2e.cpu-cpu.4KiB.a-to-b")
        framework = get("torch.p2p.send-recv-ack.cpu-cpu.4KiB")
        self.assertFalse(comparability.check(native, framework, "regression")["compatible"])
        result = comparability.check(native, framework, "native_vs_framework")
        self.assertTrue(result["compatible"], result["mismatches"])
        self.assertTrue(any("not an isolated adapter overhead" in label for label in result["labels"]))

    def test_native_vs_framework_requires_a_native_and_a_framework_family(self):
        a = get("native.transfer.e2e.cpu-cpu.4KiB.a-to-b")
        self.assertFalse(comparability.check(a, copy.deepcopy(a), "native_vs_framework")["compatible"])
        collective = get("native.allreduce.ring.1MiB.2r")
        self.assertFalse(comparability.check(a, collective, "native_vs_framework")["compatible"])

    def test_round_trip_is_not_comparable_with_a_one_way_scope(self):
        rtt = get("native.p2p-rtt.4KiB")
        one_way = get("native.p2p-rtt.4KiB", timing_scope="native_one_way_per_message")
        self.assertFalse(comparability.check(rtt, one_way)["compatible"])

    def test_async_submission_is_not_comparable_with_completion(self):
        completion = get("native.async.cuda-chunked.256KiB.16MiB.a-to-b")
        submission = get("native.async.cuda-chunked.256KiB.16MiB.a-to-b", timing_scope="enqueue_return", completion_semantics="enqueue() returned; nothing is complete")
        result = comparability.check(completion, submission)
        self.assertFalse(result["compatible"])
        self.assertIn("completion semantics", reasons(result))

    def test_collective_algorithm_and_size_must_match(self):
        ring = get("native.allreduce.ring.1MiB.2r")
        reference = get("native.allreduce.reference.1MiB.2r")
        self.assertEqual([m["dimension"] for m in comparability.check(ring, reference)["mismatches"]], ["algorithm"])

    def test_metric_aggregation_must_match(self):
        a = get("native.tcp-rtt.4KiB")
        b = get("native.tcp-rtt.4KiB", aggregation="median_of_repetition_medians")
        self.assertIn("aggregation", [m["dimension"] for m in comparability.check(a, b)["mismatches"]])

    def test_unset_defining_dimension_cannot_establish_equivalence(self):
        a = get("native.transfer.e2e.16MiB.host-host.a-to-b", payload_bytes=None)
        result = comparability.check(a, copy.deepcopy(a))
        self.assertFalse(result["compatible"])
        self.assertEqual(result["mismatches"][0]["kind"], "missing")
        self.assertIn("not recorded", reasons(result))


class InferenceMismatches(unittest.TestCase):
    def test_vllm_against_exo_is_never_a_transport_benchmark(self):
        v = get("vllm.tpot.p128-o32.c1.cuda>metal")
        e = get("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac")
        for purpose in comparability.PURPOSES:
            result = comparability.check(v, e, purpose)
            self.assertFalse(result["compatible"], purpose)
        self.assertIn("different inference runtimes", reasons(comparability.check(v, e)))

    def test_quantization_mismatch_with_the_same_runtime_family(self):
        a = get("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac")
        b = get("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac", quantization="bf16")
        result = comparability.check(a, b, "ab_transport")
        self.assertFalse(result["compatible"])
        self.assertIn("quantization", reasons(result))

    def test_ring_19_9_against_tbccl_9_19_is_not_isolated_transport_overhead(self):
        ring = get("exo.tpot.p128-o32.c1.MlxRing.mac>linux")
        tbccl = get("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac")
        result = comparability.check(ring, tbccl, "ab_transport")
        self.assertFalse(result["compatible"])
        self.assertEqual({m["dimension"] for m in result["mismatches"]}, {"stage_order", "layer_split"})
        self.assertIn("cannot be attributed to the transport", reasons(result))

    def test_matched_placement_ab_is_allowed_and_labelled(self):
        ring = get("exo.tpot.p128-o32.c1.MlxRing.mac>linux")
        tbccl = get("exo.tpot.p128-o32.c1.MlxTbccl.mac>linux")
        result = comparability.check(ring, tbccl, "ab_transport")
        self.assertTrue(result["compatible"], result["mismatches"])
        self.assertTrue(any("not protocol overhead" in label for label in result["labels"]))
        self.assertFalse(comparability.check(ring, tbccl, "regression")["compatible"])

    def test_stage_order_contrast_is_allowed_only_as_a_labelled_placement_contrast(self):
        a, b = get("vllm.tpot.p128-o32.c1.cuda>metal"), get("vllm.tpot.p128-o32.c1.metal>cuda")
        self.assertFalse(comparability.check(a, b, "regression")["compatible"])
        self.assertFalse(comparability.check(a, b, "ab_transport")["compatible"])
        result = comparability.check(a, b, "ab_placement")
        self.assertTrue(result["compatible"], result["mismatches"])
        self.assertTrue(any("confounded" in label and "not attributable to the transport" in label for label in result["labels"]))

    def test_placement_contrast_does_not_allow_a_backend_difference_at_the_same_time(self):
        ring = get("exo.tpot.p128-o32.c1.MlxRing.mac>linux")
        tbccl = get("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac")
        self.assertEqual({m["dimension"] for m in comparability.check(ring, tbccl, "ab_placement")["mismatches"]}, {"backend"})

    def test_traced_against_untraced_is_rejected(self):
        a = get("exo.tpot.p128-o32.c1.MlxTbccl.mac>linux")
        b = get("exo.tpot.p128-o32.c1.MlxTbccl.mac>linux", instrumentation="traced")
        result = comparability.check(a, b)
        self.assertFalse(result["compatible"])
        self.assertIn("traced", reasons(result))

    def test_27b_short_prompt_against_06b_long_prompt_is_not_model_scaling(self):
        big = get("exo.tpot.short-prompt.c1.MlxTbccl.27B")
        small = get("exo.tpot.p128-o32.c1.MlxTbccl.mac>linux", workload="p512-o16-greedy")
        result = comparability.check(big, small, "ab_variant")
        self.assertFalse(result["compatible"])
        self.assertIn("model-size scaling requires a fixed workload", reasons(result))

    def test_variant_comparison_is_allowed_only_on_the_variant(self):
        default = get("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac", variant="default", stage_order="linux>mac", layer_split="9/19")
        step = get("exo.tpot.p128-o32.c1.MlxTbccl.activity-step.linux>mac")
        result = comparability.check(default, step, "ab_variant")
        self.assertTrue(result["compatible"], result["mismatches"])
        self.assertIn("experimental", " ".join(result["labels"]))

    def test_client_placement_differs(self):
        a = get("vllm.tpot.p128-o32.c1.cuda>metal")
        b = get("vllm.tpot.p128-o32.c1.cuda>metal", client_placement="remote")
        self.assertIn("different host", reasons(comparability.check(a, b)))

    def test_extra_dimensions_are_compared(self):
        a = get("torch.allreduce.cpu-cpu.1MiB.blocking")
        b = get("torch.allreduce.cpu-cpu.1MiB.blocking", extra_dimensions={"blocking_mode": "async"})
        self.assertIn("extra_dimensions.blocking_mode", [m["dimension"] for m in comparability.check(a, b)["mismatches"]])


if __name__ == "__main__":
    unittest.main()
