"""Validation of Exo cluster-state documents (offline). The single-node state is real; the two-node states are built from it so each defect is one change."""
import copy
import json
import unittest

from benchmarks.suite.stream import exo_validation as validation
from benchmarks.suite.tests.common import FIXTURES

REAL = json.loads((FIXTURES / "exo" / "state_single_ring_ready.json").read_text())


def two_node(kind="MlxTbcclInstance", layers=(7, 21), total=28, ready=True, model="mlx-community/Qwen3-0.6B-8bit"):
    st = copy.deepcopy(REAL)
    iid = next(iter(st["instances"]))
    body = next(iter(st["instances"][iid].values()))
    shard0 = next(iter(body["shardAssignments"]["runnerToShard"].values()))["PipelineShardMetadata"]
    runner_to_shard, node_to_runner, runners = {}, {}, {}
    start = 0
    for rank, n in enumerate(layers):
        shard = copy.deepcopy(shard0)
        shard.update({"deviceRank": rank, "worldSize": len(layers), "startLayer": start, "endLayer": start + n, "nLayers": total})
        shard["modelCard"]["modelId"] = model
        runner_to_shard[f"r{rank}"] = {"PipelineShardMetadata": shard}
        node_to_runner[f"n{rank}"] = f"r{rank}"
        runners[f"r{rank}"] = {"RunnerReady": {}} if ready else {"RunnerLoading": {}}
        start += n
    body["shardAssignments"].update({"runnerToShard": runner_to_shard, "nodeToRunner": node_to_runner})
    st["instances"] = {iid: {kind: body}}
    st["runners"] = runners
    st["topology"] = {"nodes": list(node_to_runner), "connections": {}}
    return st


class Real(unittest.TestCase):
    def test_real_single_node_layout(self):
        lay = validation.layout(REAL)
        self.assertEqual(lay["kind"], "MlxRingInstance")
        self.assertEqual(lay["world_size"], 1)
        self.assertEqual((lay["stages"][0]["start_layer"], lay["stages"][0]["end_layer"], lay["stages"][0]["total_layers"]), (0, 28, 28))
        self.assertEqual(validation.check_ready(REAL), [])
        self.assertEqual(validation.check_placement(REAL, backend="MlxRing", world_size=1, model_id="mlx-community/Qwen3-0.6B-8bit"), [])


class Placement(unittest.TestCase):
    def check(self, st, **kw):
        kw.setdefault("backend", "MlxTbccl")
        return validation.check_placement(st, **kw)

    def test_good_two_stage_tbccl_placement(self):
        self.assertEqual(self.check(two_node(), layers=(7, 21)), [])

    def test_wrong_instance_kind(self):
        self.assertTrue(any("is not MlxTbcclInstance" in p for p in self.check(two_node(kind="MlxRingInstance"))))

    def test_unexpected_single_node_placement(self):
        self.assertTrue(any("1 runners placed" in p for p in self.check(REAL, backend="MlxRing")))

    def test_stage_without_a_node_is_not_a_two_node_placement(self):
        st = two_node()
        iid = next(iter(st["instances"]))
        body = next(iter(st["instances"][iid].values()))
        body["shardAssignments"]["nodeToRunner"] = {"n0": "r0"}
        self.assertTrue(any("no node assignment" in p for p in self.check(st)))

    def test_incorrect_layer_assignment(self):
        self.assertTrue(any("differ from the intended" in p for p in self.check(two_node(layers=(14, 14)), layers=(7, 21))))

    def test_layers_that_leave_a_gap(self):
        st = two_node()
        iid = next(iter(st["instances"]))
        body = next(iter(st["instances"][iid].values()))
        body["shardAssignments"]["runnerToShard"]["r1"]["PipelineShardMetadata"]["startLayer"] = 9
        self.assertTrue(any("not contiguous" in p for p in self.check(st)))

    def test_last_stage_not_at_the_last_layer(self):
        self.assertTrue(any("last layer" in p for p in self.check(two_node(layers=(7, 20)))))

    def test_model_mismatch(self):
        self.assertTrue(any("is not mlx-community/Qwen3-0.6B-8bit" in p for p in self.check(two_node(model="other/model"), model_id="mlx-community/Qwen3-0.6B-8bit")))

    def test_no_instance_or_many(self):
        self.assertTrue(self.check({"instances": {}}))
        two = copy.deepcopy(REAL)
        two["instances"]["second"] = two["instances"][next(iter(two["instances"]))]
        self.assertTrue(self.check(two, backend="MlxRing"))

    def test_unknown_backend(self):
        self.assertTrue(any("unknown backend" in p for p in self.check(two_node(), backend="Nccl")))


def topo(edges, nodes=("n0", "n1")):
    conns = {}
    for a, b in edges:
        conns.setdefault(a, {})[b] = [{"sinkMultiaddr": {"ip_address": "192.0.2.1"}}]
    return {"topology": {"nodes": list(nodes), "connections": conns}}


class Connectivity(unittest.TestCase):
    def test_both_directions_are_needed(self):
        self.assertTrue(validation.connected(topo([("n0", "n1"), ("n1", "n0")])))
        self.assertFalse(validation.connected(topo([("n0", "n1")])))
        self.assertFalse(validation.connected(topo([])))

    def test_node_count_must_match(self):
        self.assertFalse(validation.connected(topo([("n0", "n1"), ("n1", "n0")]), node_count=3))
        self.assertFalse(validation.connected(topo([], nodes=("n0",)), node_count=2))

    def test_empty_connection_lists_do_not_count(self):
        st = topo([("n0", "n1"), ("n1", "n0")])
        st["topology"]["connections"]["n1"]["n0"] = []
        self.assertFalse(validation.connected(st))


class LinkAddresses(unittest.TestCase):
    def with_hosts(self, hosts):
        st = two_node()
        iid = next(iter(st["instances"]))
        next(iter(st["instances"][iid].values()))["hostsByNode"] = hosts
        return st

    def test_addresses_on_the_link_pass(self):
        st = self.with_hosts({"n0": "192.0.2.1", "n1": "192.0.2.2"})
        self.assertEqual(validation.check_placement(st, backend="MlxTbccl", link_addresses=["192.0.2.1", "192.0.2.2"]), [])

    def test_ring_style_host_lists_with_a_wildcard_listen_address(self):
        st = self.with_hosts({"n0": [{"ip": "0.0.0.0", "port": 1}, {"ip": "192.0.2.2", "port": 1}], "n1": [{"ip": "192.0.2.1", "port": 1}, {"ip": "0.0.0.0", "port": 1}]})
        self.assertEqual(validation.check_placement(st, backend="MlxTbccl", link_addresses=["192.0.2.1", "192.0.2.2"]), [])

    def test_management_lan_address_is_reported(self):
        st = self.with_hosts({"n0": "192.0.2.1", "n1": "192.0.2.7"})
        self.assertTrue(any("outside the data link" in p for p in validation.check_placement(st, backend="MlxTbccl", link_addresses=["192.0.2.1", "192.0.2.2"])))

    def test_no_addresses_at_all_is_reported(self):
        st = self.with_hosts({})
        self.assertTrue(validation.check_placement(st, backend="MlxTbccl", link_addresses=["192.0.2.1", "192.0.2.2"]))


class Readiness(unittest.TestCase):
    def test_not_ready(self):
        self.assertTrue(any("RunnerLoading" in p for p in validation.check_ready(two_node(ready=False))))

    def test_failed_runner(self):
        st = two_node()
        st["runners"]["r1"] = {"RunnerFailed": {"errorMessage": "x"}}
        self.assertTrue(any("failed" in p for p in validation.check_ready(st)))

    def test_missing_runner_status(self):
        st = two_node()
        del st["runners"]["r0"]
        self.assertTrue(any("no status" in p for p in validation.check_ready(st)))


class Cluster(unittest.TestCase):
    def test_membership(self):
        self.assertEqual(validation.check_cluster(two_node()), [])
        self.assertTrue(validation.check_cluster(REAL))
        self.assertTrue(validation.check_cluster(two_node(), expected_nodes=["n0", "zzz"]))

    def test_discovery_failure_is_an_empty_topology(self):
        self.assertTrue(validation.check_cluster({"topology": {"nodes": []}}))

    def test_masters(self):
        self.assertEqual(validation.check_masters({"a": "m", "b": "m"}), [])
        self.assertTrue(any("disagree" in p for p in validation.check_masters({"a": "m1", "b": "m2"})))
        self.assertTrue(validation.check_masters({"a": None, "b": None}))
        self.assertTrue(validation.check_masters({"a": "m", "b": None}))


class Election(unittest.TestCase):
    SELF = "INFO Node elected Master - promoting self\nINFO Unpausing API\n"
    OTHER = "INFO Node n0 elected master\n"

    def test_log_lines(self):
        self.assertEqual(validation.last_election(self.SELF), ("self", None))
        self.assertEqual(validation.last_election(self.OTHER), ("other", "n0"))
        self.assertIsNone(validation.last_election("nothing"))

    def test_last_election_wins(self):
        self.assertEqual(validation.last_election(self.SELF + "Node n1 elected master - demoting self\n"), ("other", "n1"))

    def test_masters_by_node_and_agreement(self):
        masters = validation.masters_by_node({"n0": self.SELF, "n1": self.OTHER})
        self.assertEqual(masters, {"n0": "n0", "n1": "n0"})
        self.assertEqual(validation.check_masters(masters), [])
        split = validation.masters_by_node({"n0": self.SELF, "n1": self.SELF})
        self.assertTrue(validation.check_masters(split))
        self.assertTrue(validation.check_masters(validation.masters_by_node({"n0": self.SELF, "n1": "no election"})))


class Agreement(unittest.TestCase):
    def test_same_view(self):
        self.assertEqual(validation.check_agreement({"n0": two_node(), "n1": two_node()}), [])

    def test_different_runner_states(self):
        self.assertTrue(any("runner states" in p for p in validation.check_agreement({"n0": two_node(), "n1": two_node(ready=False)})))

    def test_different_membership(self):
        other = two_node()
        other["topology"] = {"nodes": ["n0"], "connections": {}}
        self.assertTrue(any("memberships" in p for p in validation.check_agreement({"n0": two_node(), "n1": other})))

    def test_different_placement(self):
        self.assertTrue(any("placements" in p for p in validation.check_agreement({"n0": two_node(layers=(7, 21)), "n1": two_node(layers=(14, 14))})))


if __name__ == "__main__":
    unittest.main()
