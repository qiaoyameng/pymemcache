import hashlib
import json
import math

import pytest
from unittest import mock

from pymemcache.client.hash import HashClient
from pymemcache.client.migration import estimate_migration, format_report, main
from pymemcache.client.rendezvous import RendezvousHash


def fixed_hash(mapping):
    """A hash function returning fixed values keyed by the full "node-key"."""
    def hash_function(value, seed=0):
        return mapping[value]
    return hash_function


def node_hash(mapping):
    """A hash function returning fixed values keyed only by the node part."""
    def hash_function(value, seed=0):
        node, _, _ = value.rpartition("-")
        return mapping[node]
    return hash_function


def md5_hash(value, seed=0):
    return int(hashlib.md5(value.encode()).hexdigest(), 16) % (1 << 32)


HASH_MAX = (1 << 32) - 1


@pytest.mark.unit()
def test_weighted_equal_hash_heavier_wins():
    # Identical hash for every node: the weight alone decides, so the
    # heaviest node wins and the ranking follows the weights.
    rendezvous = RendezvousHash(
        nodes=["a", "b", "c"],
        weights={"a": 1, "b": 2, "c": 3},
        hash_function=node_hash({"a": 100, "b": 100, "c": 100}),
    )
    assert "c" == rendezvous.get_node("key")
    assert ["c", "b", "a"] == rendezvous.get_nodes("key")


@pytest.mark.unit()
def test_weighted_equal_weight_higher_hash_wins():
    # Identical weights: a higher hash value gives a higher score because
    # -w / log(u) grows as u approaches 1.
    rendezvous = RendezvousHash(
        nodes=["a", "b"],
        weights={"a": 1, "b": 1},
        hash_function=node_hash({"a": 10, "b": 20}),
    )
    assert "b" == rendezvous.get_node("key")
    assert ["b", "a"] == rendezvous.get_nodes("key")


@pytest.mark.unit()
def test_weighted_score_matches_formula():
    # u = (h + 1) / (2**32 + 1); score = -weight / log(u).
    # "a": u ~= 0.5 with weight 1 -> score ~= 1.4427
    # "b": u ~= 0.25 with weight 3 -> score ~= 2.1640  (b wins)
    # "c": u ~= 0.5 with weight 1.5 -> score ~= 2.1640... use weight 1 -> a ties?
    half = (1 << 31) - 1
    quarter = (1 << 30) - 1
    rendezvous = RendezvousHash(
        nodes=["a", "b"],
        weights={"a": 1, "b": 3},
        hash_function=node_hash({"a": half, "b": quarter}),
    )
    u_a = (half + 1) / ((1 << 32) + 1)
    u_b = (quarter + 1) / ((1 << 32) + 1)
    score_a = -1.0 / math.log(u_a)
    score_b = -3.0 / math.log(u_b)
    assert score_b > score_a
    assert "b" == rendezvous.get_node("key")


@pytest.mark.unit()
def test_weighted_boundary_hash_values():
    # Hash values of 0 and the 32-bit maximum must not trigger log(0) or a
    # division by zero (log(1)); negative (signed) hashes are normalized.
    rendezvous = RendezvousHash(
        nodes=["zero", "max", "neg"],
        weights={"zero": 1, "max": 1, "neg": 1},
        hash_function=node_hash({"zero": 0, "max": HASH_MAX, "neg": -(1 << 31)}),
    )
    winner = rendezvous.get_node("key")
    # u for "max" is closest to 1, so it wins; all scores are finite.
    assert "max" == winner
    ranked = rendezvous.get_nodes("key")
    assert ["max", "neg", "zero"] == ranked
    # Repeated calls are perfectly stable.
    for _ in range(10):
        assert rendezvous.get_nodes("key") == ranked


@pytest.mark.unit()
def test_weighted_all_zero_hashes_deterministic():
    # Every hash is 0: scores tie exactly, and the tie is broken
    # deterministically by node name (largest string first), matching the
    # historical collision behavior.
    rendezvous = RendezvousHash(
        nodes=["a", "c", "b"],
        weights={},
        hash_function=node_hash({"a": 0, "b": 0, "c": 0}),
    )
    for i in range(100):
        assert "c" == rendezvous.get_node(i)
        assert ["c", "b", "a"] == rendezvous.get_nodes(i)


@pytest.mark.unit()
def test_get_nodes_count_and_consistency():
    rendezvous = RendezvousHash(
        nodes=["a", "b", "c"], weights={}, hash_function=md5_hash
    )
    for i in range(100):
        ranked = rendezvous.get_nodes("key-%d" % i)
        assert 3 == len(ranked)
        assert sorted(ranked) == ["a", "b", "c"]
        assert ranked[0] == rendezvous.get_node("key-%d" % i)
        assert ranked[:2] == rendezvous.get_nodes("key-%d" % i, count=2)


@pytest.mark.unit()
def test_weight_increase_only_moves_keys_to_that_node():
    keys = ["key-%d" % i for i in range(500)]
    rendezvous = RendezvousHash(
        nodes=["a", "b", "c"],
        weights={"a": 1, "b": 1, "c": 1},
        hash_function=md5_hash,
    )
    before = {key: rendezvous.get_node(key) for key in keys}

    rendezvous.set_weight("c", 8)
    after = {key: rendezvous.get_node(key) for key in keys}

    moved = [key for key in keys if before[key] != after[key]]
    assert moved, "expected a heavier node to take over some keys"
    # Every key that moved went to "c"; "a" and "b" never trade keys.
    assert all(after[key] == "c" for key in moved)
    assert all(before[key] != "c" for key in moved)
    untouched = [key for key in keys if before[key] == after[key]]
    assert all(after[key] == before[key] for key in untouched)


@pytest.mark.unit()
def test_added_node_only_takes_keys_from_others():
    keys = ["key-%d" % i for i in range(500)]
    rendezvous = RendezvousHash(
        nodes=["a", "b", "c"], weights={}, hash_function=md5_hash
    )
    before = {key: rendezvous.get_node(key) for key in keys}

    rendezvous.add_node("d", weight=2)
    after = {key: rendezvous.get_node(key) for key in keys}

    moved = [key for key in keys if before[key] != after[key]]
    assert moved, "expected the new node to take over some keys"
    assert all(after[key] == "d" for key in moved)


@pytest.mark.unit()
def test_unweighted_mode_rejects_weights():
    rendezvous = RendezvousHash(nodes=["a"])
    with pytest.raises(ValueError):
        rendezvous.add_node("b", weight=2)
    with pytest.raises(ValueError):
        rendezvous.set_weight("a", 2)
    # Unweighted routing is untouched.
    assert "a" == rendezvous.get_node("anything")


@pytest.mark.unit()
def test_duplicate_nodes():
    rendezvous = RendezvousHash(nodes=["a", "a", "b"], weights={"b": 2})
    assert ["a", "b"] == rendezvous.nodes
    assert {"a": 1.0, "b": 2.0} == rendezvous.weights

    rendezvous.add_node("a")
    rendezvous.add_node("a", weight=9)
    assert ["a", "b"] == rendezvous.nodes
    # Re-adding an existing node is a no-op and never reweights it.
    assert 1.0 == rendezvous.weights["a"]

    # Duplicates never appear in a ranking.
    assert ["b", "a"] == rendezvous.get_nodes("key", count=2)


@pytest.mark.unit()
def test_invalid_weights():
    for bad in (0, -1, -0.5, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            RendezvousHash(weights={"a": bad})
        rendezvous = RendezvousHash(nodes=["a"], weights={})
        with pytest.raises(ValueError):
            rendezvous.add_node("b", weight=bad)
        with pytest.raises(ValueError):
            rendezvous.set_weight("a", bad)

    for bad in ("2", None.__class__, True, [1]):
        with pytest.raises(TypeError):
            RendezvousHash(weights={"a": bad})


@pytest.mark.unit()
def test_remove_node():
    rendezvous = RendezvousHash(
        nodes=["a", "b", "c"], weights={"c": 5}, hash_function=md5_hash
    )
    rendezvous.remove_node("c")
    assert ["a", "b"] == rendezvous.nodes
    assert "c" not in rendezvous.weights
    for i in range(100):
        assert "c" not in rendezvous.get_nodes("key-%d" % i)

    with pytest.raises(ValueError):
        rendezvous.remove_node("c")
    with pytest.raises(ValueError):
        rendezvous.set_weight("c", 2)

    rendezvous.remove_node("a")
    rendezvous.remove_node("b")
    assert rendezvous.get_node("key") is None
    assert [] == rendezvous.get_nodes("key")


@pytest.mark.unit()
def test_default_weight_is_one():
    rendezvous = RendezvousHash(nodes=["a"], weights={"b": 4})
    assert 1.0 == rendezvous.weights["a"]
    assert 4.0 == rendezvous.weights["b"]
    rendezvous.add_node("c")
    assert 1.0 == rendezvous.weights["c"]


@pytest.mark.unit()
def test_hash_client_server_weights():
    with mock.patch("pymemcache.client.hash.HashClient.client_class"):
        client = HashClient(
            [("10.0.0.1", 11211), ("10.0.0.2", 11211), ("10.0.0.3", 11211)],
            server_weights={"10.0.0.2:11211": 3},
        )
    assert client.hasher.weighted
    assert {
        "10.0.0.1:11211": 1.0,
        "10.0.0.2:11211": 3.0,
        "10.0.0.3:11211": 1.0,
    } == client.hasher.weights

    # Removing a server drops its weight too.
    client.remove_server("10.0.0.2", 11211)
    assert "10.0.0.2:11211" not in client.hasher.weights


@pytest.mark.unit()
def test_hash_client_accepts_hasher_instance():
    hasher = RendezvousHash(weights={}, hash_function=md5_hash)
    client = HashClient([], hasher=hasher)
    assert client.hasher is hasher


@pytest.mark.unit()
def test_hash_client_without_weights_keeps_legacy_routing():
    with mock.patch("pymemcache.client.hash.HashClient.client_class"):
        client = HashClient([("10.0.0.1", 11211), ("10.0.0.2", 11211)])
    assert not client.hasher.weighted
    assert {} == client.hasher.weights


@pytest.mark.unit()
def test_estimate_migration_invariants():
    keys = ["key-%d" % i for i in range(1000)]
    before = RendezvousHash(nodes=["a", "b", "c"], hash_function=md5_hash)
    after = RendezvousHash(
        nodes=["a", "b", "c"], weights={"c": 6}, hash_function=md5_hash
    )

    result = estimate_migration(keys, before, after)
    stats = result["nodes"]

    assert result["sample_size"] == 1000
    assert result["moved"] > 0
    # Flow conservation: everything that leaves a node arrives somewhere.
    assert sum(s["in"] for s in stats.values()) == result["moved"]
    assert sum(s["out"] for s in stats.values()) == result["moved"]
    for node, s in stats.items():
        assert s["after"] - s["before"] == s["in"] - s["out"]
    # Weighted mode is monotonic: only "c" gains keys in this sample.
    assert stats["a"]["in"] == 0
    assert stats["b"]["in"] == 0
    assert stats["c"]["out"] == 0
    # The result is careful about being sample-based only.
    assert "sample" in result["note"]
    assert "not a measurement of, the full keyspace" in result["note"]
    assert "sample" in format_report(result)


@pytest.mark.unit()
def test_estimate_migration_deterministic_small_case():
    collide = node_hash({"a": 7, "b": 7})
    keys = ["k1", "k2", "k3"]
    before = RendezvousHash(nodes=["a", "b"], hash_function=collide)
    after = RendezvousHash(nodes=["b"], hash_function=collide)

    result = estimate_migration(keys, before, after)
    # With equal hashes every key sat on "b" already (name tie-break), so
    # nothing moves when "a" disappears.
    assert 0 == result["moved"]
    assert {"before": 0, "after": 0, "in": 0, "out": 0} == result["nodes"]["a"]
    assert {"before": 3, "after": 3, "in": 0, "out": 0} == result["nodes"]["b"]


@pytest.mark.unit()
def test_migration_cli(tmp_path, capsys):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    keys_path = tmp_path / "keys.txt"
    before_path.write_text(json.dumps({"nodes": ["a:1", "b:1"]}))
    after_path.write_text(
        json.dumps({"nodes": ["a:1", "b:1"], "weights": {"b:1": 4}})
    )
    keys_path.write_text("\n".join("key-%d" % i for i in range(200)))

    assert 0 == main([str(before_path), str(after_path), str(keys_path)])
    out = capsys.readouterr().out
    assert "sample of 200 keys" in out
    assert "a:1" in out and "b:1" in out
    assert "describe only that sample" in out
