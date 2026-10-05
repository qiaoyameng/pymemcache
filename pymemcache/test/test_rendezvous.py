from pymemcache.client.rendezvous import RendezvousHash
import pytest


def make_fixed_hash(scores):
    """Hash function returning a fixed value per node name."""

    def fixed_hash(value, seed):
        node = value.rsplit("-", 1)[0]
        return scores[node]

    return fixed_hash


@pytest.mark.unit()
def test_init_no_options():
    rendezvous = RendezvousHash()
    assert 0 == len(rendezvous.nodes)
    assert 1361238019 == rendezvous.hash_function("6666")
    assert rendezvous.weighted is False


@pytest.mark.unit()
def test_init():
    nodes = ["0", "1", "2"]
    rendezvous = RendezvousHash(nodes=nodes)
    assert 3 == len(rendezvous.nodes)
    assert 1361238019 == rendezvous.hash_function("6666")


@pytest.mark.unit()
def test_seed():
    rendezvous = RendezvousHash(seed=10)
    assert 2981722772 == rendezvous.hash_function("6666")


@pytest.mark.unit()
def test_add_node():
    rendezvous = RendezvousHash()
    rendezvous.add_node("1")

    assert 1 == len(rendezvous.nodes)
    rendezvous.add_node("1")

    assert 1 == len(rendezvous.nodes)
    rendezvous.add_node("2")

    assert 2 == len(rendezvous.nodes)
    rendezvous.add_node("1")

    assert 2 == len(rendezvous.nodes)


@pytest.mark.unit()
def test_remove_node():
    nodes = ["0", "1", "2"]
    rendezvous = RendezvousHash(nodes=nodes)
    rendezvous.remove_node("2")

    assert 2 == len(rendezvous.nodes)

    with pytest.raises(ValueError):
        rendezvous.remove_node("2")

    assert 2 == len(rendezvous.nodes)

    rendezvous.remove_node("1")
    assert 1 == len(rendezvous.nodes)

    rendezvous.remove_node("0")
    assert 0 == len(rendezvous.nodes)


@pytest.mark.unit()
def test_get_node():
    nodes = ["0", "1", "2"]
    rendezvous = RendezvousHash(nodes=nodes)
    assert "0" == rendezvous.get_node("ok")
    assert "1" == rendezvous.get_node("mykey")
    assert "2" == rendezvous.get_node("wat")


@pytest.mark.unit()
def test_get_node_after_removal():
    nodes = ["0", "1", "2"]
    rendezvous = RendezvousHash(nodes=nodes)
    rendezvous.remove_node("1")

    assert "0" == rendezvous.get_node("ok")
    assert "0" == rendezvous.get_node("mykey")
    assert "2" == rendezvous.get_node("wat")


@pytest.mark.unit()
def test_get_node_after_addition():
    nodes = ["0", "1", "2"]
    rendezvous = RendezvousHash(nodes=nodes)
    assert "0" == rendezvous.get_node("ok")
    assert "1" == rendezvous.get_node("mykey")
    assert "2" == rendezvous.get_node("wat")
    assert "2" == rendezvous.get_node("lol")
    rendezvous.add_node("3")

    assert "0" == rendezvous.get_node("ok")
    assert "1" == rendezvous.get_node("mykey")
    assert "2" == rendezvous.get_node("wat")
    assert "3" == rendezvous.get_node("lol")


@pytest.mark.unit()
def test_grow():
    rendezvous = RendezvousHash()

    placements = {}

    for i in range(10):
        rendezvous.add_node(str(i))
        placements[str(i)] = []

    for i in range(1000):
        node = rendezvous.get_node(str(i))
        placements[node].append(i)

    new_placements = {}

    for i in range(20):
        rendezvous.add_node(str(i))
        new_placements[str(i)] = []

    for i in range(1000):
        node = rendezvous.get_node(str(i))
        new_placements[node].append(i)

    keys = [k for sublist in placements.values() for k in sublist]
    new_keys = [k for sublist in new_placements.values() for k in sublist]
    assert sorted(keys) == sorted(new_keys)

    added = 0
    removed = 0

    for node, assignments in new_placements.items():
        after = set(assignments)
        before = set(placements.get(node, []))
        removed += len(before.difference(after))
        added += len(after.difference(before))

    assert added == removed
    assert 1062 == (added + removed)


@pytest.mark.unit()
def test_shrink():
    rendezvous = RendezvousHash()

    placements = {}
    for i in range(10):
        rendezvous.add_node(str(i))
        placements[str(i)] = []

    for i in range(1000):
        node = rendezvous.get_node(str(i))
        placements[node].append(i)

    rendezvous.remove_node("9")
    new_placements = {}
    for i in range(9):
        new_placements[str(i)] = []

    for i in range(1000):
        node = rendezvous.get_node(str(i))
        new_placements[node].append(i)

    keys = [k for sublist in placements.values() for k in sublist]
    new_keys = [k for sublist in new_placements.values() for k in sublist]
    assert sorted(keys) == sorted(new_keys)

    added = 0
    removed = 0
    for node, assignments in placements.items():
        after = set(assignments)
        before = set(new_placements.get(node, []))
        removed += len(before.difference(after))
        added += len(after.difference(before))

    assert added == removed
    assert 202 == (added + removed)


def collide(key, seed):
    return 1337


@pytest.mark.unit()
def test_rendezvous_collision():
    nodes = ["c", "b", "a"]
    rendezvous = RendezvousHash(nodes, hash_function=collide)

    for i in range(1000):
        assert "c" == rendezvous.get_node(i)


@pytest.mark.unit()
def test_rendezvous_names():
    nodes = [1, 2, 3, "a", "b", "lol.wat.com"]
    rendezvous = RendezvousHash(nodes, hash_function=collide)

    for i in range(10):
        assert "lol.wat.com" == rendezvous.get_node(i)

    nodes = [1, "a", "0"]
    rendezvous = RendezvousHash(nodes, hash_function=collide)

    for i in range(10):
        assert "a" == rendezvous.get_node(i)


@pytest.mark.unit()
def test_weighted_selection_prefers_weight_over_raw_hash():
    # Node "c" has the highest raw hash, but "a" carries the most weight.
    scores = {"a": 100, "b": 200, "c": 300}
    fixed = make_fixed_hash(scores)

    unweighted = RendezvousHash(["a", "b", "c"], hash_function=fixed)
    assert "c" == unweighted.get_node("somekey")

    weighted = RendezvousHash(
        ["a", "b", "c"],
        hash_function=fixed,
        weights={"a": 3.0, "b": 2.0, "c": 1.0},
    )
    assert weighted.weighted is True
    assert "a" == weighted.get_node("somekey")
    assert ["a", "b", "c"] == weighted.get_nodes("somekey")


@pytest.mark.unit()
def test_weighted_equal_hashes_highest_weight_wins():
    rendezvous = RendezvousHash(
        ["a", "b", "c"],
        hash_function=collide,
        weights={"a": 1.0, "b": 3.0, "c": 2.0},
    )
    for i in range(100):
        assert "b" == rendezvous.get_node(str(i))
        assert ["b", "c", "a"] == rendezvous.get_nodes(str(i))


@pytest.mark.unit()
def test_weighted_tie_break_is_stable():
    # Equal hashes and equal weights: the lexicographically largest node
    # name wins, matching the unweighted tie-breaking rule.
    rendezvous = RendezvousHash(
        ["a", "b", "c"], hash_function=collide, weights={}
    )
    for i in range(100):
        assert "c" == rendezvous.get_node(str(i))
        assert ["c", "b", "a"] == rendezvous.get_nodes(str(i))
        # Repeated calls must agree exactly.
        assert rendezvous.get_nodes(str(i)) == rendezvous.get_nodes(str(i))


@pytest.mark.unit()
def test_weighted_boundary_hash_values():
    # Hash values of 0 and 2**32 - 1 sit on the normalization boundary;
    # scoring must stay finite (no log(0)) and deterministic.
    boundary = make_fixed_hash({"low": 0, "high": 2**32 - 1})
    rendezvous = RendezvousHash(
        ["low", "high"], hash_function=boundary, weights={}
    )
    for i in range(100):
        winner = rendezvous.get_node(str(i))
        assert "high" == winner
        assert ["high", "low"] == rendezvous.get_nodes(str(i))

    zero = make_fixed_hash({"a": 0, "b": 0})
    rendezvous = RendezvousHash(["a", "b"], hash_function=zero, weights={})
    assert "b" == rendezvous.get_node("key")

    # Out-of-range hash values are normalized into the 32-bit space.
    wrapped = make_fixed_hash({"a": -1, "b": 2**32})
    rendezvous = RendezvousHash(["a", "b"], hash_function=wrapped, weights={})
    assert rendezvous.get_node("key") == rendezvous.get_node("key")


@pytest.mark.unit()
def test_weight_increase_only_takes_over_keys():
    nodes = ["n0", "n1", "n2", "n3"]
    rendezvous = RendezvousHash(nodes, weights={node: 1.0 for node in nodes})

    keys = [str(i) for i in range(2000)]
    before = {key: rendezvous.get_node(key) for key in keys}

    rendezvous.set_weight("n1", 4.0)
    after = {key: rendezvous.get_node(key) for key in keys}

    moved = 0
    for key in keys:
        if before[key] != after[key]:
            moved += 1
            # Keys may only move *to* the node whose weight grew; the
            # remaining nodes must never swap keys among themselves.
            assert "n1" == after[key]
            assert "n1" != before[key]
    assert moved > 0


@pytest.mark.unit()
def test_weighted_add_node_only_takes_over_keys():
    nodes = ["n0", "n1", "n2"]
    rendezvous = RendezvousHash(nodes, weights={node: 1.0 for node in nodes})

    keys = [str(i) for i in range(2000)]
    before = {key: rendezvous.get_node(key) for key in keys}

    rendezvous.add_node("n3", weight=2.0)
    after = {key: rendezvous.get_node(key) for key in keys}

    moved = 0
    for key in keys:
        if before[key] != after[key]:
            moved += 1
            assert "n3" == after[key]
    assert moved > 0


@pytest.mark.unit()
def test_weighted_remove_node_redistributes_to_survivors():
    nodes = ["n0", "n1", "n2"]
    rendezvous = RendezvousHash(nodes, weights={node: 1.0 for node in nodes})

    keys = [str(i) for i in range(1000)]
    before = {key: rendezvous.get_node(key) for key in keys}

    rendezvous.remove_node("n1")
    after = {key: rendezvous.get_node(key) for key in keys}

    for key in keys:
        assert "n1" != after[key]
        if before[key] != "n1":
            # Keys not owned by the removed node must not move.
            assert before[key] == after[key]


@pytest.mark.unit()
def test_add_node_duplicate_and_weight_updates():
    rendezvous = RendezvousHash()
    rendezvous.add_node("1")
    rendezvous.add_node("1")
    assert 1 == len(rendezvous.nodes)
    assert rendezvous.weighted is False

    rendezvous.add_node("1", weight=2.0)
    assert 1 == len(rendezvous.nodes)
    assert rendezvous.weighted is True
    assert {"1": 2.0} == rendezvous.weights

    # Re-adding without a weight keeps the existing weight.
    rendezvous.add_node("1")
    assert {"1": 2.0} == rendezvous.weights

    # Re-adding with a weight updates it.
    rendezvous.add_node("1", weight=5.0)
    assert {"1": 5.0} == rendezvous.weights

    rendezvous.add_node("2")
    assert {"1": 5.0, "2": 1.0} == rendezvous.weights


@pytest.mark.unit()
def test_invalid_weights_rejected():
    rendezvous = RendezvousHash(["a"])

    for bad_weight in (0, -1, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rendezvous.add_node("b", weight=bad_weight)
        with pytest.raises(ValueError):
            rendezvous.set_weight("a", bad_weight)
        with pytest.raises(ValueError):
            RendezvousHash(["a"], weights={"a": bad_weight})

    for not_a_number in ("heavy", None.__class__, object()):
        with pytest.raises(TypeError):
            rendezvous.add_node("b", weight=not_a_number)

    with pytest.raises(TypeError):
        rendezvous.add_node("b", weight=True)

    # Failed attempts must not enable weighted mode or add nodes.
    assert rendezvous.weighted is False
    assert ["a"] == rendezvous.nodes


@pytest.mark.unit()
def test_set_weight_unknown_node():
    rendezvous = RendezvousHash(["a"], weights={})
    with pytest.raises(ValueError):
        rendezvous.set_weight("missing", 2.0)


@pytest.mark.unit()
def test_weighted_remove_node():
    rendezvous = RendezvousHash(["a", "b"], weights={"a": 2.0, "b": 1.0})
    rendezvous.remove_node("a")
    assert ["b"] == rendezvous.nodes
    assert {"b": 1.0} == rendezvous.weights

    with pytest.raises(ValueError):
        rendezvous.remove_node("a")

    # A node re-added after removal falls back to the default weight.
    rendezvous.add_node("a")
    assert {"b": 1.0, "a": 1.0} == rendezvous.weights


@pytest.mark.unit()
def test_get_nodes_unweighted_matches_get_node():
    rendezvous = RendezvousHash(["0", "1", "2", "3"])
    for i in range(100):
        key = str(i)
        ranked = rendezvous.get_nodes(key)
        assert 4 == len(ranked)
        assert sorted(ranked) == ["0", "1", "2", "3"]
        assert ranked[0] == rendezvous.get_node(key)
        assert ranked == rendezvous.get_nodes(key)


@pytest.mark.unit()
def test_weighted_empty_hasher():
    rendezvous = RendezvousHash(weights={})
    assert rendezvous.get_nodes("key") == []
    assert rendezvous.get_node("key") is None
