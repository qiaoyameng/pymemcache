import json

import pytest

from pymemcache.client.migration import (
    estimate_migration,
    format_report,
    main,
)


@pytest.mark.unit()
def test_estimate_migration_node_addition():
    keys = ["key-%d" % i for i in range(500)]
    result = estimate_migration(
        keys, ["n0", "n1", "n2"], ["n0", "n1", "n2", "n3"]
    )

    assert result["sample_size"] == 500
    nodes = result["nodes"]
    assert sum(stats["before"] for stats in nodes.values()) == 500
    assert sum(stats["after"] for stats in nodes.values()) == 500

    # The new node only receives keys; existing nodes only lose keys.
    assert nodes["n3"]["before"] == 0
    assert nodes["n3"]["inflow"] == result["moved_keys"]
    assert nodes["n3"]["outflow"] == 0
    assert result["moved_keys"] > 0
    for node in ("n0", "n1", "n2"):
        assert nodes[node]["inflow"] == 0
        assert nodes[node]["after"] <= nodes[node]["before"]
    assert result["moved_keys"] == sum(
        stats["outflow"] for stats in nodes.values()
    )

    # The report must not present the sample as the full population.
    assert "sample" in result["disclaimer"]
    assert "does not represent the full key population" in result["disclaimer"]


@pytest.mark.unit()
def test_estimate_migration_weight_increase():
    keys = ["key-%d" % i for i in range(500)]
    result = estimate_migration(
        keys, {"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 4.0}
    )

    nodes = result["nodes"]
    # Keys only flow towards the node whose weight increased.
    assert nodes["a"]["inflow"] == 0
    assert nodes["b"]["outflow"] == 0
    assert nodes["b"]["inflow"] == nodes["a"]["outflow"]
    assert result["moved_keys"] == nodes["b"]["inflow"]
    assert result["moved_keys"] > 0


@pytest.mark.unit()
def test_estimate_migration_identical_topologies():
    keys = ["key-%d" % i for i in range(100)]
    result = estimate_migration(keys, ["n0", "n1"], ["n0", "n1"])
    assert result["moved_keys"] == 0
    for stats in result["nodes"].values():
        assert stats["inflow"] == 0
        assert stats["outflow"] == 0
        assert stats["before"] == stats["after"]


@pytest.mark.unit()
def test_format_report_includes_disclaimer():
    result = estimate_migration(["k1", "k2"], ["a"], ["a", "b"])
    text = format_report(result)
    assert "sample of 2 keys" in text
    assert "does not represent the full key population" in text
    assert "a: before=2" in text
    assert "inflow=" in text and "outflow=" in text


@pytest.mark.unit()
def test_main_with_key_file(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    keys = tmp_path / "keys.txt"
    before.write_text(json.dumps(["n0", "n1"]))
    after.write_text(json.dumps({"n0": 1.0, "n1": 1.0, "n2": 2.0}))
    keys.write_text("alpha\nbeta\ngamma\n")

    assert main([str(before), str(after), "--keys", str(keys)]) == 0
    out = capsys.readouterr().out
    assert "sample of 3 keys" in out
    assert "does not represent the full key population" in out
    assert "n2: before=0" in out


@pytest.mark.unit()
def test_main_with_synthesized_sample(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(["n0", "n1"]))
    after.write_text(json.dumps(["n0", "n1", "n2"]))

    assert main([str(before), str(after), "--sample", "200"]) == 0
    out = capsys.readouterr().out
    assert "sample of 200 keys" in out
    assert "n2: before=0" in out
