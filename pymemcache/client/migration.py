"""
Offline estimation of key movement between two cluster topologies.

Given a finite sample of keys and two :class:`RendezvousHash` topologies
(typically an unweighted "before" and a weighted "after"), this module
reports, per node, how many sampled keys flow in or out when moving from
one topology to the other.

The numbers describe *only the provided sample*. They are an estimate of
what the full keyspace would do, never a measurement of it.

Run from the command line::

    python -m pymemcache.client.migration before.json after.json keys.txt

Each topology file is JSON, e.g.::

    {"nodes": ["10.0.0.1:11211", "10.0.0.2:11211"],
     "weights": {"10.0.0.2:11211": 3},
     "seed": 0}

``weights`` may be omitted or null for unweighted (legacy) routing. The
keys file holds one key per line.
"""

import argparse
import json
import sys

from pymemcache.client.rendezvous import RendezvousHash

SAMPLE_DISCLAIMER = (
    "Counts are computed from a finite sample of {sample_size} keys and "
    "describe only that sample; they are an estimate for, not a "
    "measurement of, the full keyspace."
)


def build_hasher(spec):
    """
    Build a :class:`RendezvousHash` from a topology spec mapping with
    optional ``nodes``, ``weights`` and ``seed`` entries.
    """
    if not isinstance(spec, dict):
        raise TypeError("topology spec must be a mapping, got %r" % (spec,))
    return RendezvousHash(
        nodes=spec.get("nodes"),
        seed=spec.get("seed", 0),
        weights=spec.get("weights"),
    )


def estimate_migration(keys, before, after):
    """
    Compare two topologies over a finite key sample.

    Args:
      keys: iterable of sample keys (consumed once).
      before: :class:`RendezvousHash` for the current topology.
      after: :class:`RendezvousHash` for the target topology.

    Returns a dict with ``sample_size``, total ``moved`` keys, a per-node
    breakdown under ``nodes`` (``before``/``after`` counts plus ``in`` and
    ``out`` flows) and a ``note`` stressing the result is sample-based.
    """
    keys = list(keys)
    all_nodes = list(dict.fromkeys(list(before.nodes) + list(after.nodes)))
    stats = {
        node: {"before": 0, "after": 0, "in": 0, "out": 0} for node in all_nodes
    }
    moved = 0
    for key in keys:
        old = before.get_node(key)
        new = after.get_node(key)
        if old is not None:
            stats[old]["before"] += 1
        if new is not None:
            stats[new]["after"] += 1
        if old != new:
            moved += 1
            if old is not None:
                stats[old]["out"] += 1
            if new is not None:
                stats[new]["in"] += 1
    return {
        "sample_size": len(keys),
        "moved": moved,
        "nodes": stats,
        "note": SAMPLE_DISCLAIMER.format(sample_size=len(keys)),
    }


def format_report(result):
    """Render the result of :func:`estimate_migration` as plain text."""
    lines = [
        "Key migration estimate (sample of %d keys)" % result["sample_size"],
        "keys that change node in this sample: %d" % result["moved"],
        "",
        "%-30s %8s %8s %8s %8s" % ("node", "before", "after", "in", "out"),
    ]
    for node, stats in result["nodes"].items():
        lines.append(
            "%-30s %8d %8d %8d %8d"
            % (node, stats["before"], stats["after"], stats["in"], stats["out"])
        )
    lines += ["", "note: " + result["note"]]
    return "\n".join(lines)


def _load_topology(path):
    with open(path, encoding="utf-8") as handle:
        return build_hasher(json.load(handle))


def _load_keys(path):
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m pymemcache.client.migration",
        description=(
            "Estimate per-node key movement between two rendezvous-hash "
            "topologies over a finite key sample."
        ),
    )
    parser.add_argument("before", help="JSON topology file for the current layout")
    parser.add_argument("after", help="JSON topology file for the target layout")
    parser.add_argument("keys", help="text file with one sample key per line")
    args = parser.parse_args(argv)

    before = _load_topology(args.before)
    after = _load_topology(args.after)
    keys = _load_keys(args.keys)

    result = estimate_migration(keys, before, after)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
