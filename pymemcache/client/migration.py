"""
Offline estimation of key movement between two cluster topologies.

Given a finite sample of keys and two topologies (before/after), this
module reports, per node, how many sampled keys flow in and out when the
:class:`~pymemcache.client.rendezvous.RendezvousHash` routing changes.

The result is computed over the provided key sample only. It is an
estimate of, not a measurement of, the full key population: ratios may
not hold for keys outside the sample.

Run as a script::

    python -m pymemcache.client.migration before.json after.json \
        --keys keys.txt

where each topology file is JSON, either a list of node names
(unweighted) or an object mapping node name to positive weight, and the
key file holds one key per line. Use ``--sample N`` instead of
``--keys`` to synthesize ``N`` keys named ``key-0`` .. ``key-N-1``.
"""

import argparse
import json
import sys

from pymemcache.client.rendezvous import RendezvousHash

#: Disclaimer attached to every report: the numbers describe the sample,
#: not the whole key population.
SAMPLE_DISCLAIMER = (
    "Estimate computed from a finite sample of {sample_size} keys; "
    "it does not represent the full key population."
)


def _build_hasher(topology, seed):
    if isinstance(topology, dict):
        return RendezvousHash(
            nodes=list(topology), weights=dict(topology), seed=seed
        )
    return RendezvousHash(nodes=list(topology), seed=seed)


def estimate_migration(keys, before, after, seed=0):
    """
    Estimate per-node key movement between two topologies.

    Args:
      keys: finite iterable of sample keys.
      before: node list (unweighted) or ``{node: weight}`` mapping.
      after: node list (unweighted) or ``{node: weight}`` mapping.
      seed: hash seed used for both topologies.

    Returns a dict with ``sample_size``, ``moved_keys``, a per-node
    ``nodes`` breakdown (``before``/``after`` counts plus ``inflow`` and
    ``outflow``), and a ``disclaimer`` noting that the figures describe
    only the sample.
    """
    keys = list(keys)
    before_hasher = _build_hasher(before, seed)
    after_hasher = _build_hasher(after, seed)

    nodes = sorted(
        {str(node) for node in before_hasher.nodes}
        | {str(node) for node in after_hasher.nodes}
    )
    report = {
        node: {"before": 0, "after": 0, "inflow": 0, "outflow": 0}
        for node in nodes
    }

    moved_keys = 0
    for key in keys:
        old_node = before_hasher.get_node(key)
        new_node = after_hasher.get_node(key)
        if old_node is not None:
            report[str(old_node)]["before"] += 1
        if new_node is not None:
            report[str(new_node)]["after"] += 1
        if old_node != new_node:
            moved_keys += 1
            if old_node is not None:
                report[str(old_node)]["outflow"] += 1
            if new_node is not None:
                report[str(new_node)]["inflow"] += 1

    return {
        "sample_size": len(keys),
        "moved_keys": moved_keys,
        "nodes": report,
        "disclaimer": SAMPLE_DISCLAIMER.format(sample_size=len(keys)),
    }


def format_report(result):
    lines = [
        "Key migration estimate (sample of %d keys)"
        % result["sample_size"],
        "moved keys: %d" % result["moved_keys"],
        "per-node breakdown:",
    ]
    for node, stats in result["nodes"].items():
        lines.append(
            "  %s: before=%d after=%d inflow=%d outflow=%d"
            % (
                node,
                stats["before"],
                stats["after"],
                stats["inflow"],
                stats["outflow"],
            )
        )
    lines.append("note: %s" % result["disclaimer"])
    return "\n".join(lines)


def _load_topology(path):
    with open(path, encoding="utf-8") as topology_file:
        return json.load(topology_file)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pymemcache.client.migration",
        description=(
            "Estimate key movement between two RendezvousHash topologies "
            "over a finite key sample."
        ),
    )
    parser.add_argument("before", help="JSON file with the current topology")
    parser.add_argument("after", help="JSON file with the target topology")
    key_source = parser.add_mutually_exclusive_group(required=True)
    key_source.add_argument(
        "--keys", help="file with one sample key per line"
    )
    key_source.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="synthesize N sample keys named key-0 .. key-N-1",
    )
    parser.add_argument("--seed", type=int, default=0, help="hash seed")
    args = parser.parse_args(argv)

    if args.keys is not None:
        with open(args.keys, encoding="utf-8") as keys_file:
            keys = [line.rstrip("\n") for line in keys_file if line.strip()]
    else:
        keys = ["key-%d" % i for i in range(args.sample)]

    result = estimate_migration(
        keys,
        _load_topology(args.before),
        _load_topology(args.after),
        seed=args.seed,
    )
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
