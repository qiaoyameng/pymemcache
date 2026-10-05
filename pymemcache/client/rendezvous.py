import math

from pymemcache.client.murmur3 import murmur3_32


#: Modulus used to normalize 32-bit hash values into the open unit interval.
_HASH_MODULUS = 1 << 32


class RendezvousHash:
    """
    Implements the Highest Random Weight (HRW) hashing algorithm most
    commonly referred to as rendezvous hashing.

    Originally developed as part of python-clandestined.

    Copyright (c) 2014 Ernest W. Durbin III

    Weighted mode
    -------------
    Passing ``weights`` to the constructor (even an empty mapping) enables
    weighted rendezvous hashing. Each node then carries a positive weight
    and the score for a ``(node, key)`` pair becomes ``-weight / log(u)``
    where ``u`` is the hash normalized into the open interval ``(0, 1)``.
    The node with the highest score wins, so a node's expected share of
    keys is proportional to its weight.

    Weighted mode is strictly monotonic: adding a node or raising a node's
    weight can only cause that node to take keys over from others, it never
    causes keys to move between two unrelated nodes.

    When ``weights`` is not given, routing behaves exactly as it always
    has (raw hash comparison) and existing placements are preserved.
    """

    def __init__(self, nodes=None, seed=0, hash_function=murmur3_32, weights=None):
        """
        Constructor.

        Args:
          nodes: optional iterable of initial nodes.
          seed: optional seed passed to ``hash_function``.
          hash_function: optional ``f(value, seed) -> int`` hash function.
          weights: optional mapping of ``node -> positive weight``. Supplying
                   any mapping (including ``{}``) enables weighted mode;
                   nodes without an explicit weight default to ``1.0``.
        """
        self.nodes = []
        self.seed = seed
        self.hash_function = lambda x: hash_function(x, seed)
        self.weighted = weights is not None
        self.weights = {}
        if self.weighted:
            initial = list(nodes) if nodes is not None else []
            for node in initial:
                self.add_node(node)
            for node, weight in (weights or {}).items():
                if node in self.nodes:
                    self.set_weight(node, weight)
                else:
                    self.add_node(node, weight=weight)
        elif nodes is not None:
            self.nodes = nodes

    @staticmethod
    def _validate_weight(weight):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("weight must be a real number, got %r" % (weight,))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                "weight must be a positive finite number, got %r" % (weight,)
            )

    def add_node(self, node, weight=None):
        """
        Add ``node`` to the ring. Adding a node that is already present is
        a no-op and never changes its weight; use :meth:`set_weight` for
        that.

        In weighted mode ``weight`` defaults to ``1.0``. In unweighted
        (legacy) mode passing a weight is an error.
        """
        if self.weighted:
            if weight is None:
                weight = 1.0
            self._validate_weight(weight)
            if node not in self.nodes:
                self.nodes.append(node)
                self.weights[node] = float(weight)
        else:
            if weight is not None:
                raise ValueError(
                    "weights are not enabled on this RendezvousHash; pass "
                    "weights=... to the constructor to enable weighted mode"
                )
            if node not in self.nodes:
                self.nodes.append(node)

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            self.weights.pop(node, None)
        else:
            raise ValueError("No such node %s to remove" % (node))

    def set_weight(self, node, weight):
        """
        Change the weight of an existing node. Raising a node's weight can
        only move keys *to* that node, never between other nodes.
        """
        if not self.weighted:
            raise ValueError(
                "weights are not enabled on this RendezvousHash; pass "
                "weights=... to the constructor to enable weighted mode"
            )
        self._validate_weight(weight)
        if node not in self.nodes:
            raise ValueError("No such node %s to reweight" % (node))
        self.weights[node] = float(weight)

    def _score(self, node, key):
        raw = self.hash_function(f"{node}-{key}")
        if not self.weighted:
            return raw
        # Normalize into the open interval (0, 1): the modulo tames signed
        # or oversized hash outputs, and the +1 offsets keep ``u`` away
        # from both 0 (log(0) is undefined) and 1 (log(1) == 0 would
        # divide by zero).
        u = (raw % _HASH_MODULUS + 1) / (_HASH_MODULUS + 1)
        return -self.weights[node] / math.log(u)

    def get_nodes(self, key, count=None):
        """
        Return the nodes ranked by preference for ``key``, best first.

        The ranking is fully deterministic: ties on score are broken by
        the node's string representation (lexicographically largest wins,
        matching the historical collision behavior), so repeated calls
        always produce the same order regardless of float quirks.

        Args:
          key: the key to rank nodes for.
          count: optional maximum number of nodes to return; ``None``
                 (the default) ranks every node.
        """
        scored = [(self._score(node, key), node) for node in self.nodes]
        # Sort by node name (descending) first so that the stable score
        # sort below breaks score ties by name, deterministically.
        scored.sort(key=lambda item: str(item[1]), reverse=True)
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = [node for _, node in scored]
        if count is None:
            return ranked
        return ranked[:count]

    def get_node(self, key):
        if self.weighted:
            ranked = self.get_nodes(key, count=1)
            return ranked[0] if ranked else None

        high_score = -1
        winner = None

        for node in self.nodes:
            score = self.hash_function(f"{node}-{key}")

            if score > high_score:
                (high_score, winner) = (score, node)
            elif score == high_score:
                (high_score, winner) = (score, max(str(node), str(winner)))

        return winner
