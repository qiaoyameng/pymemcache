import math

from pymemcache.client.murmur3 import murmur3_32

#: Modulus used to normalize 32-bit hash values into the unit interval.
_HASH_MODULUS = 2**32


def _validate_weight(node, weight):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError(
            "Weight for node %r must be a number, got %r" % (node, weight)
        )
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            "Weight for node %r must be a positive finite number, got %r"
            % (node, weight)
        )
    return float(weight)


class RendezvousHash:
    """
    Implements the Highest Random Weight (HRW) hashing algorithm most
    commonly referred to as rendezvous hashing.

    Originally developed as part of python-clandestined.

    Copyright (c) 2014 Ernest W. Durbin III

    Nodes may optionally carry a positive weight so that clusters with
    heterogeneous capacities receive a proportional share of keys. Weighted
    mode is enabled explicitly by passing ``weights`` to the constructor or a
    ``weight`` to :meth:`add_node`/:meth:`set_weight`; when no weight is ever
    specified the legacy unweighted routing is used unchanged.
    """

    def __init__(self, nodes=None, seed=0, hash_function=murmur3_32, weights=None):
        """
        Constructor.

        Args:
          nodes: optional iterable of node names.
          seed: seed passed to ``hash_function``.
          hash_function: callable ``(value, seed) -> 32-bit int``.
          weights: optional dict mapping node name to a positive finite
                   weight. Passing any mapping (even empty) enables weighted
                   mode. Nodes without an entry default to weight 1.0, and
                   entries for nodes added later are honored once the node
                   joins.
        """
        self.nodes = []
        self.seed = seed
        if nodes is not None:
            self.nodes = list(nodes)
        self.hash_function = lambda x: hash_function(x, seed)
        self._weights = {}
        self._weighted = weights is not None
        if weights is not None:
            for node, weight in weights.items():
                self._weights[node] = _validate_weight(node, weight)

    @property
    def weighted(self):
        """Whether weighted routing has been explicitly enabled."""
        return self._weighted

    @property
    def weights(self):
        """Effective weight of every current node (defaulting to 1.0)."""
        return {node: self._weights.get(node, 1.0) for node in self.nodes}

    def add_node(self, node, weight=None):
        if weight is not None:
            weight = _validate_weight(node, weight)
            self._weighted = True
        if node not in self.nodes:
            self.nodes.append(node)
        if weight is not None:
            self._weights[node] = weight

    def set_weight(self, node, weight):
        """Set (or update) the weight of an existing node, enabling
        weighted mode. Increasing a weight can only cause the node to take
        keys over from other nodes, never swap keys between them."""
        if node not in self.nodes:
            raise ValueError("No such node %s to weight" % (node))
        self._weights[node] = _validate_weight(node, weight)
        self._weighted = True

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            self._weights.pop(node, None)
        else:
            raise ValueError("No such node %s to remove" % (node))

    def _weighted_score(self, node, key):
        # Normalize the hash into the open interval (0, 1). The +0.5 offset
        # keeps ``u`` away from both 0 and 1 so that ``math.log`` is always
        # finite and non-zero, even for hash values of 0 or 2**32 - 1.
        hash_value = self.hash_function(f"{node}-{key}") % _HASH_MODULUS
        u = (hash_value + 0.5) / _HASH_MODULUS
        return -self._weights.get(node, 1.0) / math.log(u)

    def get_nodes(self, key):
        """
        Return every node ranked from most to least preferred for ``key``.

        The ranking is a total, deterministic order: scores are compared
        first and exact ties are broken by the node's string representation
        (lexicographically largest first), matching the tie-breaking of
        :meth:`get_node`. Nodes with equal score and equal string form keep
        their insertion order, so repeated calls always agree.
        """
        if self._weighted:
            scored = [
                (self._weighted_score(node, key), str(node), node)
                for node in self.nodes
            ]
        else:
            scored = [
                (self.hash_function(f"{node}-{key}"), str(node), node)
                for node in self.nodes
            ]
        # Sort only on (score, str(node)); the sort is stable, so nodes that
        # compare equal on both keep their insertion order deterministically.
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [node for _, _, node in scored]

    def get_node(self, key):
        if self._weighted:
            ranked = self.get_nodes(key)
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
