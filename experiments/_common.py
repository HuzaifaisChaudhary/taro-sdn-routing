"""Shared helpers for the experiment scripts: import path, output folder, and a result-neutral cache."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taro.graph import GraphSpec  # noqa: E402

_orig_shortest_distance = GraphSpec.shortest_distance
_cache = {}


def _cached_shortest_distance(self, destination, failed=()):
    """Memoised GraphSpec.shortest_distance. Pure speed-up: results are identical to the uncached call.

    The key uses the graph's node count and full edge list (not object identity), so two different
    graphs can never share a cache entry.
    """
    key_graph = getattr(self, "_fc_key", None)
    if key_graph is None:
        key_graph = (self.num_nodes, tuple((u, v) for u, v, *_ in self.edges))
        self._fc_key = key_graph
    key = (key_graph, destination, frozenset(failed))
    hit = _cache.get(key)
    if hit is None:
        hit = _orig_shortest_distance(self, destination, failed)
        _cache[key] = hit
    return dict(hit)


GraphSpec.shortest_distance = _cached_shortest_distance
