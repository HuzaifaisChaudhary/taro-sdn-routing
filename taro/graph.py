from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict, Optional
import json
from pathlib import Path
import math
import networkx as nx
import numpy as np


@dataclass
class GraphSpec:
    """Directed topology with physical links represented in both directions."""

    num_nodes: int
    edges: List[Tuple[int, int, float, float]]
    name: str = "synthetic"
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_nodes <= 1:
            raise ValueError("A graph must contain at least two nodes")
        for u, v, capacity, delay in self.edges:
            if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
                raise ValueError(f"edge {(u, v)} has an invalid node")
            if u == v or capacity <= 0 or delay < 0:
                raise ValueError(f"invalid edge {(u, v, capacity, delay)}")

    @property
    def edge_pairs(self) -> List[Tuple[int, int]]:
        return [(u, v) for u, v, _, _ in self.edges]

    def to_networkx(self, failed: Optional[Iterable[Tuple[int, int]]] = None) -> nx.DiGraph:
        failed_set = set(failed or [])
        g = nx.DiGraph()
        g.add_nodes_from(range(self.num_nodes))
        for u, v, capacity, delay in self.edges:
            if (u, v) not in failed_set:
                g.add_edge(u, v, capacity=capacity, delay=delay)
        return g

    def capacities(self) -> Dict[Tuple[int, int], float]:
        return {(u, v): c for u, v, c, _ in self.edges}

    def delays(self) -> Dict[Tuple[int, int], float]:
        return {(u, v): d for u, v, _, d in self.edges}

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_nodes": self.num_nodes, "edges": self.edges,
                       "name": self.name, "metadata": self.metadata}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GraphSpec":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["num_nodes"], [tuple(e) for e in d["edges"]], d.get("name", "loaded"), d.get("metadata", {}))

    @classmethod
    def from_graphml(cls, path: str, name: str | None = None, default_capacity: float = 100.0,
                     default_delay: float = 1.0) -> "GraphSpec":
        graph = nx.read_graphml(path)
        nodes = list(graph.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        edges = []
        for u, v, data in graph.edges(data=True):
            cap = float(data.get("capacity", data.get("Bandwidth", default_capacity)))
            delay = float(data.get("delay", data.get("Delay", default_delay)))
            edges.extend([(index[u], index[v], cap, delay), (index[v], index[u], cap, delay)])
        return cls(len(nodes), edges, name or Path(path).stem, {"source_file": str(path), "format": "graphml"})

    @classmethod
    def ring(cls, n: int = 6, capacity: float = 100.0, delay: float = 1.0) -> "GraphSpec":
        edges = []
        for i in range(n):
            j = (i + 1) % n
            edges.extend([(i, j, capacity, delay), (j, i, capacity, delay)])
        return cls(n, edges, name=f"ring-{n}")

    @classmethod
    def line(cls, n: int = 4, capacity: float = 100.0, delay: float = 1.0) -> "GraphSpec":
        edges = []
        for i in range(n - 1):
            edges.extend([(i, i + 1, capacity, delay), (i + 1, i, capacity, delay)])
        return cls(n, edges, name=f"line-{n}")

    @classmethod
    def random(cls, n: int, seed: int, p: float = 0.25) -> "GraphSpec":
        rng = np.random.default_rng(seed)
        undirected = nx.gnp_random_graph(n, p, seed=seed)
        if not nx.is_connected(undirected):
            undirected = nx.minimum_spanning_tree(nx.complete_graph(n), algorithm="kruskal")
            extra = nx.gnp_random_graph(n, p, seed=seed)
            undirected.add_edges_from(extra.edges())
        edges = []
        for u, v in undirected.edges():
            cap = float(rng.choice([50, 100, 200]))
            delay = float(rng.uniform(0.5, 5.0))
            edges.extend([(u, v, cap, delay), (v, u, cap, delay)])
        return cls(n, edges, name=f"random-{n}-{seed}")

    def shortest_distance(self, destination: int, failed: Iterable[Tuple[int, int]] = ()) -> Dict[int, float]:
        g = self.to_networkx(failed)
        try:
            return dict(nx.single_source_shortest_path_length(g.reverse(copy=False), destination))
        except nx.NetworkXError:
            return {}

    def eligible_out_edges(self, node: int, destination: int, failed: Iterable[Tuple[int, int]] = ()) -> List[Tuple[int, int]]:
        dist = self.shortest_distance(destination, failed)
        if node == destination:
            return []
        return [(u, v) for u, v in self.edge_pairs if u == node and (u, v) not in set(failed)
                and dist.get(v, math.inf) < dist.get(node, math.inf)]

    def validate_connected(self) -> None:
        g = self.to_networkx()
        if not nx.is_strongly_connected(g):
            raise ValueError(f"{self.name} must be strongly connected for the default scenarios")
