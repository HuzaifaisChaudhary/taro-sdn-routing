from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable
import json
import numpy as np

from .graph import GraphSpec


@dataclass
class Scenario:
    topology: GraphSpec
    demands: np.ndarray  # [time, source, destination]
    failures: List[List[Tuple[int, int]]]
    name: str = "scenario"
    metadata: Dict[str, str] | None = None

    def to_manifest(self, path: str) -> None:
        payload = {"name": self.name, "metadata": self.metadata or {},
                   "topology": {"name": self.topology.name, "num_nodes": self.topology.num_nodes,
                                "edges": self.topology.edges},
                   "shape": list(self.demands.shape), "failures": self.failures}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def poisson_demands(n: int, horizon: int, seed: int, mean: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.poisson(mean, size=(horizon, n, n)).astype(float)
    for t in range(horizon):
        np.fill_diagonal(d[t], 0.0)
    return d


def heavy_tail_bursts(demands: np.ndarray, seed: int, fraction: float = 0.1, multipliers=(2.0, 4.0)) -> np.ndarray:
    out = demands.copy()
    rng = np.random.default_rng(seed)
    n = out.shape[1]
    pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    k = max(1, int(len(pairs) * fraction))
    chosen = rng.choice(len(pairs), size=k, replace=False)
    for idx in chosen:
        s, t = pairs[int(idx)]
        start = int(rng.integers(0, max(1, out.shape[0] - 3)))
        width = int(rng.integers(1, max(2, out.shape[0] // 4)))
        out[start:start + width, s, t] *= float(rng.choice(multipliers))
    return out


def failure_schedule(graph: GraphSpec, horizon: int, seed: int, percentage: float = 0.1) -> List[List[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    physical = [(u, v) for u, v in graph.edge_pairs if u < v]
    count = max(1, int(len(physical) * percentage))
    selected = [tuple(x) for x in rng.choice(np.array(physical, dtype=int), size=count, replace=False)]
    failed = []
    for u, v in selected:
        failed.extend([(u, v), (v, u)])
    event = [failed if horizon // 3 <= t < 2 * horizon // 3 else [] for t in range(horizon)]
    return event


def make_scenario(graph: GraphSpec, horizon: int = 60, seed: int = 7, burst: bool = False,
                  failures: float = 0.0, load: float = 1.0) -> Scenario:
    d = poisson_demands(graph.num_nodes, horizon, seed, mean=8.0 * load)
    if burst:
        d = heavy_tail_bursts(d, seed + 101)
    fs = failure_schedule(graph, horizon, seed + 202, failures) if failures else [[] for _ in range(horizon)]
    return Scenario(graph, d, fs, name=f"{graph.name}-seed{seed}", metadata={"burst": str(burst), "load": str(load)})


def udp_flood_demands(demands: np.ndarray, seed: int, fraction: float = 0.2, multiplier: float = 8.0) -> np.ndarray:
    out = demands.copy(); rng = np.random.default_rng(seed); n = out.shape[1]
    pairs = [(s, d) for s in range(n) for d in range(n) if s != d]
    for idx in rng.choice(len(pairs), max(1, int(len(pairs) * fraction)), replace=False):
        s, d = pairs[int(idx)]; start = int(rng.integers(0, max(1, len(out) // 2))); out[start:, s, d] *= multiplier
    return out


def corrupt_telemetry(demands: np.ndarray, seed: int, fraction: float = 0.1) -> np.ndarray:
    out = demands.copy(); rng = np.random.default_rng(seed); mask = rng.random(out.shape) < fraction
    out[mask] *= rng.uniform(0.5, 1.5, size=mask.sum()); return out
