from __future__ import annotations

from typing import Dict, Any
import numpy as np

from .graph import GraphSpec
from .routing import RoutingPolicy, RoutingDecision, RoutingTable, ecmp_table


class HeuristicTARO(RoutingPolicy):
    """Deterministic graph policy used as a CPU baseline and smoke-test target.

    It scores eligible links with capacity, delay, utilization, queue and previous
    change. The interface is identical to a learned TARO policy, so it can be used
    while the optional PyTorch trainer is unavailable.
    """
    name = "taro-heuristic"

    def __init__(self, max_weight_change: float = 0.1, seed: int = 7):
        self.max_weight_change = max_weight_change
        self.rng = np.random.default_rng(seed)
        self.previous: RoutingTable = {}

    def decide(self, graph: GraphSpec, observation: Dict[str, Any], failed=()):
        table: RoutingTable = {}
        failed_set = set(failed)
        util = observation.get("utilization", {})
        queue = observation.get("queue", {})
        for d in range(graph.num_nodes):
            for u in range(graph.num_nodes):
                if u == d:
                    continue
                eligible = graph.eligible_out_edges(u, d, failed_set)
                if not eligible:
                    table[(u, d)] = {}
                    continue
                scores = []
                for e in eligible:
                    _, _, cap, delay = next(x for x in graph.edges if (x[0], x[1]) == e)
                    score = cap / max(1 + delay, 1e-6) * (1 - min(0.95, util.get(e, 0.0))) / (1 + queue.get(e, 0.0) / max(cap, 1e-9))
                    scores.append(max(score, 1e-6))
                probs = np.asarray(scores, dtype=float)
                probs /= probs.sum()
                old = self.previous.get((u, d), {})
                if old:
                    for i, e in enumerate(eligible):
                        probs[i] = np.clip(probs[i], old.get(e, 0.0) - self.max_weight_change, old.get(e, 0.0) + self.max_weight_change)
                    probs = np.maximum(probs, 0)
                    probs /= max(probs.sum(), 1e-9)
                table[(u, d)] = {e: float(p) for e, p in zip(eligible, probs)}
        self.previous = table
        return RoutingDecision(table)


class StableHeuristicTARO(HeuristicTARO):
    """Keep an installed split until its total-variation change is material."""
    name = "taro-stable-heuristic"

    def __init__(self, max_weight_change=0.1, update_threshold=0.1, seed=7):
        super().__init__(max_weight_change=max_weight_change, seed=seed)
        self.update_threshold = update_threshold

    def decide(self, graph, observation, failed=()):
        previous = self.previous
        proposal = super().decide(graph, observation, failed)
        for key, new in proposal.table.items():
            old = previous.get(key)
            if not old or set(old) != set(new):
                continue
            distance = 0.5 * sum(abs(old.get(edge, 0) - new.get(edge, 0))
                                 for edge in set(old) | set(new))
            if distance < self.update_threshold:
                proposal.table[key] = old.copy()
        self.previous = proposal.table
        return proposal
