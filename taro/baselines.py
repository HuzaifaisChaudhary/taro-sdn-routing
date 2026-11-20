"""Phase 2 baseline policies and an LP utilization lower bound."""
from __future__ import annotations

from typing import Any, Dict
import heapq
import numpy as np
import networkx as nx

from .graph import GraphSpec
from .routing import RoutingPolicy, RoutingDecision, RoutingTable, ecmp_table, load_aware_table


class DQNPolicy(RoutingPolicy):
    """Masked tabular Q-routing baseline with a shared edge score table."""
    name = "dqn-routing"

    def __init__(self, seed: int = 7, epsilon: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.q: Dict[tuple, float] = {}

    def decide(self, graph, observation, failed=()):
        table = {}
        util = observation.get("utilization", {})
        for d in range(graph.num_nodes):
            for u in range(graph.num_nodes):
                if u == d:
                    continue
                eligible = graph.eligible_out_edges(u, d, failed)
                if not eligible:
                    table[(u, d)] = {}
                    continue
                if self.rng.random() < self.epsilon:
                    edge = eligible[int(self.rng.integers(len(eligible)))]
                else:
                    edge = min(eligible, key=lambda e: self.q.get((u, d, e), 0.0) + util.get(e, 0.0))
                table[(u, d)] = {edge: 1.0}
        return RoutingDecision(table)


class DDPGPolicy(RoutingPolicy):
    """Continuous link-weight traffic-engineering baseline.

    Link weights are updated from utilization and then converted to shortest paths,
    matching the action space used by the DDPG-style Phase 1 baseline.
    """
    name = "ddpg-link-weight"

    def __init__(self, step: float = 0.2):
        self.step = step
        self.weights: Dict[tuple, float] = {}

    def decide(self, graph, observation, failed=()):
        g = graph.to_networkx(failed)
        util = observation.get("utilization", {})
        for u, v in g.edges:
            e = (u, v)
            self.weights[e] = max(0.01, self.weights.get(e, 1.0) + self.step * util.get(e, 0.0))
            g[u][v]["cost"] = self.weights[e] * max(g[u][v].get("delay", 1.0), 0.01)
        table = {}
        for d in range(graph.num_nodes):
            for u in range(graph.num_nodes):
                if u == d:
                    continue
                try:
                    path = nx.shortest_path(g, u, d, weight="cost")
                    table[(u, d)] = {(path[0], path[1]): 1.0} if len(path) > 1 else {}
                except nx.NetworkXNoPath:
                    table[(u, d)] = {}
        return RoutingDecision(table)


class GraphEmbeddingPPOPolicy(RoutingPolicy):
    """Grace-style fixed graph embedding baseline."""
    name = "graph-embedding-ppo"

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.embeddings: Dict[int, np.ndarray] = {}

    def decide(self, graph, observation, failed=()):
        g = graph.to_networkx(failed)
        centrality = nx.betweenness_centrality(g) if len(g) else {}
        table = {}
        for d in range(graph.num_nodes):
            for u in range(graph.num_nodes):
                if u == d:
                    continue
                eligible = graph.eligible_out_edges(u, d, failed)
                if not eligible:
                    table[(u, d)] = {}
                    continue
                edge = max(eligible, key=lambda e: centrality.get(e[1], 0.0) - observation.get("utilization", {}).get(e, 0.0))
                table[(u, d)] = {edge: 1.0}
        return RoutingDecision(table)


class GTSRStylePolicy(GraphEmbeddingPPOPolicy):
    name = "gtsr-style-gnn"


def max_utilization_lp(graph: GraphSpec, demand: np.ndarray, failed=()) -> float:
    """Small-network multi-commodity-flow lower bound using scipy linprog."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return float("nan")
    edges = [e for e in graph.edges if (e[0], e[1]) not in set(failed)]
    pairs = [(s, d) for s in range(graph.num_nodes) for d in range(graph.num_nodes) if s != d and demand[s, d] > 0]
    if not pairs or not edges:
        return 0.0
    nflow = len(edges) * len(pairs)
    zidx = nflow
    c = np.zeros(nflow + 1); c[zidx] = 1.0
    A = []; b = []
    for i, (_, _, cap, _) in enumerate(edges):
        row = np.zeros(nflow + 1)
        for k in range(len(pairs)): row[k * len(edges) + i] = 1.0
        row[zidx] = -cap
        A.append(row); b.append(0.0)
    Aeq = []; beq = []
    for k, (s, d) in enumerate(pairs):
        for node in range(graph.num_nodes):
            row = np.zeros(nflow + 1)
            for i, (u, v, _, _) in enumerate(edges):
                if u == node: row[k * len(edges) + i] += 1.0
                if v == node: row[k * len(edges) + i] -= 1.0
            Aeq.append(row); beq.append(float(demand[s, d]) if node == s else -float(demand[s, d]) if node == d else 0.0)
    result = linprog(c, A_ub=np.asarray(A), b_ub=np.asarray(b), A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=(0, None), method="highs")
    return float(result.fun) if result.success else float("nan")
