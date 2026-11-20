from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Any
import time
import numpy as np
import networkx as nx

from .graph import GraphSpec


RoutingTable = Dict[Tuple[int, int], Dict[Tuple[int, int], float]]


@dataclass
class RoutingDecision:
    table: RoutingTable
    inference_ms: float = 0.0
    valid: bool = True
    fallback: str | None = None


def ecmp_table(graph: GraphSpec, failed: Iterable[Tuple[int, int]] = ()) -> RoutingTable:
    start = time.perf_counter()
    failed_set = set(failed)
    g = graph.to_networkx(failed_set)
    table: RoutingTable = {}
    for destination in range(graph.num_nodes):
        for node in range(graph.num_nodes):
            if node == destination:
                continue
            eligible = graph.eligible_out_edges(node, destination, failed_set)
            if not eligible:
                table[(node, destination)] = {}
                continue
            weights = {e: 1.0 / len(eligible) for e in eligible}
            table[(node, destination)] = weights
    return table


def shortest_path_table(graph: GraphSpec, failed=(), utilization=None) -> RoutingTable:
    failed_set = set(failed)
    g = graph.to_networkx(failed_set)
    table: RoutingTable = {}
    for d in range(graph.num_nodes):
        for u in range(graph.num_nodes):
            if u == d:
                continue
            try:
                # The shared action mask is hop-decreasing. Using the same
                # shortest-hop metric keeps the baseline valid under that mask.
                path = nx.shortest_path(g, u, d, weight=None)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                table[(u, d)] = {}
                continue
            table[(u, d)] = {(path[0], path[1]): 1.0} if len(path) > 1 else {}
    return table


def load_aware_table(graph: GraphSpec, failed=(), utilization=None) -> RoutingTable:
    util = utilization or {}
    g = graph.to_networkx(failed)
    for u, v in g.edges:
        g[u][v]["cost"] = g[u][v]["delay"] * (1.0 + 3.0 * float(util.get((u, v), 0.0)))
    table: RoutingTable = {}
    for d in range(graph.num_nodes):
        for u in range(graph.num_nodes):
            if u == d:
                continue
            allowed = graph.eligible_out_edges(u, d, failed)
            if not allowed:
                table[(u, d)] = {}
                continue
            scored = []
            for e in allowed:
                v = e[1]
                try:
                    tail_cost = nx.shortest_path_length(g, v, d, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                scored.append((g[e[0]][e[1]]["cost"] + tail_cost, e))
            table[(u, d)] = {min(scored)[1]: 1.0} if scored else {}
    return table


def valid_table(graph: GraphSpec, table: RoutingTable, failed=()) -> bool:
    failed_set = set(failed)
    for (u, d), weights in table.items():
        if u == d:
            continue
        if any(e in failed_set or e not in graph.edge_pairs for e in weights):
            return False
        if weights and abs(sum(weights.values()) - 1.0) > 1e-5:
            return False
        if any(v <= 0 for v in weights.values()):
            return False
        allowed = set(graph.eligible_out_edges(u, d, failed_set))
        if any(e not in allowed for e in weights):
            return False
    return True


class RoutingPolicy(ABC):
    name = "policy"

    @abstractmethod
    def decide(self, graph: GraphSpec, observation: Dict[str, Any], failed=()) -> RoutingDecision:
        raise NotImplementedError


class OSPFPolicy(RoutingPolicy):
    name = "ospf"
    def decide(self, graph, observation, failed=()):
        started = time.perf_counter()
        table = shortest_path_table(graph, failed, observation.get("utilization"))
        return RoutingDecision(table, (time.perf_counter() - started) * 1000)


class ECMPPolicy(RoutingPolicy):
    name = "ecmp"
    def decide(self, graph, observation, failed=()):
        started = time.perf_counter()
        table = ecmp_table(graph, failed)
        return RoutingDecision(table, (time.perf_counter() - started) * 1000)


class LoadAwarePolicy(RoutingPolicy):
    name = "load-aware-dijkstra"
    def decide(self, graph, observation, failed=()):
        started = time.perf_counter()
        table = load_aware_table(graph, failed, observation.get("utilization"))
        return RoutingDecision(table, (time.perf_counter() - started) * 1000)
