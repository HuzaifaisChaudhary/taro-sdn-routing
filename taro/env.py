from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple, List
import time
import numpy as np

from .config import TAROConfig
from .graph import GraphSpec
from .scenarios import Scenario
from .routing import RoutingTable, RoutingDecision, ecmp_table, valid_table


@dataclass
class StepMetrics:
    offered: float = 0.0
    delivered: float = 0.0
    dropped: float = 0.0
    delay_sum: float = 0.0
    delay_samples: int = 0
    max_utilization: float = 0.0
    rerouting_ratio: float = 0.0
    flow_updates: int = 0
    route_flaps: int = 0
    inference_ms: float = 0.0
    fallback: str | None = None

    def as_dict(self) -> Dict[str, float | str | None]:
        return self.__dict__.copy()


class RoutingEnv:
    """Deterministic, time-stepped flow-level simulator.

    Traffic is conserved: offered = delivered + dropped + queued at the end of an
    episode. Routing is destination-based and loop-free by construction.
    """

    def __init__(self, scenario: Scenario, config: TAROConfig | None = None):
        self.scenario = scenario
        self.graph = scenario.topology
        self.config = config or TAROConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.capacity = self.graph.capacities()
        self.delay = self.graph.delays()
        self._history: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.queues = {e: 0.0 for e in self.graph.edge_pairs}
        self._queue_by_destination = {e: np.zeros(self.graph.num_nodes) for e in self.graph.edge_pairs}
        self._queue_delay = {e: np.zeros(self.graph.num_nodes) for e in self.graph.edge_pairs}
        self._pending = np.zeros((self.graph.num_nodes, self.graph.num_nodes))
        self._pending_delay = np.zeros_like(self._pending)
        self._last_utilization = {e: 0.0 for e in self.graph.edge_pairs}
        self._step_served = {e: 0.0 for e in self.graph.edge_pairs}
        self.last_table: RoutingTable = ecmp_table(self.graph)
        self.total = StepMetrics()
        self._history = []
        return self.observe()

    @property
    def done(self) -> bool:
        return self.t >= self.scenario.demands.shape[0]

    def observe(self) -> Dict[str, Any]:
        utilization = dict(self._last_utilization)
        queue = dict(self.queues)
        demand = self.scenario.demands[min(self.t, self.scenario.demands.shape[0] - 1)]
        return {"t": self.t, "demand": demand.copy(), "utilization": utilization,
                "queue": queue, "failed": list(self.scenario.failures[min(self.t, len(self.scenario.failures) - 1)]),
                "history": list(self._history)}

    def features(self, destination: int, observation: Dict[str, Any] | None = None) -> Dict[str, np.ndarray]:
        o = observation or self.observe()
        failed = set(o["failed"])
        n = self.graph.num_nodes
        dist = self.graph.shortest_distance(destination, failed)
        node = np.zeros((n, 6), dtype=np.float32)
        edge = np.zeros((len(self.graph.edges), 6), dtype=np.float32)
        for u in range(n):
            incoming = sum(o["demand"][s, u] for s in range(n) if s != u)
            outgoing = sum(o["demand"][u, d] for d in range(n) if d != u)
            degree_in = sum(1 for a, b in self.graph.edge_pairs if b == u and (a, b) not in failed)
            degree_out = sum(1 for a, b in self.graph.edge_pairs if a == u and (a, b) not in failed)
            node[u] = [incoming, outgoing, max(0, incoming - outgoing), degree_in, degree_out, dist.get(u, n + 1)]
        for i, (u, v, cap, delay) in enumerate(self.graph.edges):
            e = (u, v)
            edge[i] = [cap, o["utilization"][e], delay, 1.0 if e in failed else 0.0,
                       o["queue"][e] / cap, 0.0]
        node[:, :3] /= max(float(np.max(node[:, :3])), 1.0)
        node[:, 3:] /= max(n, 1)
        edge[:, 0] /= max(float(np.max(edge[:, 0])), 1.0)
        edge[:, 1] = np.clip(edge[:, 1], 0, 1)
        edge[:, 2] /= max(float(np.max(edge[:, 2])), 1.0)
        edge[:, 4] = np.clip(edge[:, 4], 0, 1)
        return {"node": node, "edge": edge}

    def _compile_table(self, table: RoutingTable, failed: Iterable[Tuple[int, int]]) -> RoutingTable:
        failed_set = set(failed)
        fallback = ecmp_table(self.graph, failed_set)
        compiled: RoutingTable = {}
        for key, weights in table.items():
            allowed = set(self.graph.eligible_out_edges(key[0], key[1], failed_set))
            clean = {e: max(0.0, float(w)) for e, w in weights.items() if e in allowed and np.isfinite(w)}
            total = sum(clean.values())
            compiled[key] = ({e: w / total for e, w in clean.items()} if total > 1e-9 else fallback.get(key, {}))
        return compiled

    def _routing_change(self, old: RoutingTable, new: RoutingTable) -> Tuple[float, int, int]:
        changes = []
        updates = 0
        flaps = 0
        for key in set(old) | set(new):
            a, b = old.get(key, {}), new.get(key, {})
            edges = set(a) | set(b)
            tv = 0.5 * sum(abs(a.get(e, 0) - b.get(e, 0)) for e in edges)
            changes.append(tv)
            if tv > 1e-6:
                updates += 1
            if tv > 0.5:
                flaps += 1
        return float(np.mean(changes) if changes else 0.0), updates, flaps

    def _forward(self, source: int, destination: int, amount: float, table: RoutingTable,
                 failed: Iterable[Tuple[int, int]], max_hops: int = 64) -> Tuple[float, float, float]:
        failed_set = set(failed)
        current = {source: amount}
        delivered = 0.0
        delay_sum = 0.0
        dropped = 0.0
        for _ in range(max_hops):
            nxt: Dict[int, float] = {}
            progressed = False
            for node, volume in current.items():
                if volume <= 0:
                    continue
                if node == destination:
                    delivered += volume
                    delay_sum += volume * 0.0
                    continue
                weights = table.get((node, destination), {})
                if not weights:
                    dropped += volume
                    continue
                for (u, v), weight in weights.items():
                    if (u, v) in failed_set:
                        dropped += volume * weight
                        continue
                    part = volume * weight
                    edge_key = (u, v)
                    self.queues[edge_key] += part
                    queue_limit = 4.0 * self.capacity[edge_key]
                    if self.queues[edge_key] > queue_limit:
                        dropped += self.queues[edge_key] - queue_limit
                        self.queues[edge_key] = queue_limit
                    # Queue service is shared by all flows at this time step.
                    # Forward only the capacity available at the end of the
                    # step; excess remains queued and is measured as congestion.
                    available = max(0.0, self.capacity[edge_key] - self._step_served[edge_key])
                    service = min(available, self.queues[edge_key])
                    self.queues[edge_key] -= service
                    self._step_served[edge_key] += service
                    nxt[v] = nxt.get(v, 0.0) + service
                    delay_sum += service * self.delay[(u, v)]
                    progressed = True
            current = nxt
            if not progressed:
                break
        dropped += sum(current.values())
        return delivered, dropped, delay_sum

    def step(self, decision: RoutingDecision) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode is complete")
        started = time.perf_counter()
        failed = self.scenario.failures[self.t]
        self._step_served = {e: 0.0 for e in self.graph.edge_pairs}
        table = self._compile_table(decision.table, failed)
        if not valid_table(self.graph, table, failed):
            table = ecmp_table(self.graph, failed)
            decision = RoutingDecision(table, decision.inference_ms, False, "invalid-routing-table")
        change, updates, flaps = self._routing_change(self.last_table, table)
        demand = self.scenario.demands[self.t]
        metrics = StepMetrics(rerouting_ratio=change, flow_updates=updates, route_flaps=flaps,
                              inference_ms=decision.inference_ms, fallback=decision.fallback)
        interval = self.config.control_interval
        incoming = self._pending + demand * interval
        incoming_delay = self._pending_delay
        self._pending = np.zeros_like(incoming)
        self._pending_delay = np.zeros_like(incoming)
        metrics.offered = float((demand.sum() - np.trace(demand)) * interval)
        arrivals = {e: np.zeros(self.graph.num_nodes) for e in self.graph.edge_pairs}
        arrival_delay = {e: np.zeros(self.graph.num_nodes) for e in self.graph.edge_pairs}
        for node in range(self.graph.num_nodes):
            for dest in range(self.graph.num_nodes):
                volume = float(incoming[node, dest])
                if node == dest:
                    metrics.delivered += volume
                    metrics.delay_sum += float(incoming_delay[node, dest])
                    continue
                weights = table.get((node, dest), {})
                if not weights:
                    metrics.dropped += volume
                for edge, weight in weights.items():
                    arrivals[edge][dest] += volume * weight
                    arrival_delay[edge][dest] += incoming_delay[node, dest] * weight
        failed_set = set(failed)
        for edge in self.graph.edge_pairs:
            q, qdelay = self._queue_by_destination[edge], self._queue_delay[edge]
            if edge in failed_set:
                metrics.dropped += float(q.sum() + arrivals[edge].sum())
                q[:] = 0
                qdelay[:] = 0
                self._last_utilization[edge] = 0.0
                continue
            q += arrivals[edge]
            qdelay += arrival_delay[edge]
            total = float(q.sum())
            limit = 4.0 * self.capacity[edge] * interval
            if total > limit:
                metrics.dropped += total - limit
                q *= limit / total
                qdelay *= limit / total
                total = limit
            capacity = self.capacity[edge] * interval
            served = min(total, capacity)
            if total:
                departing = q * (served / total)
                departing_delay = qdelay * (served / total)
                q -= departing
                qdelay -= departing_delay
                self._pending[edge[1]] += departing
                self._pending_delay[edge[1]] += departing_delay + departing * self.delay[edge]
            qdelay += q * interval
            self._step_served[edge] = served
            self._last_utilization[edge] = served / capacity if capacity else 0.0
        self.queues = {edge: float(q.sum()) for edge, q in self._queue_by_destination.items()}
        metrics.delay_samples = int(round(metrics.delivered))
        metrics.max_utilization = max(self._last_utilization.values(), default=0.0)
        self.total.offered += metrics.offered
        self.total.delivered += metrics.delivered
        self.total.dropped += metrics.dropped
        self.total.delay_sum += metrics.delay_sum
        self.total.delay_samples += metrics.delay_samples
        self.total.max_utilization = max(self.total.max_utilization, metrics.max_utilization)
        self.total.flow_updates += updates
        self.total.route_flaps += flaps
        self.total.rerouting_ratio += change
        self.last_table = table
        self._history.append({"utilization": metrics.max_utilization, "loss": metrics.dropped / max(metrics.offered, 1e-9), "delay": metrics.delay_sum / max(metrics.delivered, 1e-9)})
        self._history = self._history[-self.config.history_length:]
        self.t += 1
        delay_norm = metrics.delay_sum / max(metrics.delivered, 1e-9) / 10.0
        loss = metrics.dropped / max(metrics.offered, 1e-9)
        update_norm = updates / max(self.graph.num_nodes ** 2, 1)
        w = self.config.reward_weights
        reward = -(w["delay"] * delay_norm + w["loss"] * loss + w["utilization"] * metrics.max_utilization +
                   w["rerouting"] * change + w["flow_updates"] * update_norm)
        info = metrics.as_dict()
        info["step_ms"] = (time.perf_counter() - started) * 1000
        info["loss_rate"] = loss
        info["mean_delay"] = metrics.delay_sum / max(metrics.delivered, 1e-9)
        info["queued"] = float(sum(self.queues.values()) + self._pending.sum())
        return self.observe(), float(reward), self.done, info
