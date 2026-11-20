"""Component ablation of TARO (paper Table 8, Fig. 6).

Variant re-implements the packaged StableHeuristicTARO with switchable parts; with every switch at its
default it reproduces the packaged policy exactly (checked at the end of this script).
Output: results/abl.json
"""
import json, time
import numpy as np
from _common import RESULTS
from taro.datasets import canonical_topologies
from taro.graph import GraphSpec
from taro.scenarios import make_scenario
from taro.runner import run_episode
from taro.config import TAROConfig
from taro.policies import StableHeuristicTARO
from taro import targeted_experiment as te
from taro.routing import RoutingDecision


class Variant(StableHeuristicTARO):
    def __init__(self, seed=7, cap=0.1, threshold=0.1, use_cap_delay=True, use_util=True, use_queue=True, single_path=False):
        super().__init__(max_weight_change=cap, update_threshold=threshold, seed=seed)
        self.f = (use_cap_delay, use_util, use_queue)
        self.single = single_path

    def decide(self, graph, observation, failed=()):
        previous = self.previous
        table = {}; fs = set(failed); util = observation.get("utilization", {}); queue = observation.get("queue", {})
        cd, uu, qq = self.f
        t0 = time.perf_counter()
        for d in range(graph.num_nodes):
            for u in range(graph.num_nodes):
                if u == d:
                    continue
                el = graph.eligible_out_edges(u, d, fs)
                if not el:
                    table[(u, d)] = {}; continue
                sc = []
                for e in el:
                    _, _, cap, dl = next(x for x in graph.edges if (x[0], x[1]) == e)
                    s = (cap / max(1 + dl, 1e-6) if cd else 1.0)
                    if uu: s *= (1 - min(0.95, util.get(e, 0.0)))
                    if qq: s /= (1 + queue.get(e, 0.0) / max(cap, 1e-9))
                    sc.append(max(s, 1e-6))
                p = np.asarray(sc, float); p /= p.sum()
                old = previous.get((u, d), {})
                if old:
                    for i, e in enumerate(el):
                        p[i] = np.clip(p[i], old.get(e, 0.0) - self.max_weight_change, old.get(e, 0.0) + self.max_weight_change)
                    p = np.maximum(p, 0); p /= max(p.sum(), 1e-9)
                w = {el[int(np.argmax(p))]: 1.0} if self.single else {e: float(x) for e, x in zip(el, p)}
                if old and set(old) == set(w):
                    tv = 0.5 * sum(abs(old.get(e, 0) - w.get(e, 0)) for e in set(old) | set(w))
                    if tv < self.update_threshold:
                        w = old.copy()
                table[(u, d)] = w
        self.previous = table
        return RoutingDecision(table, (time.perf_counter() - t0) * 1000)


V = {"full (Stable TARO)": {}, "no stability threshold": {"threshold": 0.0},
     "threshold 0.05": {"threshold": 0.05}, "threshold 0.20": {"threshold": 0.20}, "threshold 0.30": {"threshold": 0.30},
     "cap 0.05": {"cap": 0.05}, "cap 0.20": {"cap": 0.20}, "no change cap": {"cap": 1.0},
     "no capacity/delay term": {"use_cap_delay": False}, "no utilisation term": {"use_util": False},
     "no queue term": {"use_queue": False}, "single path (argmax)": {"single_path": True}}

# check: with all components on, Variant equals the packaged policy
for s in (101, 102):
    sc = te.make_scenario(s)
    a = run_episode(Variant(seed=s), sc, TAROConfig(seed=s))[1]
    b = run_episode(StableHeuristicTARO(seed=s), sc, TAROConfig(seed=s))[1]
    assert a["delivered"] == b["delivered"] and a["flow_updates"] == b["flow_updates"], "Variant does not reproduce TARO"

man = canonical_topologies(str(RESULTS / "topologies"))
g = GraphSpec.from_json(man["topologies"]["NSFNET"])
cases = [("NSFNET", sn, (lambda s, b=b, f=f, l=l: make_scenario(g, horizon=20, seed=s, burst=b, failures=f, load=l)), range(7, 12))
         for sn, b, f, l in [("normal", False, 0, 1.0), ("heavy", False, 0, 2.0), ("burst", True, 0, 1.0), ("failure-10", False, .1, 1.0)]]
cases.append(("targeted-8", "burst (selected graph)", (lambda s: te.make_scenario(s)), range(101, 111)))
rows = []
for topo, sn, mk, seeds in cases:
    for seed in seeds:
        sc = mk(seed)
        for vn, kw in V.items():
            rec, tot = run_episode(Variant(seed=seed, **kw), sc, TAROConfig(seed=seed))
            rows.append({"topology": topo, "scenario": sn, "seed": seed, "variant": vn,
                         "delivery": tot["delivered"] / tot["offered"], "delay": tot["delay_sum"] / max(tot["delivered"], 1e-9),
                         "updates": tot["flow_updates"], "flaps": tot["route_flaps"], "rerouting": tot["rerouting_ratio"] / len(rec),
                         "inference_ms": float(np.mean([r["inference_ms"] for r in rec]))})
    print(topo, sn, "done", flush=True)
json.dump(rows, open(RESULTS / "abl.json", "w"))
