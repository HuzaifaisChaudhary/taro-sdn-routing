"""Qualitative example (paper Fig. 3) and update-timing numbers quoted in Sect. 5.2.

Outputs: results/qual.json, results/freeze.json
"""
import json
import numpy as np
from _common import RESULTS
from taro import targeted_experiment as te
from taro.env import RoutingEnv
from taro.config import TAROConfig
from taro.policies import HeuristicTARO, StableHeuristicTARO
from taro.datasets import canonical_topologies
from taro.graph import GraphSpec
from taro.scenarios import make_scenario

SEED = 101
sc = te.make_scenario(SEED); g = sc.topology


def trace(policy, key=None, edge=None):
    env = RoutingEnv(sc, TAROConfig(seed=SEED)); obs = env.reset(); upd, w, tables = [], [], []
    while not env.done:
        dec = policy.decide(g, obs, obs["failed"]); obs, r, _, info = env.step(dec)
        upd.append(info["flow_updates"]); tables.append({k: dict(v) for k, v in env.last_table.items()})
        if key is not None:
            w.append(env.last_table[key].get(edge, 0.0))
    return upd, w, tables, env


# entry whose split weight varies most without the threshold
_, _, hist, _ = trace(HeuristicTARO(seed=SEED))
best = None
for key in hist[0]:
    if len(hist[0][key]) < 2:
        continue
    e0 = sorted(hist[0][key])[0]
    series = [h[key].get(e0, 0.0) for h in hist]
    if best is None or max(series) - min(series) > best[0]:
        best = (max(series) - min(series), key, e0)
_, key, e0 = best

out = {"seed": SEED, "methods": {}, "trace": {"node": key[0], "dest": key[1], "edge": list(e0), "weights": {}}}
for name, pol in [("taro-stable", StableHeuristicTARO(seed=SEED)), ("taro-heuristic", HeuristicTARO(seed=SEED)),
                  ("taro-cap02", StableHeuristicTARO(max_weight_change=0.2, seed=SEED))]:
    upd, w, _, env = trace(pol, key, e0)
    out["methods"][name] = {"updates": upd, "total_updates": int(sum(upd)), "delivery": env.total.delivered / env.total.offered}
    out["trace"]["weights"][name] = w
out["multi_entries"] = sum(1 for k in hist[0] if len(hist[0][k]) > 1); out["entries"] = len(hist[0])
json.dump(out, open(RESULTS / "qual.json", "w"))
print("entry", key, "edge", e0, {k: v["total_updates"] for k, v in out["methods"].items()})


def updates(scn, seed):
    env = RoutingEnv(scn, TAROConfig(seed=seed)); obs = env.reset(); pol = StableHeuristicTARO(seed=seed); u = []
    while not env.done:
        dec = pol.decide(scn.topology, obs, obs["failed"]); obs, r, _, info = env.step(dec); u.append(info["flow_updates"])
    return u


fz = {}
first, total = [], []
for s in range(101, 111):
    u = updates(te.make_scenario(s), s); first.append(u[0]); total.append(sum(u))
fz["targeted_first"] = float(np.mean(first)); fz["targeted_total"] = float(np.mean(total))
fz["multi8"] = sum(1 for u in range(8) for d in range(8) if u != d and len(g.eligible_out_edges(u, d)) > 1)
man = canonical_topologies(str(RESULTS / "topologies")); g14 = GraphSpec.from_json(man["topologies"]["NSFNET"])
fz["multi14"] = sum(1 for u in range(14) for d in range(14) if u != d and len(g14.eligible_out_edges(u, d)) > 1)
for sn, b, f, l in [("normal", False, 0, 1.0), ("heavy", False, 0, 2.0), ("failure-10", False, .1, 1.0)]:
    F, T, A = [], [], []
    for s in range(7, 12):
        u = updates(make_scenario(g14, horizon=20, seed=s, burst=b, failures=f, load=l), s)
        F.append(u[0]); T.append(sum(u)); A.append(sum(u[1:]))
    fz[sn] = {"first": float(np.mean(F)), "total": float(np.mean(T)), "after": float(np.mean(A))}
json.dump(fz, open(RESULTS / "freeze.json", "w"))
print(fz)
