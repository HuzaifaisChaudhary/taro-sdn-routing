"""R30 robustness check: the targeted burst workload on 30 random 8-node graphs (paper Table 4).

Output: results/robust.json
"""
import json
from _common import RESULTS
from taro import targeted_experiment as te
from taro.policies import HeuristicTARO, StableHeuristicTARO
from taro.routing import ECMPPolicy, OSPFPolicy, LoadAwarePolicy
from taro.runner import run_episode
from taro.config import TAROConfig

METHODS = {"taro-stable": StableHeuristicTARO, "taro-heuristic": HeuristicTARO, "ecmp": ECMPPolicy,
           "ospf-style": OSPFPolicy, "load-aware": LoadAwarePolicy}
out = []
for gs in range(0, 30):                 # graph seeds 0-29
    for seed in range(101, 106):        # workload seeds 101-105
        sc = te.make_scenario(seed, graph_seed=gs)
        for name, f in METHODS.items():
            p = f(seed=seed) if name.startswith("taro-") else f()
            rec, tot = run_episode(p, sc, TAROConfig(seed=seed))
            out.append({"graph_seed": gs, "seed": seed, "method": name,
                        "delivery": tot["delivered"] / tot["offered"],
                        "delay": tot["delay_sum"] / max(tot["delivered"], 1e-9),
                        "updates": tot["flow_updates"]})
    print("graph", gs, "done", flush=True)
json.dump(out, open(RESULTS / "robust.json", "w"))
