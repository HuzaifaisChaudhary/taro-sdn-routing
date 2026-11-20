"""Scenario matrix on the synthetic N14, N22 and N25 graphs (paper Tables 5 and 7, Figs. 4 and 5).

Usage: python matrix.py NSFNET   -> results/m_nsf.json     (N14)
       python matrix.py GEANT    -> results/m_geant.json   (N22)
       python matrix.py "AT&T"   -> results/m_att.json     (N25)
       python matrix.py NSFNET --light -> results/m_light.json (quarter load, quoted in Sect. 5.4)
The graph files named NSFNET/GEANT/AT&T are random graphs of matching size, not the real topologies.
"""
import json, sys, time
from _common import RESULTS
from taro.datasets import canonical_topologies
from taro.graph import GraphSpec
from taro.scenarios import make_scenario
from taro.runner import run_episode
from taro.metrics import summarize
from taro.config import TAROConfig
from taro.policies import HeuristicTARO, StableHeuristicTARO
from taro.routing import ECMPPolicy, OSPFPolicy, LoadAwarePolicy
from taro.baselines import DQNPolicy, DDPGPolicy, GraphEmbeddingPPOPolicy

topo = sys.argv[1]
light = "--light" in sys.argv
OUTNAME = {"NSFNET": "m_nsf.json", "GEANT": "m_geant.json", "AT&T": "m_att.json"}[topo]
if light:
    OUTNAME = "m_light.json"
M = {"taro-stable": lambda s: StableHeuristicTARO(seed=s), "taro-heuristic": lambda s: HeuristicTARO(seed=s),
     "ecmp": lambda s: ECMPPolicy(), "ospf-style": lambda s: OSPFPolicy(), "load-aware": lambda s: LoadAwarePolicy(),
     "dqn-style": lambda s: DQNPolicy(seed=s), "ddpg-style": lambda s: DDPGPolicy(),
     "graph-embedding": lambda s: GraphEmbeddingPPOPolicy(seed=s)}
SCENARIOS = [("normal", False, 0, 1.0), ("heavy", False, 0, 2.0), ("burst", True, 0, 1.0),
             ("failure-05", False, .05, 1.0), ("failure-10", False, .10, 1.0), ("failure-15", False, .15, 1.0)]
if light:
    SCENARIOS = [("light", False, 0, 0.25), ("light-burst", True, 0, 0.25), ("light-failure-10", False, .10, 0.25)]

man = canonical_topologies(str(RESULTS / "topologies"))
g = GraphSpec.from_json(man["topologies"][topo])
rows = []
for sn, burst, fail, load in SCENARIOS:
    for seed in range(7, 12):
        sc = make_scenario(g, horizon=20, seed=seed, burst=burst, failures=fail, load=load)
        for name, f in M.items():
            t0 = time.time()
            rec, tot = run_episode(f(seed), sc, TAROConfig(seed=seed))
            s = summarize(rec)
            s.update({"topology": topo, "nodes": g.num_nodes, "links": len(g.edges) // 2, "scenario": sn, "seed": seed,
                      "method": name, "total_offered": tot["offered"], "total_delivered": tot["delivered"],
                      "total_dropped": tot["dropped"], "delivery_total": tot["delivered"] / tot["offered"],
                      "delay_total": tot["delay_sum"] / max(tot["delivered"], 1e-9), "updates_total": tot["flow_updates"],
                      "flaps_total": tot["route_flaps"], "maxutil_peak": tot["max_utilization"], "wall_s": time.time() - t0})
            rows.append(s)
    print(topo, sn, "done", flush=True)
json.dump(rows, open(RESULTS / OUTNAME, "w"))
