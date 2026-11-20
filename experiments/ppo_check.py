"""Training check of the GNN-PPO prototype (paper Sect. 3.7).

Trains for 40 episodes on a 10-node graph at heavy load with the update as implemented and with a
sign-corrected actor step, then evaluates on held-out workloads together with ECMP, OSPF-style and TARO.
Outputs: results/ppo_check.json, results/ppo_ref.json
"""
import json, sys
import numpy as np
from _common import RESULTS
from taro.ppo_numpy import GraphPPOPolicy
from taro.graph import GraphSpec
from taro.env import RoutingEnv
from taro.scenarios import make_scenario
from taro.config import TAROConfig
from taro.policies import StableHeuristicTARO
from taro.routing import ECMPPolicy, OSPFPolicy

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 40


class SignCorrected(GraphPPOPolicy):
    """Same update, but the actor step is applied with the opposite (gradient-ascent) sign."""
    def update(self, transitions, gamma=None):
        before = self.W_actor.copy(); bb = self.b_actor
        out = super().update(transitions, gamma)
        delta = self.W_actor - before
        self.W_actor = before - delta
        self.b_actor = bb - (self.b_actor - bb)
        return out


def run(cls, episodes, lr):
    g = GraphSpec.random(10, 7)
    cfg = TAROConfig(seed=7, learning_rate=lr)
    pol = cls(g, cfg); hist = []
    for ep in range(episodes):
        env = RoutingEnv(make_scenario(g, horizon=20, seed=1000 + ep % 4, load=2.0), cfg)   # four repeating training workloads
        obs = env.reset(); trs = []; R = 0
        while not env.done:
            dec, tr = pol.act(env, obs); obs, r, _, _ = env.step(dec); tr.reward = r; trs.append(tr); R += r
        pol.update(trs); hist.append(R)
    ev = []
    for s in range(2000, 2003):                                                                 # held-out workloads
        env = RoutingEnv(make_scenario(g, horizon=20, seed=s, load=2.0), cfg); obs = env.reset(); R = 0
        while not env.done:
            dec, _ = pol.act(env, obs, deterministic=True); obs, r, _, _ = env.step(dec); R += r
        ev.append(R)
    return hist, float(np.mean(ev))


res = {}
for name, cls in [("original", GraphPPOPolicy), ("sign-corrected", SignCorrected)]:
    for lr in (3e-4, 3e-2):
        h, e = run(cls, EPISODES, lr)
        res[f"{name} lr={lr}"] = {"hist": h, "eval": e}
        print(name, lr, "first5", round(float(np.mean(h[:5])), 3), "last5", round(float(np.mean(h[-5:])), 3), "eval", round(e, 3), flush=True)
json.dump(res, open(RESULTS / "ppo_check.json", "w"))

# reference rewards of fixed policies on the same held-out workloads
g = GraphSpec.random(10, 7); cfg = TAROConfig(seed=7); ref = {}
for name, mk in [("ecmp", lambda: ECMPPolicy()), ("ospf-style", lambda: OSPFPolicy()), ("taro-stable", lambda: StableHeuristicTARO(seed=7))]:
    ev = []
    for s in range(2000, 2003):
        env = RoutingEnv(make_scenario(g, horizon=20, seed=s, load=2.0), cfg); obs = env.reset(); R = 0; p = mk()
        while not env.done:
            obs, r, _, _ = env.step(p.decide(g, obs, obs["failed"])); R += r
        ev.append(R)
    ref[name] = float(np.mean(ev))
json.dump(ref, open(RESULTS / "ppo_ref.json", "w"))
print(ref)
