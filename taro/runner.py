from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Iterable
import json
import numpy as np

from .config import TAROConfig
from .env import RoutingEnv
from .graph import GraphSpec
from .metrics import summarize
from .policies import HeuristicTARO
from .routing import ECMPPolicy, OSPFPolicy, LoadAwarePolicy
from .baselines import DQNPolicy, DDPGPolicy, GraphEmbeddingPPOPolicy, GTSRStylePolicy
from .scenarios import make_scenario


POLICIES = {"taro": HeuristicTARO, "ecmp": ECMPPolicy, "ospf": OSPFPolicy, "load-aware": LoadAwarePolicy,
            "dqn": DQNPolicy, "ddpg": DDPGPolicy, "graph-embedding": GraphEmbeddingPPOPolicy,
            "gtsr-style": GTSRStylePolicy}


def run_episode(policy, scenario, config=None):
    env = RoutingEnv(scenario, config)
    obs = env.reset()
    records = []
    while not env.done:
        decision = policy.decide(env.graph, obs, obs["failed"])
        obs, reward, done, info = env.step(decision)
        info["reward"] = reward
        info["t"] = env.t - 1
        records.append(info)
    return records, env.total.as_dict()


def evaluate_methods(output_dir: str = "artifacts/results", seeds=(7, 8, 9), horizon=60):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph = GraphSpec.ring(6)
    all_rows = []
    for scenario_kind, burst, failures in [("normal", False, 0.0), ("burst", True, 0.0), ("failure", False, 0.1)]:
        for seed in seeds:
            scenario = make_scenario(graph, horizon=horizon, seed=seed, burst=burst, failures=failures)
            for name, cls in POLICIES.items():
                policy = cls(seed=seed) if name in {"taro", "dqn", "graph-embedding", "gtsr-style"} else cls()
                records, total = run_episode(policy, scenario, TAROConfig(seed=seed))
                summary = summarize(records)
                summary.update({"scenario": scenario_kind, "seed": seed, "method": name})
                all_rows.append(summary)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    return all_rows


def evaluate_full_matrix(output_dir: str = "artifacts/full_results", seeds=(7, 8, 9, 10, 11), horizon=40, topology_limit=None):
    """Run the Phase 2 scenario matrix across canonical topology families."""
    from .datasets import canonical_topologies
    from .graph import GraphSpec
    from .statistics import write_statistical_report
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = canonical_topologies(str(out / "datasets"))
    rows = []
    topology_items = list(manifest["topologies"].items())[:topology_limit] if topology_limit else list(manifest["topologies"].items())
    for topology_name, path in topology_items:
        graph = GraphSpec.from_json(path)
        topology_horizon = min(horizon, 8 if graph.num_nodes > 30 else 20)
        for scenario_name, burst, failure, load in [
            ("normal", False, 0.0, 1.0), ("heavy", False, 0.0, 2.0),
            ("burst", True, 0.0, 1.0), ("failure-05", False, 0.05, 1.0),
            ("failure-10", False, 0.10, 1.0), ("failure-15", False, 0.15, 1.0)]:
            for seed in seeds:
                scenario = make_scenario(graph, horizon=topology_horizon, seed=seed, burst=burst, failures=failure, load=load)
                for name, cls in POLICIES.items():
                    policy = cls(seed=seed) if name in {"taro", "dqn", "graph-embedding", "gtsr-style"} else cls()
                    records, _ = run_episode(policy, scenario, TAROConfig(seed=seed))
                    summary = summarize(records)
                    summary.update({"topology": topology_name, "scenario": scenario_name, "seed": seed, "method": name})
                    rows.append(summary)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_statistical_report(str(out / "summary.json"), str(out / "statistics"))
    return rows
