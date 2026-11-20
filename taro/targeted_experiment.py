"""Paired burst experiment on an exploratory heterogeneous-capacity graph.

The seed-4 graph was selected in an exploratory search. Workload seeds below
are held out from that search; this is not a held-out topology evaluation.
"""
from pathlib import Path
import json
import numpy as np

from .config import TAROConfig
from .graph import GraphSpec
from .policies import HeuristicTARO, StableHeuristicTARO
from .routing import ECMPPolicy, OSPFPolicy, LoadAwarePolicy
from .runner import run_episode
from .scenarios import Scenario


def make_scenario(seed, horizon=24, drain=10, graph_seed=4):
    base = GraphSpec.random(8, graph_seed, p=.35)
    rng = np.random.default_rng(graph_seed)
    edges = [(u, v, float(rng.choice([2, 4, 8, 16])), delay)
             for u, v, _, delay in base.edges]
    graph = GraphSpec(base.num_nodes, edges, name=f"heterogeneous-{graph_seed}")
    demand = np.zeros((horizon + drain, 8, 8))
    workload = np.random.default_rng(seed)
    for t in range(horizon):
        variation = workload.uniform(.85, 1.15)
        if t % 6 < 3:
            demand[t, 0, 5], demand[t, 5, 0], demand[t, 1, 6] = (
                80 * variation, 80 * variation, 50 * variation)
        else:
            demand[t, 2, 7], demand[t, 7, 2], demand[t, 3, 4] = (
                80 * variation, 80 * variation, 50 * variation)
    return Scenario(graph, demand, [[] for _ in demand],
                    name=f"paired-burst-seed-{seed}")


def execute(output="artifacts/targeted_burst", seeds=range(101, 111)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = {"taro-stable": StableHeuristicTARO, "taro-heuristic": HeuristicTARO, "ecmp": ECMPPolicy,
               "ospf-style": OSPFPolicy, "load-aware": LoadAwarePolicy}
    runs = []
    for seed in seeds:
        scenario = make_scenario(seed)
        for name, factory in methods.items():
            policy = factory(seed=seed) if name.startswith("taro-") else factory()
            records, total = run_episode(policy, scenario, TAROConfig(seed=seed))
            remaining = records[-1]["queued"]
            accounted = total["delivered"] + total["dropped"] + remaining
            if not np.isclose(total["offered"], accounted, atol=1e-7):
                raise AssertionError(f"Traffic conservation failed: {seed}, {name}")
            runs.append({"seed": seed, "method": name, "offered": total["offered"],
                         "delivered": total["delivered"], "dropped": total["dropped"],
                         "remaining": remaining, "delivery_fraction": total["delivered"] / total["offered"],
                         "mean_delay": total["delay_sum"] / max(total["delivered"], 1e-9),
                         "changed_entries": total["flow_updates"],
                         "mean_route_change": total["rerouting_ratio"] / len(records)})
    payload = {"note": "Exploratory graph selected by search; independent workload seeds only. "
                       "TARO methods are heuristics, not trained PPO. Threshold .1 was selected on "
                       "workload seeds 31-33. Delay and loss are fluid estimates.",
               "graph_seed": 4, "workload_seeds": list(seeds), "runs": runs}
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_figures(payload, output)
    return payload


def make_figures(payload, output):
    """Plot all paired runs and disclose the selection and model type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t

    runs = payload["runs"]
    names = ["taro-stable", "taro-heuristic", "ecmp", "ospf-style", "load-aware"]
    colors = ["#168477", "#79b7a4", "#426f9e", "#777777", "#d19236"]
    measures = [
        ("delivery_fraction", "Delivered / offered", "Delivery fraction", "delivery.png"),
        ("mean_delay", "Model delay units", "Mean delivered-traffic delay", "delay.png"),
        ("changed_entries", "Changed entries per run", "Routing-update proxy", "updates.png"),
    ]
    summary = {}
    for metric, ylabel, title, filename in measures:
        values = [[row[metric] for row in runs if row["method"] == name] for name in names]
        means = [float(np.mean(v)) for v in values]
        errors = [float(t.ppf(.975, len(v)-1) * np.std(v, ddof=1) / np.sqrt(len(v)))
                  if len(v) > 1 else 0.0 for v in values]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(names, means, color=colors, yerr=errors, capsize=4)
        for index, series in enumerate(values):
            ax.scatter(np.full(len(series), index), series, s=13, color="#262626", alpha=.55, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title + "\n10 workload seeds, one selected synthetic topology")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
        summary[metric] = dict(zip(names, means))
    (output / "means.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
