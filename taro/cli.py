from __future__ import annotations

import argparse
from pathlib import Path
import json

from .config import TAROConfig
from .graph import GraphSpec
from .policies import HeuristicTARO
from .runner import evaluate_methods, run_episode
from .scenarios import make_scenario


def smoke_test() -> None:
    graph = GraphSpec.ring(6)
    scenario = make_scenario(graph, horizon=8, seed=7, burst=True, failures=0.1)
    policy = HeuristicTARO()
    records, total = run_episode(policy, scenario)
    assert total["offered"] >= total["delivered"]
    assert total["dropped"] >= 0
    print(json.dumps({"status": "ok", "steps": len(records), "total": total}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="TARO Phase 3 command line tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke-test")
    train = sub.add_parser("train")
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--seed", type=int, default=7)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output", default="artifacts/results")
    figures = sub.add_parser("figures")
    figures.add_argument("--summary", default="artifacts/results/summary.json")
    figures.add_argument("--output", default="artifacts/figures")
    ppo = sub.add_parser("train-ppo")
    ppo.add_argument("--episodes", type=int, default=25)
    ppo.add_argument("--seed", type=int, default=7)
    ppo.add_argument("--output", default="artifacts/checkpoints/taro_ppo.npz")
    mn = sub.add_parser("mininet-smoke")
    mn.add_argument("--nodes", type=int, default=4)
    full = sub.add_parser("evaluate-full")
    full.add_argument("--output", default="artifacts/full_results")
    full.add_argument("--topologies", type=int, default=None)
    full.add_argument("--seeds", type=int, default=5)
    full.add_argument("--horizon", type=int, default=20)
    ab = sub.add_parser("ablations")
    ab.add_argument("--output", default="artifacts/ablations")
    ds = sub.add_parser("prepare-datasets")
    ds.add_argument("--output", default="artifacts/datasets")
    args = parser.parse_args()
    if args.command == "smoke-test":
        smoke_test()
    elif args.command == "train":
        Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
        config = TAROConfig(seed=args.seed)
        config.save(f"artifacts/checkpoints/taro_seed{args.seed}_config.json")
        from .ppo_numpy import train_ppo
        graph = GraphSpec.random(10, args.seed)
        npz_path = f"artifacts/checkpoints/taro_seed{args.seed}.npz"
        result = train_ppo(graph, episodes=max(1, args.steps // 40), output=npz_path, config=config)
        result["steps_requested"] = args.steps
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        rows = evaluate_methods(args.output)
        print(json.dumps({"output": args.output, "runs": len(rows)}, indent=2))
    elif args.command == "figures":
        from .analysis import make_figures, make_comparison_table
        summary = args.summary
        if not Path(summary).exists():
            evaluate_methods(str(Path(summary).parent))
        paths = make_figures(summary, args.output)
        table = make_comparison_table(summary)
        print(json.dumps({"figures": paths, "table": table}, indent=2))
    elif args.command == "train-ppo":
        from .ppo_numpy import train_ppo
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result = train_ppo(GraphSpec.random(10, args.seed), args.episodes, args.output, TAROConfig(seed=args.seed))
        print(json.dumps(result, indent=2))
    elif args.command == "mininet-smoke":
        from .mininet_adapter import MininetAdapter
        adapter = MininetAdapter(GraphSpec.line(args.nodes))
        try:
            adapter.start()
            loss = adapter.ping_all()
            sample = adapter.poll()
            print(json.dumps({"ping_loss_percent": loss, "interfaces_measured": len(sample.port_bytes)}, indent=2))
        finally:
            adapter.stop()
    elif args.command == "evaluate-full":
        from .runner import evaluate_full_matrix
        rows = evaluate_full_matrix(args.output, seeds=tuple(range(7, 7 + args.seeds)), horizon=args.horizon, topology_limit=args.topologies)
        print(json.dumps({"output": args.output, "runs": len(rows)}, indent=2))
    elif args.command == "ablations":
        from .ablations import run_ablation_suite
        rows = run_ablation_suite(args.output)
        print(json.dumps({"output": args.output, "runs": len(rows)}, indent=2))
    elif args.command == "prepare-datasets":
        from .datasets import canonical_topologies
        print(json.dumps(canonical_topologies(args.output), indent=2))


if __name__ == "__main__":
    main()
