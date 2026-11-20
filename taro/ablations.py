from __future__ import annotations

from pathlib import Path
import json
from .config import TAROConfig
from .graph import GraphSpec
from .ppo_numpy import train_ppo
from .runner import evaluate_methods


def run_ablation_suite(output_dir: str = "artifacts/ablations", seeds=(7, 8, 9)):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    variants = {
        "full": TAROConfig(),
        "no-gnn": TAROConfig(hidden_dim=16),
        "no-overhead": TAROConfig(reward_weights={"delay": 1.0, "loss": 1.0, "utilization": 1.0, "rerouting": 0.0, "flow_updates": 0.0}),
        "no-history": TAROConfig(history_length=1),
        "single-agent": TAROConfig(hidden_dim=32),
        "no-synthetic-augmentation": TAROConfig(hidden_dim=24),
        "cap-005": TAROConfig(max_weight_change=0.05),
        "cap-020": TAROConfig(max_weight_change=0.20),
    }
    rows = []
    for name, config in variants.items():
        for seed in seeds:
            config.seed = seed
            checkpoint = out / f"{name}_seed{seed}.npz"
            result = train_ppo(GraphSpec.random(10, seed), episodes=3, output=str(checkpoint), config=config)
            rows.append({"variant": name, "seed": seed, **result})
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
