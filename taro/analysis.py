from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import numpy as np
import pandas as pd


def load_summary(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def make_figures(summary_path: str = "artifacts/results/summary.json", output_dir: str = "artifacts/figures") -> list[str]:
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_summary(summary_path)
    paths = []
    for metric, ylabel, filename in [
        ("mean_delay", "Mean delay", "delay_by_scenario.png"),
        ("delivery_ratio", "Delivery ratio", "delivery_by_scenario.png"),
        ("max_utilization", "Maximum utilization", "utilization_by_scenario.png"),
        ("flow_updates", "Forwarding updates", "updates_by_scenario.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot = df.pivot_table(index="scenario", columns="method", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scenario")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def make_comparison_table(summary_path: str = "artifacts/results/summary.json", output: str = "artifacts/results/comparison.csv") -> str:
    df = load_summary(summary_path)
    cols = [c for c in ["scenario", "method", "seed", "mean_delay", "delivery_ratio", "max_utilization", "flow_updates", "loss_rate"] if c in df]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values(["scenario", "method", "seed"]).to_csv(output, index=False)
    return output
