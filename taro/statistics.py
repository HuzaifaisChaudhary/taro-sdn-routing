from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import numpy as np
import pandas as pd


def confidence_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    from scipy.stats import t
    for keys, group in df.groupby(["scenario", "method"]):
        x = group[metric].dropna().to_numpy(float)
        mean = float(x.mean()) if len(x) else float("nan")
        half = float(t.ppf(0.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        rows.append({"scenario": keys[0], "method": keys[1], "metric": metric, "n": len(x), "mean": mean,
                     "ci_low": mean-half, "ci_high": mean+half})
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, metric: str, reference: str = "ospf") -> pd.DataFrame:
    from scipy.stats import wilcoxon, friedmanchisquare
    rows = []
    for scenario, group in df.groupby("scenario"):
        pivot = group.pivot(index="seed", columns="method", values=metric)
        if reference in pivot:
            for method in pivot.columns:
                if method == reference: continue
                joined = pivot[[reference, method]].dropna()
                if len(joined) >= 2:
                    stat, p = wilcoxon(joined[reference], joined[method])
                    rows.append({"scenario": scenario, "comparison": f"{method} vs {reference}", "test": "wilcoxon", "stat": float(stat), "p": float(p)})
        cols = [c for c in pivot.columns if pivot[c].notna().all()]
        if len(cols) >= 3 and len(pivot) >= 2:
            stat, p = friedmanchisquare(*(pivot[c].to_numpy() for c in cols))
            rows.append({"scenario": scenario, "comparison": ",".join(cols), "test": "friedman", "stat": float(stat), "p": float(p)})
            ranks = pivot[cols].rank(axis=1, method="average")
            avg = ranks.mean()
            for i, left in enumerate(cols):
                for right in cols[i + 1:]:
                    rows.append({"scenario": scenario, "comparison": f"{left} vs {right}", "test": "nemenyi-rank-difference",
                                 "stat": float(abs(avg[left] - avg[right])), "p": float("nan")})
    return pd.DataFrame(rows)


def write_statistical_report(summary_path: str, output_dir: str) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(json.loads(Path(summary_path).read_text(encoding="utf-8")))
    ci = pd.concat([confidence_frame(df, m) for m in ["mean_delay", "delivery_ratio", "max_utilization", "flow_updates"]], ignore_index=True)
    tests = paired_tests(df, "mean_delay")
    ci.to_csv(out / "confidence_intervals.csv", index=False)
    tests.to_csv(out / "statistical_tests.csv", index=False)
    return {"confidence_intervals": str(out / "confidence_intervals.csv"), "statistical_tests": str(out / "statistical_tests.csv")}
