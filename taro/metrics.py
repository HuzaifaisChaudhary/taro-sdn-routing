from __future__ import annotations

from typing import Iterable, Dict, Any
import json
import numpy as np
import pandas as pd


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(records)
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out = {}
    for name in ["offered", "delivered", "dropped", "mean_delay", "max_utilization", "rerouting_ratio", "flow_updates", "inference_ms", "loss_rate"]:
        if name in frame:
            out[name] = float(frame[name].mean())
    out["throughput"] = out.get("delivered", 0.0)
    out["delivery_ratio"] = out.get("delivered", 0.0) / max(out.get("offered", 0.0), 1e-9)
    if "mean_delay" in frame:
        out["p95_delay"] = float(np.percentile(frame["mean_delay"].to_numpy(float), 95))
    return out


def confidence_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(x.mean())
    if len(x) == 1:
        return mean, mean, mean
    from scipy.stats import t
    half = float(t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean - half, mean + half


def save_records(records, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
