from __future__ import annotations

"""Dataset manifests and deterministic preprocessing for Phase 3."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple
import hashlib
import json
import numpy as np


@dataclass
class DataManifest:
    name: str
    source: str
    local_path: str
    license: str
    checksum_sha256: str
    units: str
    sampling_interval: str
    notes: str = ""

    @classmethod
    def from_file(cls, name: str, source: str, path: str, license: str = "unknown",
                  units: str = "unknown", sampling_interval: str = "unknown", notes: str = ""):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return cls(name, source, path, license, digest, units, sampling_interval, notes)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def clean_traffic_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clean a traffic tensor and return an auditable exclusion report."""
    x = np.asarray(matrix, dtype=float).copy()
    report = {"input_shape": list(x.shape), "nan_count": int(np.isnan(x).sum()),
              "negative_count": int((x < 0).sum()), "excluded_time_indices": []}
    x[~np.isfinite(x)] = np.nan
    x[x < 0] = np.nan
    if x.ndim == 2:
        x = x[None, ...]
    valid = np.isfinite(x).all(axis=(1, 2))
    report["excluded_time_indices"] = np.where(~valid)[0].tolist()
    x = np.nan_to_num(x, nan=0.0)
    x[:, np.arange(x.shape[1]), np.arange(x.shape[2])] = 0.0
    report["output_shape"] = list(x.shape)
    report["kept_time_steps"] = int(valid.sum())
    return x[valid], report


def chronological_split(values: np.ndarray, train=0.70, validation=0.15):
    n = len(values)
    a = int(n * train)
    b = a + int(n * validation)
    return values[:a], values[a:b], values[b:]


def normalize_by_capacity(values: np.ndarray, capacities: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    scale = float(np.max(capacities)) if np.max(capacities) > 0 else 1.0
    return values / scale, {"capacity_scale": scale}
