"""Reproducible topology and traffic manifests for Phase 2 evaluation."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from .graph import GraphSpec
from .scenarios import poisson_demands


def load_traffic_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy": return np.load(p)
    if p.suffix.lower() in {".csv", ".tsv"}:
        return np.loadtxt(p, delimiter="\t" if p.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"Unsupported traffic format: {p.suffix}")


def canonical_topologies(output_dir: str = "artifacts/datasets") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    specs = {
        "NSFNET": GraphSpec.random(14, 1401, 0.24),
        "GEANT": GraphSpec.random(22, 1402, 0.18),
        "AT&T": GraphSpec.random(25, 1403, 0.16),
        "GTS": GraphSpec.random(48, 1404, 0.08),
    }
    result = {}
    for name, graph in specs.items():
        path = out / f"{name.lower().replace('&','and')}.json"
        graph.metadata.update({"source": "phase2-compatible synthetic topology placeholder", "dataset": name})
        graph.to_json(path)
        result[name] = str(path)
    manifest = {"topologies": result, "traffic": {}, "note": "Replace generated topology placeholders with licensed Topology Zoo files for final publication."}
    for name, graph in specs.items():
        traffic = poisson_demands(graph.num_nodes, 120, 2000 + graph.num_nodes, mean=4.0)
        p = out / f"{name.lower().replace('&','and')}_traffic.npy"; np.save(p, traffic); manifest["traffic"][name] = str(p)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
