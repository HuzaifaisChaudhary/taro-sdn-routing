from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json


@dataclass
class TAROConfig:
    control_interval: float = 1.0
    history_length: int = 4
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    rollout_length: int = 1024
    minibatch_size: int = 128
    max_grad_norm: float = 0.5
    max_weight_change: float = 0.1
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "delay": 1.0, "loss": 1.0, "utilization": 1.0,
        "rerouting": 1.0, "flow_updates": 1.0,
    })
    seed: int = 7
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TAROConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
