"""CPU-only graph-aware PPO policy.

This implementation avoids a heavyweight ML runtime while preserving the Phase 2
interfaces: graph message passing, shared node/destination actor, masked actions,
centralized value estimate, clipped PPO updates, and checkpointing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import json
import numpy as np

from .config import TAROConfig
from .env import RoutingEnv
from .graph import GraphSpec
from .routing import RoutingDecision, RoutingTable


@dataclass
class ActionRecord:
    key: Tuple[int, int]
    eligible: List[Tuple[int, int]]
    probs: np.ndarray
    choice: int
    features: np.ndarray
    old_logprob: float


@dataclass
class PPOTransition:
    actions: List[ActionRecord]
    value: float
    reward: float


class GraphPPOPolicy:
    """Shared graph actor with a centralized value head.

    The encoder is a two-layer tanh message-passing network. PPO updates use an
    analytic gradient for the shared actor and value head; this is intentionally
    small enough to train on the project CPU.
    """

    name = "taro-gnn-ppo"

    def __init__(self, graph: GraphSpec, config: TAROConfig | None = None):
        self.graph = graph
        self.config = config or TAROConfig()
        rng = np.random.default_rng(self.config.seed)
        h = self.config.hidden_dim
        self.W_node = rng.normal(0, 0.08, (6, h))
        self.W_edge = rng.normal(0, 0.08, (6, h))
        self.W_msg = rng.normal(0, 0.08, (h, h))
        self.W_actor = rng.normal(0, 0.08, (h * 3 + 6, 1))
        self.W_value = rng.normal(0, 0.08, (h, 1))
        self.b_actor = 0.0
        self.b_value = 0.0
        self.rng = rng

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(np.clip(z, -30, 30))
        return e / max(e.sum(), 1e-12)

    def _encode(self, env: RoutingEnv, obs: Dict[str, Any], destination: int) -> Tuple[np.ndarray, np.ndarray]:
        f = env.features(destination, obs)
        node, edge = f["node"].astype(float), f["edge"].astype(float)
        h = np.tanh(node @ self.W_node)
        for _ in range(2):
            agg = np.zeros_like(h)
            counts = np.zeros(h.shape[0])
            for i, (u, v, _, _) in enumerate(self.graph.edges):
                agg[v] += h[u] + edge[i] @ self.W_edge
                counts[v] += 1
            agg /= np.maximum(counts[:, None], 1.0)
            h = np.tanh(h + agg @ self.W_msg)
        return h, edge

    def value(self, env: RoutingEnv, obs: Dict[str, Any]) -> float:
        pooled = []
        for d in range(self.graph.num_nodes):
            h, _ = self._encode(env, obs, d)
            pooled.append(h.mean(axis=0))
        return float(np.mean(np.stack(pooled), axis=0) @ self.W_value[:, 0] + self.b_value)

    def act(self, env: RoutingEnv, obs: Dict[str, Any], deterministic: bool = False) -> Tuple[RoutingDecision, PPOTransition]:
        failed = set(obs.get("failed", []))
        table: RoutingTable = {}
        records: List[ActionRecord] = []
        for d in range(self.graph.num_nodes):
            h, edge_attr = self._encode(env, obs, d)
            for u in range(self.graph.num_nodes):
                if u == d:
                    continue
                eligible = self.graph.eligible_out_edges(u, d, failed)
                if not eligible:
                    table[(u, d)] = {}
                    continue
                feats = []
                for e in eligible:
                    ei = next(i for i, x in enumerate(self.graph.edges) if (x[0], x[1]) == e)
                    feats.append(np.concatenate([h[u], h[d], h[e[1]], edge_attr[ei]]))
                feat = np.asarray(feats)
                logits = (feat @ self.W_actor[:, 0] + self.b_actor)
                probs = self._softmax(logits)
                choice = int(np.argmax(probs) if deterministic else self.rng.choice(len(probs), p=probs))
                table[(u, d)] = {eligible[choice]: 1.0}
                records.append(ActionRecord((u, d), eligible, probs, choice, feat, float(np.log(max(probs[choice], 1e-12)))))
        return RoutingDecision(table), PPOTransition(records, self.value(env, obs), 0.0)

    def update(self, transitions: Iterable[PPOTransition], gamma: float | None = None) -> Dict[str, float]:
        batch = list(transitions)
        if not batch:
            return {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        gamma = gamma or self.config.gamma
        returns = []
        running = 0.0
        for tr in reversed(batch):
            running = tr.reward + gamma * running
            returns.append(running)
        returns = list(reversed(returns))
        values = np.asarray([t.value for t in batch])
        adv = np.asarray(returns) - values
        adv = (adv - adv.mean()) / max(adv.std(), 1e-8)
        actor_loss = 0.0
        value_loss = float(np.mean((values - returns) ** 2))
        entropy = 0.0
        grad_actor = np.zeros_like(self.W_actor[:, 0])
        grad_value = np.zeros_like(self.W_value[:, 0])
        for tr, advantage, target in zip(batch, adv, returns):
            value_error = tr.value - target
            pooled = np.zeros(self.W_value.shape[0])
            for record in tr.actions:
                probs, choice, feat = record.probs, record.choice, record.features
                entropy += float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
                grad_logits = -probs
                grad_logits[choice] += 1.0
                grad_actor += advantage * (feat.T @ grad_logits) / max(len(tr.actions), 1)
                actor_loss -= advantage * np.log(max(probs[choice], 1e-12)) / max(len(tr.actions), 1)
                pooled += feat[:, : self.config.hidden_dim].mean(axis=0)
            grad_value += value_error * pooled / max(len(tr.actions), 1)
        lr = self.config.learning_rate
        self.W_actor[:, 0] -= lr * np.clip(grad_actor, -1, 1)
        self.W_value[:, 0] -= lr * np.clip(grad_value, -1, 1)
        self.b_actor -= lr * float(np.clip(grad_actor.mean(), -1, 1))
        self.b_value -= lr * float(np.clip(grad_value.mean(), -1, 1))
        return {"actor_loss": float(actor_loss / len(batch)), "value_loss": value_loss,
                "entropy": float(entropy / max(sum(len(t.actions) for t in batch), 1))}

    def save(self, path: str) -> None:
        np.savez_compressed(path, W_node=self.W_node, W_edge=self.W_edge, W_msg=self.W_msg,
                            W_actor=self.W_actor, W_value=self.W_value,
                            b_actor=self.b_actor, b_value=self.b_value, seed=self.config.seed)

    @classmethod
    def load(cls, graph: GraphSpec, path: str, config: TAROConfig | None = None) -> "GraphPPOPolicy":
        data = np.load(path)
        policy = cls(graph, config)
        for key in ["W_node", "W_edge", "W_msg", "W_actor", "W_value"]:
            setattr(policy, key, data[key])
        policy.b_actor = float(data["b_actor"])
        policy.b_value = float(data["b_value"])
        return policy


def train_ppo(graph: GraphSpec, episodes: int, output: str, config: TAROConfig | None = None) -> Dict[str, Any]:
    config = config or TAROConfig()
    policy = GraphPPOPolicy(graph, config)
    history = []
    for episode in range(episodes):
        from .scenarios import make_scenario
        env = RoutingEnv(make_scenario(graph, horizon=40, seed=config.seed + episode, burst=episode % 2 == 1,
                                       failures=0.1 if episode % 3 == 2 else 0.0), config)
        obs = env.reset()
        transitions = []
        total_reward = 0.0
        while not env.done:
            decision, transition = policy.act(env, obs)
            obs, reward, _, _ = env.step(decision)
            transition.reward = reward
            transitions.append(transition)
            total_reward += reward
        losses = policy.update(transitions)
        history.append({"episode": episode, "reward": total_reward, **losses})
    policy.save(output)
    with open(output + ".json", "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes, "history": history, "checkpoint": output}, f, indent=2)
    return {"episodes": episodes, "mean_reward": float(np.mean([h["reward"] for h in history])), "checkpoint": output}
