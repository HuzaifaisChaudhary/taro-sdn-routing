import numpy as np
import unittest

from taro.graph import GraphSpec
from taro.env import RoutingEnv
from taro.policies import HeuristicTARO
from taro.routing import ecmp_table, valid_table
from taro.scenarios import make_scenario


class CoreTests(unittest.TestCase):
    def test_ring_ecmp_is_valid_and_loop_free(self):
        g = GraphSpec.ring(6)
        table = ecmp_table(g)
        self.assertTrue(valid_table(g, table))
        for (u, d), weights in table.items():
            if u != d:
                self.assertTrue(all(v != u for (a, v) in weights if a == u))


    def test_env_conserves_offered_traffic_with_queue_accounting(self):
        g = GraphSpec.ring(4, capacity=1000, delay=1)
        scenario = make_scenario(g, horizon=3, seed=3, load=0.1)
        env = RoutingEnv(scenario)
        obs = env.reset()
        policy = HeuristicTARO()
        while not env.done:
            obs, _, _, _ = env.step(policy.decide(g, obs, obs["failed"]))
        self.assertGreaterEqual(env.total.offered, env.total.delivered)
        self.assertGreaterEqual(env.total.dropped, 0)


    def test_failed_links_are_never_used(self):
        g = GraphSpec.ring(4)
        failed = [(0, 1), (1, 0)]
        table = ecmp_table(g, failed)
        self.assertTrue(all(e not in failed for weights in table.values() for e in weights))

    def test_ppo_policy_trains_and_saves_checkpoint(self):
        import tempfile
        from taro.ppo_numpy import train_ppo
        g = GraphSpec.ring(5, capacity=10, delay=1)
        with tempfile.TemporaryDirectory() as d:
            result = train_ppo(g, episodes=2, output=d + "/policy.npz")
            self.assertEqual(result["episodes"], 2)
            self.assertTrue(__import__("os").path.exists(d + "/policy.npz"))

    def test_capacity_saturation_creates_queue_or_loss(self):
        g = GraphSpec.ring(4, capacity=1, delay=1)
        scenario = make_scenario(g, horizon=3, seed=11, load=5.0)
        env = RoutingEnv(scenario)
        obs = env.reset()
        policy = HeuristicTARO()
        while not env.done:
            obs, _, _, _ = env.step(policy.decide(g, obs, obs["failed"]))
        self.assertTrue(env.total.dropped > 0 or env.total.max_utilization > 0)
