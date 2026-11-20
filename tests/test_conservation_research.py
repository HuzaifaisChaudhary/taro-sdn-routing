import unittest
import numpy as np
from taro.env import RoutingEnv
from taro.graph import GraphSpec
from taro.routing import ECMPPolicy
from taro.scenarios import Scenario


class DestinationConservation(unittest.TestCase):
    def test_destinations_share_capacity_without_losing_identity(self):
        graph = GraphSpec.line(3, capacity=1)
        demand = np.zeros((10, 3, 3))
        demand[0, 0, 1] = 2
        demand[0, 0, 2] = 1
        env = RoutingEnv(Scenario(graph, demand, [[] for _ in demand]))
        observation = env.reset()
        while not env.done:
            decision = ECMPPolicy().decide(graph, observation, observation["failed"])
            observation, _, _, info = env.step(decision)
            self.assertAlmostEqual(env.total.offered, env.total.delivered + env.total.dropped + info["queued"])
            self.assertTrue(all(v <= env.capacity[e] * env.config.control_interval + 1e-9
                                for e, v in env._step_served.items()))
        self.assertAlmostEqual(env.total.delivered, 3.0)

    def test_failure_accounts_for_queued_traffic(self):
        graph = GraphSpec.line(3, capacity=1)
        demand = np.zeros((6, 3, 3))
        demand[0, 0, 2] = 4
        failures = [[], [(0, 1), (1, 0)], [], [], [], []]
        env = RoutingEnv(Scenario(graph, demand, failures))
        observation = env.reset()
        while not env.done:
            observation, _, _, info = env.step(ECMPPolicy().decide(graph, observation, observation["failed"]))
            self.assertAlmostEqual(env.total.offered, env.total.delivered + env.total.dropped + info["queued"])
