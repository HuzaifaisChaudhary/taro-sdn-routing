import tempfile
import unittest
from pathlib import Path
import numpy as np
from taro.targeted_experiment import execute, make_scenario
from taro.env import RoutingEnv
from taro.policies import StableHeuristicTARO
from taro.routing import valid_table


class TargetedExperimentTest(unittest.TestCase):
    def test_stability_never_keeps_a_failed_link(self):
        scenario = make_scenario(101)
        env = RoutingEnv(scenario)
        policy = StableHeuristicTARO(update_threshold=1)
        observation = env.reset()
        policy.decide(scenario.topology, observation)
        failed = scenario.topology.edge_pairs[0]
        observation["failed"] = [failed]
        decision = policy.decide(scenario.topology, observation, [failed])
        self.assertTrue(valid_table(scenario.topology, decision.table, [failed]))
        self.assertFalse(any(failed in weights for weights in decision.table.values()))

    def test_workloads_are_repeatable_and_distinct(self):
        self.assertTrue(np.array_equal(make_scenario(101).demands, make_scenario(101).demands))
        self.assertFalse(np.array_equal(make_scenario(101).demands, make_scenario(102).demands))
        self.assertEqual(float(make_scenario(101).demands[-10:].sum()), 0)

    def test_paired_runs_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = execute(folder, seeds=range(101, 103))
            self.assertEqual(len(result["runs"]), 10)
            for row in result["runs"]:
                self.assertAlmostEqual(row["offered"], row["delivered"] + row["dropped"] + row["remaining"])
            self.assertTrue((Path(folder) / "results.json").exists())
            self.assertTrue((Path(folder) / "delivery.png").exists())
