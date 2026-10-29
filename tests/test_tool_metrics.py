import unittest

from evaluation.tool_metrics import trajectory_stats
from scripts.tools.tool_chain import is_tool_error


class TrajectoryStatsTest(unittest.TestCase):
    def test_stats(self):
        trace = [
            {
                "turn": 1,
                "calls": [
                    {"function_name": "get_leading_object_detection", "parameters": []},
                    {"function_name": "get_current_shoulder", "parameters": []},
                ],
                "observations": [
                    "Leading object detected, object id: 1",
                    "Tool environment unavailable: missing frame_token",
                ],
            }
        ]
        stats = trajectory_stats(trace)
        self.assertEqual(stats["num_turns"], 1)
        self.assertEqual(stats["num_tool_calls"], 2)
        self.assertEqual(stats["successful_tool_calls"], 1)
        self.assertEqual(stats["tool_success_rate"], 0.5)
        self.assertEqual(
            stats["tool_names"],
            ["get_leading_object_detection", "get_current_shoulder"],
        )

    def test_empty_trace(self):
        stats = trajectory_stats(None)
        self.assertEqual(stats["num_tool_calls"], 0)
        self.assertIsNone(stats["tool_success_rate"])

    def test_error_detection(self):
        self.assertTrue(
            is_tool_error("Tool get_x failed or is unavailable.")
        )
        self.assertFalse(is_tool_error("Leading object detected"))


if __name__ == "__main__":
    unittest.main()
