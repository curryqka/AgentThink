import json
import os
import unittest

from scripts.tools.rewards import (
    agentthink_reward,
    extract_final_answer,
    final_answer_reward,
    step_matching_score,
    step_reasoning_reward,
    token_f1,
    tool_format_score,
    tool_integration_score,
    tool_use_reward,
)

TEXT_COMPLETION = (
    "**Step-by-Step Reasoning**:\n\n"
    "1. There is a truck ahead in the same lane.\n"
    "2. The lane divider is 1.5 m to the left.\n\n"
    "**Final Answer**: Maintain speed behind the truck."
)

TRACE_COMPLETION = (
    "**Step-by-Step Reasoning**:\n"
    "Step 1: Which object leads the ego vehicle?\n"
    "Tool: get_leading_object_detection([])\n"
    "Tool Output: Leading object: truck\n"
    "Step 2: Where is the lane divider?\n"
    "Tool: get_current_lane_divider([])\n"
    "Tool Output: Divider: 1.5 m left\n\n"
    "**Final Answer**: Maintain speed behind the truck."
)

JSON_COMPLETION = json.dumps(
    {
        "Question": "What should the ego vehicle do?",
        "Chain": [
            {
                "Tool": {"function_name": "get_leading_object_detection", "parameters": []},
                "Sub": "Which object is directly ahead?",
                "Guess_Answer": "A truck",
                "key_words": ["truck", "ahead"],
                "Missing_flag": "False",
                "next_action": "continue reasoning",
            },
            {
                "Tool": {"function_name": "get_current_lane_divider", "parameters": []},
                "Sub": "Where is the lane divider?",
                "Guess_Answer": "1.5 m left",
                "key_words": ["lane", "divider"],
                "Missing_flag": "False",
                "next_action": "conclude",
            },
        ],
        "final_answer_keywords": ["truck", "speed"],
        "final_answer": "Maintain speed behind the truck.",
    }
)


class RewardTest(unittest.TestCase):
    def test_extract_final_answer_text(self):
        self.assertEqual(
            extract_final_answer(TEXT_COMPLETION),
            "Maintain speed behind the truck.",
        )

    def test_extract_final_answer_json(self):
        self.assertEqual(
            extract_final_answer(JSON_COMPLETION),
            "Maintain speed behind the truck.",
        )

    def test_token_f1_identical(self):
        self.assertEqual(token_f1("a b c", "a b c"), 1.0)

    def test_final_answer_exact(self):
        rewards = final_answer_reward(
            [JSON_COMPLETION], ["Maintain speed behind the truck."]
        )
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.9)

    def test_final_answer_option_label(self):
        rewards = final_answer_reward(["**Final Answer**: B) Slow down"], ["B) Slow down"])
        self.assertEqual(rewards, [1.0])

    def test_step_matching(self):
        key_steps = [{"Steps": [["truck"], ["lane", "divider"]]}]
        self.assertEqual(step_matching_score(TEXT_COMPLETION, key_steps), 1.0)

    def test_step_reward_range(self):
        rewards = step_reasoning_reward(
            [TEXT_COMPLETION, JSON_COMPLETION],
            ["Maintain speed behind the truck."] * 2,
            [{"Steps": [["truck"], ["lane", "divider"]]}] * 2,
        )
        self.assertTrue(all(0.0 <= reward <= 1.0 for reward in rewards))

    def test_tool_format(self):
        self.assertEqual(tool_format_score(JSON_COMPLETION), 1.0)
        self.assertEqual(tool_format_score(TRACE_COMPLETION), 1.0)
        self.assertEqual(tool_format_score(TEXT_COMPLETION), 0.0)

    def test_tool_integration(self):
        self.assertEqual(tool_integration_score(JSON_COMPLETION), 1.0)
        self.assertEqual(tool_integration_score(TRACE_COMPLETION), 1.0)

    def test_tool_use_range(self):
        rewards = tool_use_reward([JSON_COMPLETION, TEXT_COMPLETION])
        self.assertTrue(all(0.0 <= reward <= 1.0 for reward in rewards))

    def test_combined_reward_and_weights(self):
        os.environ.pop("AGENTTHINK_REWARD_WEIGHTS", None)
        rewards = agentthink_reward(
            [JSON_COMPLETION],
            ["Maintain speed behind the truck."],
            [{"Steps": [["truck"]]}],
        )
        self.assertEqual(len(rewards), 1)
        self.assertTrue(0.0 <= rewards[0] <= 1.0)


if __name__ == "__main__":
    unittest.main()
