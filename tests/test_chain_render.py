import unittest

from scripts.tools.chain_render import render_plain, render_tool_cot, render_trace

COT = {
    "Chain": [
        {
            "Tool": {"function_name": "get_leading_object_detection", "parameters": []},
            "Sub": "Which object leads the ego vehicle?",
            "Guess_Answer": "A truck",
            "next_action": "continue reasoning",
        },
        {
            "Tool": {"function_name": "get_current_lane_divider", "parameters": ["CAM_FRONT"]},
            "Sub": "Where is the lane divider?",
            "Guess_Answer": "Left",
            "next_action": "conclude",
        },
    ],
    "final_answer": "Keep the current speed.",
}

TOOL_RESULTS = [
    {"name": "get_leading_object_detection", "args": {}, "prompt": "Leading object: truck"},
    {"name": "get_current_lane_divider", "args": {}, "prompt": "Divider: 1.5 m left"},
]


class RenderTest(unittest.TestCase):
    def test_plain_from_steps(self):
        text = render_plain({"steps": "Step one.", "final_answer": "Go."})
        self.assertIn("**Step-by-Step Reasoning**:", text)
        self.assertIn("Step one.", text)
        self.assertIn("**Final Answer**: Go.", text)

    def test_plain_from_answer_markers(self):
        text = render_plain(
            {"answer": "**Step-by-Step Reasoning**:\nLook.\n**Final Answer**: Go."}
        )
        self.assertIn("Look.", text)
        self.assertIn("Go.", text)

    def test_tool_cot_is_json(self):
        text = render_tool_cot(COT)
        self.assertTrue(text.strip().startswith("{"))
        self.assertIn('"final_answer"', text)

    def test_trace_with_observations(self):
        text = render_trace(COT, TOOL_RESULTS)
        self.assertIn("Step 1: Which object leads the ego vehicle?", text)
        self.assertIn("Tool: get_leading_object_detection([])", text)
        self.assertIn("Tool Output: Leading object: truck", text)
        self.assertIn("**Final Answer**: Keep the current speed.", text)

    def test_trace_without_observations(self):
        text = render_trace(COT, TOOL_RESULTS, include_observations=False)
        self.assertNotIn("Tool Output:", text)
        self.assertIn("Tool: get_current_lane_divider", text)


if __name__ == "__main__":
    unittest.main()
