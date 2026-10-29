import unittest

from scripts.tools.tool_chain import (
    extract_json_object,
    iter_tool_calls,
    parse_tool_chain,
    strip_markdown_fences,
    validate_tool_chain,
)

VALID_CHAIN = {
    "Question": "What is ahead?",
    "Chain": [
        {
            "Tool": {"function_name": "get_leading_object_detection", "parameters": []},
            "Sub": "Which object leads the ego vehicle?",
            "Guess_Answer": "A truck",
            "key_words": ["truck", "vehicle"],
            "Missing_flag": "False",
            "next_action": "continue reasoning",
        },
        {
            "Tool": {"function_name": "get_current_lane_divider", "parameters": []},
            "Sub": "How wide is the current lane?",
            "Guess_Answer": "",
            "key_words": ["lane", "divider"],
            "Missing_flag": "True",
            "next_action": "conclude",
        },
    ],
    "final_answer_keywords": ["truck"],
    "final_answer": "A truck is ahead.",
}


class ToolChainTest(unittest.TestCase):
    def test_strip_fences(self):
        self.assertEqual(strip_markdown_fences('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_extract_object_ignores_prefix(self):
        text = 'Here is the chain: {"final_answer": "yes"} thanks'
        self.assertEqual(extract_json_object(text), '{"final_answer": "yes"}')

    def test_parse_valid_chain(self):
        self.assertEqual(parse_tool_chain("```json\n{}\n```"), {})

    def test_parse_invalid_chain(self):
        self.assertIsNone(parse_tool_chain("not json at all"))

    def test_validate_valid(self):
        self.assertEqual(validate_tool_chain(VALID_CHAIN), [])

    def test_validate_missing_chain(self):
        self.assertTrue(validate_tool_chain({"final_answer": "x"}))

    def test_validate_bad_next_action(self):
        chain = {
            "Chain": [
                {
                    "Tool": {"function_name": "x", "parameters": []},
                    "Sub": "s",
                    "next_action": "maybe",
                }
            ]
        }
        errors = validate_tool_chain(chain)
        self.assertTrue(any("next_action" in error for error in errors))

    def test_iter_tool_calls(self):
        calls = list(iter_tool_calls(VALID_CHAIN["Chain"]))
        self.assertEqual(calls[0][0], "get_leading_object_detection")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
