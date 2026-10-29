import unittest

from evaluation.prompts import (
    JUDGE_METRIC_KEYS,
    JUDGE_RESPONSE_FORMAT,
    JUDGE_SYSTEM_PROMPT,
    PAPER_METRIC_COLUMNS,
    TOOL_JUDGE_RESPONSE_FORMAT,
    build_tool_cot_prompt,
)


class PromptTest(unittest.TestCase):
    def test_tool_prompt_contains_catalog(self):
        prompt = build_tool_cot_prompt("- get_x() # does x")
        self.assertIn("get_x()", prompt)
        self.assertIn('"Chain"', prompt)

    def test_judge_schema_matches_keys(self):
        schema = JUDGE_RESPONSE_FORMAT["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]) - {"Overall Score"}, set(JUDGE_METRIC_KEYS))
        for key in JUDGE_METRIC_KEYS:
            self.assertIn(key, JUDGE_SYSTEM_PROMPT)
        self.assertIn("Overall Score", JUDGE_SYSTEM_PROMPT)

    def test_paper_columns_exist_in_schema(self):
        schema = JUDGE_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]
        for key in PAPER_METRIC_COLUMNS.values():
            self.assertIn(key, schema)

    def test_tool_judge_schema(self):
        schema = TOOL_JUDGE_RESPONSE_FORMAT["json_schema"]["schema"]
        self.assertEqual(len(schema["properties"]), 4)


if __name__ == "__main__":
    unittest.main()
