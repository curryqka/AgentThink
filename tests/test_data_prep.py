import unittest

from evaluation.prepare_data import build_records, image_name


class PrepareDataTest(unittest.TestCase):
    def test_image_name(self):
        idx = "scene_token_frame_token_3"
        self.assertEqual(image_name(idx), "scene_token_frame_token.png")
        self.assertEqual(
            image_name(idx, "{stem}.jpg"), "scene_token_frame_token.jpg"
        )

    def test_build_records(self):
        records = [
            {
                "idx": "a_b_1",
                "question": "Q?",
                "final_answer": "A",
                "steps": "S",
                "tool_result": [{"name": "t"}],
                "system_prompts": "sys",
            },
            {"idx": "a_b_2", "question": "", "final_answer": "A"},
            {"question": "no id"},
        ]
        output = build_records(records, "{stem}.png")
        self.assertEqual(len(output), 1)
        record = output[0]
        self.assertEqual(record["id"], "a_b_1")
        self.assertEqual(record["image"], "a_b.png")
        self.assertEqual(record["conversations"][0]["value"], "Q?")
        self.assertEqual(record["conversations"][1]["value"], "A")
        self.assertEqual(record["tool_result"], [{"name": "t"}])


if __name__ == "__main__":
    unittest.main()
