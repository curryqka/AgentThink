import unittest

from evaluation.prepare_data import build_records, image_name
from scripts.tools.build_sft_data import build_record


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

    def test_grpo_record_carries_rollout_metadata(self):
        item = {
            "idx": "scene_token_frame_token_1",
            "question": "Q?",
            "final_answer": "A",
            "cot_data": {"Chain": []},
            "image": ["samples/CAM_FRONT/x.jpg"],
        }
        record = build_record(
            item,
            fmt="agentthink",
            target="grpo",
            image_dir="data/image2concat",
            include_observations=False,
            tool_data_path="data/tool_results",
            pickle_split="train",
        )
        self.assertEqual(record["frame_token"], "token")
        self.assertEqual(record["pickle_split"], "train")
        self.assertEqual(record["tool_data_path"], "data/tool_results")
        self.assertEqual(record["camera_paths"], ["samples/CAM_FRONT/x.jpg"])
        self.assertNotIn("assistant", [m["role"] for m in record["messages"]])


if __name__ == "__main__":
    unittest.main()
