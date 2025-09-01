import os
import json

# Define input and output file paths
input_file = "./data/tool_results/cot_val_Qwen2.5-VL.json"  # Change this to your actual path
output_file = "./data/DriveLMMo1_TEST_tool_results.jsonl"  # Change this to your actual path

# Read the combined JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Open the output JSONL file for writing
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data:
        question = item['question']
        # stitched multiview image stored with idx as name
        filename = item['idx'].rsplit('_', 1)[0] + '.png'
        jsonl_entry = {
            "id": item["idx"],
            "image": filename,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": f"{item['final_answer']}"
                },
                {
                    "from": "gpt",
                    "value": f"{item['steps']}"
                }
            ],
            # tool_result
            "tool_result": item.get("tool_result", []),
            # system_prompts
            "system_prompts": item.get("system_prompts", "")
        }
        # write -> json
        f_out.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

print(f"JSONL file written to {output_file}")
