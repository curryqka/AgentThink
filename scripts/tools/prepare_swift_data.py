import os
import json
import argparse

def process_data(input_file, output_file, image_dir, ratio=1.0):
    """
    Process the input JSON file and generate a JSONL file with image paths and formatted content
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSONL file
        image_dir (str): Directory containing image files
    """
    # Read the merged JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chosen_idx = round(ratio * len(data))
    chosen_data = data[:chosen_idx]

    # Open output JSONL file for writing
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in chosen_data:
            question = item['question']
            # Extract filename from index and add .png extension
            filename = item['idx'].rsplit('_', 1)[0] + '.png'
            image_path = os.path.join(image_dir, filename)
            
            if args.split_type == "train":
                answer_content = item["answer"]
            elif args.split_type == "val":
                # Concatenate steps and final answer
                answer_content = "**Step-by-Step Reasoning**:\n\n" + item["steps"].strip() + "\n\n**Final Answer**: " + item["final_answer"]

            # Create JSONL entry with messages and image path
            jsonl_entry = {
                "messages": [
                    {"role": "user", "content": f"<image>{question}"},
                    {"role": "assistant", "content": f"{answer_content}"}
                ],
                "images": [image_path]
            }
            
            # Write as JSON line
            f_out.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

    print(f"JSONL file has been written to {output_file}")

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process JSON data and generate JSONL file')
    
    # Add arguments
    parser.add_argument('--input', type=str, default="/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/data/DriveLMMo1/DriveLMMo1_TRAIN.json", help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--split_type', type=str, required=True, default="train")
    parser.add_argument('--ratio', type=float, default=1.0, help='decide the size of dataset')
    
    # Parse command line arguments
    args = parser.parse_args()

    # Call processing function with parsed arguments
    process_data(args.input, args.output, args.image_dir, args.ratio)