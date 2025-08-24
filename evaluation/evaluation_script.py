import os
import re
import json
import ast
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# OpenAI API key (replace with a secure method)
openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

# Define input and output file paths
# input_file = "results/DriveLMMo1_internvl.json"  
input_file = "results/DriveLMMo1_250422011413.json"
output_file = "DriveLMMo1-results_result.json"  
csv_output_file = "DriveLMMo1-results_result.csv"

# dataset = "./ReasoningDriveLM_TEST.json"
dataset = "/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/data/DriveLMMo1/DriveLMMo1_TEST.json"

# System prompt for evaluation
system_prompt = """
You are an autonomous driving reasoning evaluator. Your task is to assess the alignment, coherence, and quality of reasoning steps in text responses for safety-critical driving scenarios.  

You will evaluate the model-generated reasoning using the following metrics: 

1. Faithfulness-Step (1-10)  
Measures how well the model's reasoning steps align with the ground truth.  
9-10: All steps correctly match or closely reflect the reference.  
7-8: Most steps align, with minor deviations.  
5-6: Some steps align, but several are incorrect or missing.  
3-4: Few steps align; most are inaccurate or missing.  
1-2: Majority of steps are incorrect. 

2.  Informativeness-Step (1-10)  
Measures completeness of reasoning:  
9-10: Captures almost all critical information.  
7-8: Covers most key points, with minor omissions.  
5-6: Missing significant details.  
3-4: Only partial reasoning present.  
1-2: Poor extraction of relevant reasoning. 

3.  Risk Assessment Accuracy (1-10)  
Evaluates if the model correctly prioritizes high-risk objects or scenarios.  
9-10: Correctly identifies and prioritizes key dangers.  
7-8: Mostly accurate, with minor misprioritizations.  
5-6: Some important risks are overlooked.  
3-4: Significant misjudgments in risk prioritization.  
1-2: Misidentifies key risks or misses them entirely. 

4. Traffic Rule Adherence (1-10)  
Evaluates whether the response follows traffic laws and driving best practices.  
9-10: Fully compliant with legal and safe driving practices.  
7-8: Minor deviations, but mostly correct.  
5-6: Some inaccuracies in legal/safe driving recommendations.  
3-4: Several rule violations or unsafe suggestions.  
1-2: Promotes highly unsafe driving behavior. 

5.  Scene Awareness & Object Understanding (1-10)  
Measures how well the response interprets objects, their positions, and actions.  
9-10: Clearly understands all relevant objects and their relationships.  
7-8: Minor misinterpretations but mostly correct.  
5-6: Some key objects misunderstood or ignored.  
3-4: Many errors in object recognition and reasoning.  
1-2: Misidentifies or ignores key objects. 

6.  Repetition-Token (1-10)  
Identifies unnecessary repetition in reasoning.  
9-10: No redundancy, very concise.  
7-8: Minor repetition but still clear.  
5-6: Noticeable redundancy.  
3-4: Frequent repetition that disrupts reasoning.  
1-2: Excessive redundancy, making reasoning unclear. 

7.  Hallucination (1-10)  
Detects irrelevant or invented reasoning steps not aligned with ground truth.  
9-10: No hallucinations, all reasoning is grounded.  
7-8: One or two minor hallucinations.  
5-6: Some fabricated details.  
3-4: Frequent hallucinations.  
1-2: Majority of reasoning is hallucinated. 

8.  Semantic Coverage-Step (1-10)  
Checks if the response fully covers the critical reasoning elements.  
9-10: Nearly complete semantic coverage.  
7-8: Good coverage, some minor omissions.  
5-6: Partial coverage with key gaps.  
3-4: Major gaps in reasoning.  
1-2: Very poor semantic coverage.

9. Commonsense Reasoning (1-10)  
Assesses the use of intuitive driving logic in reasoning.  
9-10: Displays strong commonsense understanding.  
7-8: Mostly correct, with minor gaps.  
5-6: Some commonsense errors.  
3-4: Frequent commonsense mistakes.  
1-2: Lacks basic driving commonsense. 

10. Missing Step (1-10)  
Evaluates if any necessary reasoning steps are missing.  
9-10: No critical steps missing.  
7-8: Minor missing steps, but answer is mostly intact.  
5-6: Some important steps missing.  
3-4: Many critical reasoning gaps.  
1-2: Response is highly incomplete. 

11. Relevance (1-10)
Measures how well the response is specific to the given scenario and ground truth. This includes whether the answer is too general or vague instead of directly addressing the key aspects of the scenario.
9-10: Highly specific and directly relevant to the driving scenario. Captures critical elements precisely, with no unnecessary generalization.
7-8: Mostly relevant, but some minor parts may be overly generic or slightly off-focus.
5-6: Somewhat relevant but lacks precision; response contains vague or general reasoning without clear scenario-based details.
3-4: Mostly generic or off-topic reasoning, with significant irrelevant content.
1-2: Largely irrelevant, missing key aspects of the scenario and failing to align with the ground truth.

12. Missing Details (1-10)
Evaluates the extent to which critical information is missing from the response, impacting the reasoning quality.
9-10: No significant details are missing; response is comprehensive and complete.
7-8: Covers most important details, with minor omissions that do not severely impact reasoning.
5-6: Some essential details are missing, affecting the completeness of reasoning.
3-4: Many critical reasoning steps or contextual details are absent, making the response incomplete.
1-2: Response is highly lacking in necessary details, leaving major gaps in understanding.

Final Evaluation:  

Compute the Overall Score as the average of all metric scores.  

Avoid subjective interpretation and adhere to the given thresholds. 

Always strictly follow these scoring guidelines.  

Do not add any additional explanations beyond the structured JSON output. 

Example output format:  

{
  "Faithfulness-Step": 6.0,
  "Informativeness-Step": 6.5,
  "Risk Assessment Accuracy": 7.0,
  "Traffic Rule Adherence": 7.5,
  "Scene Awareness & Object Understanding": 8.0,
  "Repetition-Token": 7.0,
  "Hallucination": 8.5,
  "Semantic Coverage-Step": 7.5,
  "Commonsense Reasoning": 7.0,
  "Missing Step": 8.5,
  "Relevance": 8.5,
  "Missing Details": 7.0,
  "Overall Score": 7.42
}

"""


# OpenAI Response Format
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvaluationScores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Faithfulness-Step": {"type": "number"},
                "Informativeness-Step": {"type": "number"},
                "Risk Assessment Accuracy": {"type": "number"},
                "Traffic Rule Adherence":{"type": "number"},  
                "Scene Awareness & Object Understanding": {"type": "number"},
                "Repetition-Token": {"type": "number"},
                "Hallucination": {"type": "number"},
                "Semantic Coverage-Step": {"type": "number"},
                "Commonsense": {"type": "number"},
                "Missing Step": {"type": "number"},
                "Relevance": {"type": "number"},
                "Missing Details": {"type": "number"},
                "Overall Score": {"type": "number"}
            },
            "required": [
                "Faithfulness-Step",
                "Informativeness-Step",
                "Risk Assessment Accuracy",
                "Traffic Rule Adherence",
                "Scene Awareness & Object Understanding",
                "Repetition-Token",
                "Hallucination",
                "Semantic Coverage-Step",
                "Commonsense",
                "Missing Step",
                "Relevance",
                "Missing Details",
                "Overall Score"
            ],
            "additionalProperties": False
        }
    }
}


def extract_final_answer(text):
    options = ["The final answer is:","**Final Answer:**","Final Answer","Answer","Why take this action?:",
               "**Final Answer**","**Final Decision**:","Final Step:", "<CONCLUSION>"
        ]
    final_ans = ""
    for opt in options:
        if opt in text:
            final_ans = text.split(opt)[-1]
            break
    return final_ans

def extract_options(text):
    pattern = r'([A-F])\)\s*(.+)'
    matches = re.findall(pattern, text)
    options = [(label, option) for label, option in matches]
    if not options:
        if "none of the " in text.lower():
            return [('F', 'None of the option.')]
        return [('none','none')]
    return options


# Function to evaluate reasoning
def evaluate_steps(question, ground_truth, llm_response):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {question}\nGround Truth: {ground_truth}\nLLM Response: {llm_response}"}
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format=response_format,
        max_tokens=500,
        temperature=0.0,
    )
    
    return response.choices[0].message.content

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    
with open(dataset, "r", encoding="utf-8") as f:
    dataset = json.load(f)

dataset = dataset[:3]
valid_ids = [item["idx"] for item in dataset if "idx" in item]
data_dict = {}
for item in dataset:
    key = item["idx"]
    data_dict[key] = {
        "image": item["image"],
        "question": item["question"],
        "final_answer": item["final_answer"],
        "steps": item["steps"]
    }

# List to store results
evaluation_results = []

# Function to process each file
def process_file(item):
    # breakpoint()
    question = item["question"]
    try:
        idx = item["idx"]
    except:
        idx = item["id"]
    
    if idx not in valid_ids:
        return None
    
    ground_truth = "**Step-by-Step Reasoning**: "+ data_dict[idx]['steps'] + " **Final Answer**: " + data_dict[idx]["final_answer"]
    try:
        llm_response = item["llm-response"]
    except:
        llm_response = item["answer"]
            
    gt_final_answer = data_dict[idx]['final_answer']
    llm_final_answer = extract_final_answer(llm_response)
    
    # set mcq -1 for non MCQ questions
    mcq = -1
    gt = ""
    llm = ""
    if int(idx.rsplit("_",1)[-1]) in [2,3,4,5,8]:
        gt, _ = extract_options(gt_final_answer)[0]
        llm, _ = extract_options(llm_final_answer)[0]
        
        if llm == gt:
            mcq = 1
        else:
            mcq = 0
            
    # Evaluate steps
    res = {}
    try:
        res = evaluate_steps(question, ground_truth, llm_response)
        res = ast.literal_eval(res)
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        return None
    
    res["id"] = idx
    res["question"] = question
    res["ground_truth"] = ground_truth
    res["llm_response"] = llm_response
    res["mcq"] = mcq
    
    return res

# Process files in parallel
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_file, data), total=len(data), desc="Processing"))

# Remove None values from results
results = [r for r in results if r is not None]

# Save results to JSON
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=4, ensure_ascii=False)

# Convert results to DataFrame and save as CSV
df = pd.DataFrame(results)
df.to_csv(csv_output_file, index=False)

# Accuracy for MCQs
df_mcq = df[df["mcq"] != -1]
mcq_accuracy = df_mcq["mcq"].mean() 
print(f"MCQ Accuracy: {mcq_accuracy:.2%}") 

# Normalized step evaluation
max_score = 10  
percentage_correct = (df["Overall Score"].sum() / (len(df) * max_score)) * 100
print(f"Percentage of correct steps (normalized): {percentage_correct:.2f}%")

print(f"Results saved: JSON -> {output_file}, CSV -> {csv_output_file}")
