import re
import torch
import torch.nn.functional as F
from collections import Counter

def key_step_single(content, key_step):
    """
    Check if the key steps are present in the content.
    """
    if isinstance(key_step, list):
        key_step = key_step[0]
    # step_texts = re.findall(r"### Step \d+:\s*(.*?)(?=\n###|\Z)", content, re.DOTALL)
    step_texts = extract_key_steps(content)
    # step_text_combined = " ".join(step_texts).lower()
    
    steps_keywords = key_step.get("Steps", [])
    total_keywords = len(steps_keywords)

    matched_keywords = 0
    for group in steps_keywords:
        if isinstance(group, str):
            group = [group]
        if any(alt.lower() in step_texts for alt in group):
            matched_keywords += 1
    breakpoint()
    reward = matched_keywords / total_keywords if total_keywords > 0 else 0.0
    return reward

def extract_key_steps(text):
    """
    Extract key steps from the text based on various possible markers.
    """
    options = ["Step-by-Step Reasoning:", "**Step-by-Step Reasoning**:", "### Step 1:", "### Step 2:", "### Step 3:", "### Step 4:", "### Step 5:"]
    key_steps = ""
    for opt in options:
        if opt in text:
            key_steps = text.split(opt)[-1]
            break
    return key_steps.strip()

def extract_final_answer(text):
    """
    Extract the final answer from the text based on various possible markers.
    """
    options = ["The final answer is:", "**Final Answer:**", "**Final Answer**:", "Final Answer", "Answer", "Why take this action?:",
               "**Final Answer**", "**Final Decision**:","Final Step:", "<CONCLUSION>"]
    final_ans = ""
    for opt in options:
        if opt in text:
            final_ans = text.split(opt)[-1]
            break
    return final_ans.strip()

def extract_options(text):
    """
    Extract options from the text.
    """
    pattern = r'([A-F])\)\s*(.+)'
    matches = re.findall(pattern, text)
    options = [(label, option) for label, option in matches]
    if not options:
        if "none of the " in text.lower():
            return [('F', 'None of the option.')]
        return [('none','none')]
    return options

def description_accuracy_reward(completions, solution, key_steps, **kwargs):
    """Reward function that checks if the completion contains key phrases."""
    contents = [completion for completion in completions]
    rewards = []
    
    # Extract keywords from solution
    sol_final_answer = extract_final_answer(solution[0])
    keywords = ["ego vehicle", "car directly ahead", "distance management", "barrier", "restricted access", "pedestrian", "safety", "navigation", "path", "caution", "speed"]
    
    for content in contents:
        reward = 0.0
        student_final_answer = extract_final_answer(content)
        # Check for presence of keywords
        matched_keywords = [word for word in keywords if word in student_final_answer.lower()]
        reward = len(matched_keywords) / len(keywords) if keywords else 0.0
        
        rewards.append(reward)
     
    return rewards


def description_accuracy_rewardV2(completions, solution, key_steps, **kwargs):
    """Reward function that checks if the completion contains key phrases and key steps."""
    contents = [completion for completion in completions]
    rewards = []
    
    # Extract keywords from solution
    sol_final_answer = extract_final_answer(solution[0])
    keywords = ["ego vehicle", "car directly ahead", "distance management", "barrier", "restricted access", "pedestrian", "safety", "navigation", "path", "caution", "speed"]
    
    for content in contents:
        reward = 0.0
        student_final_answer = extract_final_answer(content)
        # Check for presence of keywords
        matched_keywords = [word for word in keywords if word in student_final_answer.lower()]
        keyword_reward = len(matched_keywords) / len(keywords) if keywords else 0.0
        
        # Check for key steps
        if key_steps:
            step_reward = key_step_single(content, key_steps)
            reward = 0.5 * keyword_reward + 0.5 * step_reward
        else:
            reward = keyword_reward
        
        rewards.append(reward)
    
    return rewards

def process_text_embed(model, tokenizer, sentence):


    # model_path = "/high_perf_store/mlinfra-vepfs/qiankangan/DriveLMM-o1-main/all-MiniLM-L6-v2"  # 确保这是模型文件夹的绝对路径
    # model = SentenceTransformer(model_path)
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    
    # breakpoint()
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # breakpoint()
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def description_semantic_reward(completions, solution, key_steps, **kwargs):
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('all-MiniLM-L6-v2')

    """Reward function that uses semantic similarity to evaluate the answer."""
    contents = [completion for completion in completions]
    rewards = []
    
    # Encode the solution
    sol_final_answer = extract_final_answer(solution[0])
    solution_embedding = process_text_embed(model, tokenizer, [sol_final_answer])
    
    for content in contents:
        reward = 0.0
        
        student_final_answer = extract_final_answer(content)
        # Encode the student answer
        student_embedding = process_text_embed(model, tokenizer, [student_final_answer])
        # Calculate cosine similarity
        breakpoint()
        similarity = util.cos_sim(solution_embedding, student_embedding)
        reward = float(similarity)
        breakpoint()
        rewards.append(reward)
    
    return rewards

def mc_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct for multiple-choice questions."""
    contents = [completion for completion in completions]
    rewards = []
    
    # Extract correct answer from solution
    sol_options = extract_options(solution[0])
    breakpoint()
    sol_correct_label = None
    for label, option in sol_options:
        if "correct" in option.lower() or "answer" in option.lower():
            sol_correct_label = label
            break
    if not sol_correct_label:
        sol_final_answer = extract_final_answer(solution[0])
        sol_correct_label_match = re.search(r'([A-F])', sol_final_answer)
        sol_correct_label = sol_correct_label_match.group(1) if sol_correct_label_match else None
    
    breakpoint()
    for content in contents:
        reward = 0.0
        
        # Extract student's answer
        student_final_answer = extract_final_answer(content)
        student_label_match = re.search(r'([A-F])', student_final_answer)
        student_label = student_label_match.group(1) if student_label_match else None
        
        # Compare labels
        if student_label and sol_correct_label and student_label == sol_correct_label:
            reward = 1.0
        
        rewards.append(reward)
    breakpoint()
    return rewards

def check_reasoning_completeness(text):
    """
    Check if the text contains all required sections.
    """
    required_sections = [
        "Step-by-Step Reasoning",
        "Final Answer"
    ]
    # breakpoint()
    missing_sections = [section for section in required_sections if section not in text]
    # breakpoint()
    return not missing_sections

def check_reasoning_logic(text):
    """
    Check if the sections appear in the correct order by finding their positions.
    """
    # Find the start and end positions of '**Step-by-Step Reasoning**' section
    reasoning_steps_start = text.find("**Step-by-Step Reasoning**")
    if reasoning_steps_start == -1:
        return False  # '**Reasoning Steps**' not found
    
    reasoning_steps_end = text.find("**Final Answer**", reasoning_steps_start)
    if reasoning_steps_end == -1:
        return False  # '**Final Answer**' not found after '**Reasoning Steps**'
    
    # Find the start position of '**Final Answer**' section
    final_answer_start = text.find("**Final Answer**")
    if final_answer_start == -1 or final_answer_start < reasoning_steps_start:
        return False  # '**Final Answer**' not found or not after '**Reasoning Steps**'
    
    # Check if there is content between '**Reasoning Steps**' and '**Final Answer**'
    reasoning_content = text[reasoning_steps_start:final_answer_start].strip()
    if not reasoning_content:
        return False  # No content in '**Reasoning Steps**' section
    
    # Check if there is content after '**Final Answer**'
    final_answer_content = text[final_answer_start:].strip()
    if not final_answer_content:
        return False  # No content in '**Final Answer**' section
    
    return True

def description_validity_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Check if answer format is correct
    pattern = r"\*\*Step-by-Step Reasoning\*\*:\n\n(\d+\. .+\n\n)+\*\*Final Answer\*\*: .+"
    completion_contents = [completion for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    
    # Check if the content includes all required sections and in correct order
    validity_rewards = []
    for content in completion_contents:
        if not check_reasoning_completeness(content):
            validity_rewards.append(0.0)
            continue
        if not check_reasoning_logic(content):
            validity_rewards.append(0.0)
            continue
        validity_rewards.append(1.0)
    breakpoint()
    # Combine format and validity rewards
    rewards = []
    for format_match, validity in zip(matches, validity_rewards):
        if format_match:
            rewards.append(0.5 + validity * 0.5)  # 0.5 for format, 0.5 for validity
        else:
            rewards.append(validity)  # Only validity reward if format is incorrect
    breakpoint()
    return rewards

if __name__ == "__main__":
    """debug description"""
    solution = ["**Step-by-Step Reasoning**:\n\n1. In the top middle image, the ego vehicle is right behind another vehicle. This car is directly in its path and needs to be considered for speed and distance management.\n2. The top middle image also shows a barrier ahead. This indicates restricted access which the ego vehicle must consider before proceeding.\n3. The top right image includes a person standing near the road. The ego vehicle should be cautious of pedestrians and maintain a safe distance. \n\n**Final Answer**: The three most important objects the ego vehicle must consider are the car directly ahead, the barrier indicating restricted access, and the pedestrian standing on the side of the road. These objects are crucial for safety and navigation."]

    completion = ["**Step-by-Step Reasoning**:\n\n1. In the 'front' image, there is a car directly ahead of the ego vehicle. This is crucial for the ego vehicle to consider due to potential obstacles or traffic flow.\n\n2. In the 'front right' image, there are pedestrians visible on the pathway. The presence of pedestrians is significant as they may cross the path of the ego vehicle or be influenced its movement.\n\n3. In the 'back right' image, there is another vehicle approaching from the opposite direction. This vehicle is important for the ego vehicle to monitor to avoid collisions and ensure safe passage.\n\n**Final Answer**: The three most important objects in the scene are:\n1. The car directly in front of the ego vehicle in the 'front' image.\n2. The pedestrians on the sidewalk in the 'front right' image.\n3. The approaching vehicle in the 'back right' image."]

    key_steps = [
        {
            "Steps": [
                ["car", "directly ahead"],
                ["barrier", "ahead", "restricted access"],
                ["person", "near road", " pedestrians"]
            ]
        }
    ]
    description_validity_reward(completion)
    # description_accuracy_rewardV2(completions=completion, solution=solution, key_steps=key_steps)
    # description_semantic_reward(completions=completion, solution=solution, key_steps=key_steps)
    """debug mc"""
    # solution = ["**Step-by-Step Reasoning**:\n\n1. In the \"front\" image of the top row, there is a black car ahead, in a lane that is narrowed by a divider. This suggests limited space for maneuvering.\n2. In the \"back\" image of the bottom row, the car is in a lane that has space on the left, but the right side is blocked by another car.\n3. Considering the options, \"Change lane to right\" (D) is not feasible due to the blockage. \"Come to a complete stop\" (A) or \"Brake suddenly\" (B) might not ensure collision avoidance given the proximity of the car ahead.\n4. \"Speed up\" (C) is not ideal in a potentially congested area. The safest way could be to \"Change lane to left\" (E) since it appears more space is available.\n\n**Final Answer**: E) Change lane to left."]
    # completion = ["**Step-by-Step Reasoning**:\n\n1. In the 'front' image, there's a car directly ahead of the ego vehicle, which indicates potential for collision if movement continues.\n2. The current speed of the ego vehicle is very low at 0.82 km/h, suggesting that stopping would be effective without skidding or abrupt maneuvers.\n3. There are no visible obstacles or safe lanes to the left or right that are clear for immediate lane changes.\n4. Given the proximity to the car ahead and lack of safe maneuver spaces, continuing forward is unsafe.\n\n**Final Answer**: A) Come to a complete stop."]
    # completion = ["**Step-by-Step Reasoning**:\n\n1. In the 'front' image, there's a car directly ahead of the ego vehicle, which indicates potential for collision if movement continues.\n2. The current speed of the ego vehicle is very low at 0.82 km/h, suggesting that stopping would be effective without skidding or abrupt maneuvers.\n3. There are no visible obstacles or safe lanes to the left or right that are clear for immediate lane changes.\n4. Given the proximity to the car ahead and lack of safe maneuver spaces, continuing forward is unsafe.\n\n**Final Answer**: E) Change lane to left."]
    # mc_accuracy_reward(completions=completion, solution=solution)
    
    


