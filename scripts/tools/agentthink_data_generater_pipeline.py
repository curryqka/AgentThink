# Copyright (c) Kangan Qian. All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: Script for generating chain-of-thought data using OpenAI API

import json
import os
import argparse
import logging
import time
import threading
import functools
import concurrent.futures
import platform
from tqdm import tqdm
from typing import Any, List, Tuple, Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from concurrent.futures import ThreadPoolExecutor
from scripts.tools.tool_libraries import FuncAgent
from openai import OpenAI

# Initialize OpenAI client with placeholder API key
client = OpenAI(
    api_key="your_api_key_here",
    base_url="https://api.example.com/v1"
)


class TimeoutError(Exception):
    """Custom exception for function timeout"""
    pass


def timeout(seconds):
    """
    Decorator: Adds timeout mechanism to decorated functions.
    
    Uses signal.SIGALRM on Unix systems and threading on Windows.
    
    Args:
        seconds (int): Timeout duration in seconds
    """
    def decorator(func):
        if platform.system() != "Windows":
            # Unix implementation using signals
            import signal
            
            def _handle(signum, frame):
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                old_handler = signal.signal(signal.SIGALRM, _handle)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrapper
        else:
            # Windows implementation using threading
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = [TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        result[0] = e
                
                t = threading.Thread(target=target, daemon=True)
                t.start()
                t.join(seconds)
                
                if t.is_alive():
                    raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")
                if isinstance(result[0], Exception):
                    raise result[0]
                return result[0]
            return wrapper
    return decorator


@retry(wait=wait_random_exponential(min=3, max=10), stop=stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    """Call OpenAI API with exponential backoff retry strategy"""
    return client.chat.completions.create(**kwargs)


def read_json(json_file: str) -> Any:
    """Read JSON file and return its content"""
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def save_progress(file_path: str, index: int) -> None:
    """
    Save processing progress to a JSON file
    
    Args:
        file_path: Path to save progress file
        index: Current processing index
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({"processed_index": index}, f, ensure_ascii=False, indent=4)


def get_system_prompt() -> str:
    """Generate system prompt for autonomous driving agent"""
    return """
    **A Language Agent for Autonomous Driving**
    Role: You are the brain of an autonomous vehicle (a.k.a. ego-vehicle). 
    Your task is to extract necessary information from the driving scenario that will be useful for driving question answering.

    Necessary information might include:
    - Visual infos: Visual information from specific cameras
    - Detections: Detected objects that require attention
    - Predictions: Estimated future motions of detected objects
    - Maps: Traffic lanes and road boundaries
    - Occupancy: Whether locations are occupied by other objects

    Task:
    - Determine what types of information (Visual info, Detections, Predictions, Maps, Occupancy) to extract
    - Prioritize Detections and Predictions for motion planning questions
    - Focus on Maps information for lane maintenance and road layout questions
    """


def get_user_prompt(raw_question: str, reason_part: str, raw_answer: str) -> str:
    """
    Generate user prompt for tool selection
    
    Args:
        raw_question: The driving question to answer
        reason_part: Reasoning steps from existing CoT data
        raw_answer: Final answer to the question
        
    Returns:
        Formatted user prompt
    """
    input_info = f"The question you need to answer is: {raw_question}\nThe final answer to this question is: {raw_answer}"
    tool_info_intro = generate_func_prompt()
    
    output_format = """
        Please choose tools to answer this problem and support the final answer. 
        Return a list of tool names (no more than 4) with "tool_name" and related "parameters".
        If parameters are blank, use [""] as the value.
        
        Output format example:
        [
            {"tool_name": "function_name1", "parameters": ["arg1", "arg2"]},
            {"tool_name": "function_name2", "parameters": ["arg1", ["option1", "option2"]]}
        ]
        
        STRICTLY FOLLOW THE JSON RESPONSE FORMAT. 
        RESPONSE MUST START WITH "[{" AND END WITH "}]".
        DO NOT START WITH "```json" OR ANY MARKDOWN.
    """
    
    return f"{input_info}\nAvailable tools: {tool_info_intro}\n{output_format}"


def generate_func_prompt(debug: bool = False) -> str:
    """
    Generate prompt listing available functions and their parameters
    
    Args:
        debug: Whether to print debug information
        
    Returns:
        Formatted function prompt
    """
    try:
        func_agent = FuncAgent()
        function_list = (
            func_agent.detection_func_infos + 
            func_agent.map_func_infos + 
            func_agent.prediction_func_infos + 
            func_agent.occupancy_func_infos + 
            func_agent.visual_func_infos
        )
    except Exception:
        func_agent = FuncAgent()
        function_list = (
            func_agent.detection_func_infos + 
            func_agent.map_func_infos + 
            func_agent.prediction_func_infos + 
            func_agent.occupancy_func_infos + 
            func_agent.visual_func_infos
        )
    
    prompt = "Available functions:\n"
    for info in function_list:
        param_str = ", ".join(info["parameters"]["required"]) if info["parameters"].get("required") else ""
        prompt += f"- {info['name']}({param_str}) # {info['description']}\n"
    
    if debug:
        print(prompt)
    
    return prompt


def generate_choose_func_prompt(tool_list: List[Dict]) -> str:
    """
    Generate prompt for selected tools
    
    Args:
        tool_list: List of selected tools
        
    Returns:
        Formatted prompt for selected tools
    """
    func_agent = FuncAgent()
    function_list = (
        func_agent.detection_func_infos + 
        func_agent.map_func_infos + 
        func_agent.prediction_func_infos + 
        func_agent.occupancy_func_infos + 
        func_agent.visual_func_infos
    )
    
    prompt = "Selected tools:\n"
    for tool_sample in tool_list:
        tool_name = tool_sample['tool_name']
        for info in function_list:
            if info['name'] == tool_name:
                param_str = ", ".join(info["parameters"]["required"]) if info["parameters"].get("required") else ""
                prompt += f"- {info['name']}({param_str}) # {info['description']}\n"
    
    return prompt


def get_cot_system_prompt() -> str:
    """Generate system prompt for CoT generation"""
    return """You're an autonomous driving inference optimization expert. 
    Your task is to decompose short CoT data into finer-grained atomic steps and reorganize them for optimal reasoning path. 
    Break down reasoning into minimal units, dynamically cluster atomic steps, extract and label key steps, add tool invocations, 
    and form the optimal reasoning path for the current problem."""


def get_cot_user_prompt(raw_question: str, reason_part: str, final_answer: str, tool_choose_list: List[Dict]) -> str:
    """
    Generate user prompt for CoT generation
    
    Args:
        raw_question: The driving question
        reason_part: Existing reasoning steps
        final_answer: Final answer to the question
        tool_choose_list: List of selected tools
        
    Returns:
        Formatted CoT user prompt
    """
    input_info = (
        f"Raw question: {raw_question}\n"
        f"Existing CoT data: {reason_part}\n"
        f"Final answer: {final_answer}\n\n"
        "Please reconstruct the raw CoT data into atomic CoT data:"
    )
    
    tool_use_prompt = generate_choose_func_prompt(tool_choose_list)
    tool_use_info = (
        f"For each atomic step, choose appropriate tools and parameters.\n"
        f"Available tools: {tool_use_prompt}\n"
    )
    
    output_format = """
        Also generate:
        - Sub-question for each action/reasoning step (perception, prediction, planning)
        - Guess answer for each Sub if you're confident
        - Keywords (2-5 synonyms/alternatives) related to the Guess Answer
        - Missing_flag: "False" if you can't answer, "True" otherwise
        - next_action: "continue reasoning" or "conclude"
        - Continue until reasoning chain is complete
        - Final answer and its keywords
        
        Output format example:
        {
            "Question": "",
            "Chain": [
                {
                    "Tool": {"function_name": "tool1", "parameters": ["param1"]},
                    "Sub": "Sub-question 1",
                    "Guess_Answer": "Answer 1",
                    "key_words": ["word1", "word2"],
                    "Missing_flag": "False",
                    "next_action": "continue reasoning"
                },
                ...
            ],
            "final_answer_keywords": ["keyword1", "keyword2"],
            "final_answer": "Final answer"
        }
        
        STRICTLY FOLLOW THE JSON RESPONSE FORMAT. 
        RESPONSE MUST START WITH "{".
        DO NOT START WITH "```json" OR ANY MARKDOWN.
    """
    
    return input_info + tool_use_info + output_format


def extract_key_steps(text: str) -> str:
    """
    Extract key reasoning steps from text
    
    Args:
        text: Text containing reasoning steps
        
    Returns:
        Cleaned reasoning steps
    """
    stop_markers = [
        "The final answer is:", "**Final Answer:**", "**Final Answer**:",
        "Final Answer", "Answer", "Why take this action?:",
        "**Final Answer**", "**Final Decision**:", "Final Step:", "<CONCLUSION>"
    ]
    
    start_marker = "**Step-by-Step Reasoning**:"
    
    if start_marker in text:
        text_parts = text.split(start_marker)
        relevant_part = text_parts[1].strip()
        
        for marker in stop_markers:
            if marker in relevant_part:
                relevant_part = relevant_part.split(marker)[0].strip()
        
        return relevant_part
    
    return ""


def extract_final_answer(text: str) -> str:
    """
    Extract final answer from text
    
    Args:
        text: Text containing final answer
        
    Returns:
        Extracted final answer
    """
    options = [
        "The final answer is:", "**Final Answer:**", "**Final Answer**:", 
        "Final Answer", "Answer", "Why take this action?:",
        "**Final Answer**", "**Final Decision**:", "Final Step:", "<CONCLUSION>"
    ]
    
    for opt in options:
        if opt in text:
            return text.split(opt)[-1].strip()
    
    return ""


@timeout(100)
def run_one_round_conversation(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str,
        temperature: float = 0.0,
        model_name: str = "gpt-4o-mini"
    ) -> Tuple[List[Dict], str]:
    """
    Perform one round of conversation using OpenAI API
    
    Args:
        full_messages: Conversation history
        system_message: System prompt
        user_message: User prompt
        temperature: Sampling temperature
        model_name: Model to use
        
    Returns:
        Updated conversation history and response message
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [{"type": "text", "text": user_message}]}
    ] if system_message else [{"role": "user", "content": [{"type": "text", "text": user_message}]}]
    
    response = completion_with_backoff(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    
    response_message = response.choices[0].message.content
    full_messages.append(response_message)
    
    return full_messages, response_message


def ask_tool_choice(
    full_messages: List[Dict],
    top_system_prompt: str,
    tool_choose_user_messages: str,
    temperature: float = 0.0,
    model_name: str = "gpt-4o-mini",
    max_retries: int = 10
) -> Tuple[List[Dict], List[Dict], str]:
    """
    Request tool selection from GPT
    
    Args:
        full_messages: Conversation history
        top_system_prompt: System prompt
        tool_choose_user_messages: User prompt for tool selection
        temperature: Sampling temperature
        model_name: Model to use
        max_retries: Maximum retry attempts
        
    Returns:
        tool_choose_list, full_messages, tool_choose_response_message
    """
    for attempt in range(1, max_retries + 1):
        try:
            full_messages, response_message = run_one_round_conversation(
                full_messages=full_messages,
                system_message=top_system_prompt,
                user_message=tool_choose_user_messages,
                temperature=temperature,
                model_name=model_name,
            )
            
            tool_choose_list = json.loads(response_message)
            return tool_choose_list, full_messages, response_message
            
        except json.JSONDecodeError as e:
            logging.warning(
                "Attempt %d/%d: JSON decode failed - %s; Response: %s",
                attempt, max_retries, e, response_message[:200]
            )
    
    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts")


def ask_cot_data(
    full_message: List[Dict],
    cot_system_prompt: str,
    cot_user_prompt: str,
    temperature: float = 0.0,
    model_name: str = "gpt-4o-mini",
    max_retries: int = 10
) -> Tuple[Dict, List[Dict], str]:
    """
    Request CoT data from GPT
    
    Args:
        full_message: Conversation history
        cot_system_prompt: System prompt for CoT generation
        cot_user_prompt: User prompt for CoT generation
        temperature: Sampling temperature
        model_name: Model to use
        max_retries: Maximum retry attempts
        
    Returns:
        cot_data, full_message, cot_response_message
    """
    for attempt in range(1, max_retries + 1):
        try:
            full_message, response_message = run_one_round_conversation(
                full_messages=full_message,
                system_message=cot_system_prompt,
                user_message=cot_user_prompt,
                temperature=temperature,
                model_name=model_name,
            )
            
            cot_data = json.loads(response_message)
            return cot_data, full_message, response_message
            
        except json.JSONDecodeError as e:
            logging.warning(
                "Attempt %d/%d: JSON decode failed - %s; Response: %s",
                attempt, max_retries, e, response_message[:200]
            )
    
    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts")


def gen_pipeline(args: argparse.Namespace) -> None:
    """
    Main pipeline for generating CoT data
    
    Args:
        args: Command line arguments
    """
    json_file = 'data/DriveLMMo1_TRAIN.json'
    all_samples = read_json(json_file)
    progress_file_path = 'progress.json'
    output_file = f'final_cot_{args.split}_{args.model_name}.json'
    
    new_sample = []
    start_index = 0
    
    # Load existing progress if available
    try:
        with open(progress_file_path, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
            start_index = progress_data.get('processed_index', -1) + 1
        
        with open(output_file, 'r', encoding='utf-8') as f:
            new_sample = json.load(f)
    except FileNotFoundError:
        pass
    
    if args.debug:
        all_samples = all_samples[:20]
        logging.info("Running in debug mode with %d samples", len(all_samples))
    
    # Process samples with progress tracking
    for index, sample in enumerate(tqdm(all_samples[start_index:], initial=start_index, desc="Processing samples")):
        current_index = start_index + index
        
        try:
            raw_question = sample['question']
            
            if args.split == 'train':
                raw_answer = sample['answer']
                reason_part = extract_key_steps(raw_answer)
                final_answer = extract_final_answer(raw_answer)
            elif args.split == 'test':
                final_answer = sample['final_answer']
                reason_part = sample['steps']
            
            # Step 1: Tool selection
            top_system_prompt = get_system_prompt()
            tool_choose_user_messages = get_user_prompt(
                raw_question=raw_question,
                reason_part=reason_part,
                raw_answer=final_answer
            )
            
            tool_choose_list, full_message, tool_choose_response_message = ask_tool_choice(
                full_messages=[],
                top_system_prompt=top_system_prompt,
                tool_choose_user_messages=tool_choose_user_messages,
                model_name="gpt-4o-mini"
            )
            
            if args.debug:
                logging.info("Tool selection response: %s", tool_choose_response_message[:200])
            
            # Step 2: CoT generation
            cot_system_prompt = get_cot_system_prompt()
            cot_user_prompt = get_cot_user_prompt(
                raw_question=raw_question,
                reason_part=reason_part,
                final_answer=final_answer,
                tool_choose_list=tool_choose_list
            )
            
            cot_data, full_message, cot_response_message = ask_cot_data(
                full_message=full_message,
                cot_system_prompt=cot_system_prompt,
                cot_user_prompt=cot_user_prompt,
                model_name=args.model_name
            )
            
            if args.debug:
                logging.info("CoT response: %s", cot_response_message[:200])
            
            # Update sample with CoT data
            sample["cot_data"] = cot_data
            new_sample.append(sample)
            
            # Save intermediate results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(new_sample, f, indent=4)
            
            save_progress(progress_file_path, current_index)
            
        except Exception as e:
            logging.error("Error processing sample %d: %s", current_index, str(e))
            if args.debug:
                raise
    
    # Save final results
    with open(f'cot_{args.split}_{args.model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(new_sample, f, indent=4)
    
    logging.info("Process completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CoT data for autonomous driving scenarios")
    parser.add_argument('--split', type=str, default='train', required=True, 
                        choices=['train', 'test'],
                        help="Dataset split: 'train' or 'test'")
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', required=True,
                        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-4.1-mini'],
                        help="OpenAI model to use for CoT generation")
    parser.add_argument('--debug', action="store_true",
                        help="Enable debug mode with detailed logging")
    
    args = parser.parse_args()
    
    # Initialize function agent
    func_agent = FuncAgent()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'gen_{args.split}_data_{args.model_name}.log',
        filemode='w'
    )
    
    if args.debug:
        logging.debug("Arguments: %s", vars(args))
    
    gen_pipeline(args)