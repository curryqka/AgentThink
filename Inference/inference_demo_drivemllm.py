# Copyright (c) Kangan Qian. All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: Interface for Qwen2.5-VL model inference with tool integration

import json
import time
import base64
import io
import sys
from typing import Callable, Any
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from scripts.tools.tool_libraries_simple import FuncAgent
from qwen_vl_utils import process_vision_info


def pil_to_base64(pil_image: Image.Image) -> str:
    """
    Convert a PIL image object to a base64-encoded string.
    
    Args:
        pil_image (Image.Image): PIL image object to convert
        
    Returns:
        str: Base64-encoded string representation of the image
    """
    try:
        binary_stream = io.BytesIO()
        pil_image.save(binary_stream, format="PNG")
        binary_data = binary_stream.getvalue()
        return base64.b64encode(binary_data).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Image to base64 conversion failed: {e}")


def inference_with_retry(
        inference_func: Callable,
        *args: Any,
        max_retries: int = 3,
        retry_delay: int = 3,
        **kwargs: Any
) -> str:
    """
    Execute an inference function with automatic retries on failure.
    
    Args:
        inference_func (Callable): Inference function to call
        *args: Positional arguments for the inference function
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retry attempts in seconds
        **kwargs: Keyword arguments for the inference function
        
    Returns:
        str: Output from the inference function
        
    Raises:
        RuntimeError: If maximum retries are exceeded without success
    """
    retries = 0
    while retries < max_retries:
        try:
            return inference_func(*args, **kwargs)
        except Exception as e:
            print(f"Inference error: {e}. Retry {retries+1}/{max_retries}...")
            retries += 1
            time.sleep(retry_delay)
    
    raise RuntimeError(f"Inference failed after {max_retries} retries")


class Qwen2_5VLInterface:
    def __init__(self, model_path: str) -> None:
        """
        Initialize Qwen2.5-VL model interface
        
        Args:
            model_path (str): Path to pretrained model
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def inference(self, pil_image: Image.Image, prompt: str, max_tokens: int = 4096) -> str:
        """
        Perform inference using the Qwen2.5-VL model
        
        Args:
            pil_image (Image.Image): Input image
            prompt (str): Text prompt for the model
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Model output text
        """
        # Convert image to base64 for model input
        image_base64 = pil_to_base64(pil_image)
        image_url = f"data:image;base64,{image_base64}"
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate model output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]


def execute_tool_call(
        func_agent: FuncAgent, 
        tool_name: str, 
        tool_args: dict, 
        verbose: bool = True
    ) -> dict:
    """
    Execute a tool call using the function agent
    
    Args:
        func_agent (FuncAgent): Function agent instance
        tool_name (str): Name of the tool to execute
        tool_args (dict): Arguments for the tool
        verbose (bool): Whether to print tool execution details
        
    Returns:
        dict: Tool response containing name, arguments, and prompt
    """
    try:
        tool_function = getattr(func_agent, tool_name)
    except AttributeError:
        print(f"Error: Tool '{tool_name}' not found")
        return None
    
    if not callable(tool_function):
        print(f"Error: '{tool_name}' is not a callable function")
        return None
    
    try:
        tool_prompt, tool_result_data = tool_function(**tool_args)
    except Exception as e:
        print(f"Error executing tool '{tool_name}': {e}")
        return None
    
    if tool_prompt is None:
        tool_prompt = ""
    
    tool_response = {
        "name": tool_name,
        "args": tool_args,
        "prompt": tool_prompt,
    }
    
    if verbose:
        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_args}")
        print(f"Prompt: {tool_prompt}")
    
    return tool_response


def run_chat_model_inference(
        image_path: str, 
        prompt: str, 
        model_path: str = "/path/to/model/checkpoint"
    ) -> str:
    """
    Run inference using the chat model
    
    Args:
        image_path (str): Path to input image file
        prompt (str): Text prompt for the model
        model_path (str): Path to model checkpoint
        
    Returns:
        str: Model output text
    """
    image = Image.open(image_path)
    model_interface = Qwen2_5VLInterface(model_path)
    return inference_with_retry(
        model_interface.inference, 
        image, 
        prompt, 
        max_retries=3, 
        retry_delay=3
    )


def main():
    """Main function to process JSON data and run model inference"""
    # Initialize function agent
    func_agent = FuncAgent()
    
    # Load JSON data
    json_file = "./Inference/inference_demo_data_drivemllm.json"
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    
    # Process each sample in the JSON data
    for sample in json_data:
        image_path = sample['image'][0]
        tool_chain = sample['tool_result']
        system_prompt = sample['system_prompts']
        question = sample['question']
        
        # Build tool prompt from tool chain
        tool_prompt = ""
        for tool_node in tool_chain:
            tool_name = tool_node['name']
            tool_args = tool_node['args']
            tool_response = execute_tool_call(func_agent, tool_name, tool_args)
            
            if tool_response:
                tool_prompt += tool_response['prompt']
        
        # Construct full prompt for model inference
        full_prompt = f"{system_prompt}\n{question}\nTool results:{tool_prompt}"
        
        # Run model inference
        model_output = run_chat_model_inference(image_path, full_prompt, model_path="pretrained_model/AgentThink")
        print("Model output:", model_output)


if __name__ == "__main__":
    main()