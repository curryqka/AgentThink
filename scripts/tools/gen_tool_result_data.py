# Copyright (c) OpenMMLab. All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: AgentThink class for processing driving scenario data with tool calls

import pickle
import json
from pathlib import Path
from tqdm import tqdm
from scripts.tools.tool_libraries import FuncAgent
from scripts.tools.tool_prompts import get_system_prompt, get_ego_prompts, get_detection_prompt


class AgentThink:
    def __init__(self, token: str = None, split: str = 'train', 
                 data_path: str = 'DriveLMM-o1-main/data/tool_results',
                 drivelmm_json_file: str = 'Drive-MLLM-main/data/DriveLMMo1/DriveLMMo1_TEST.json', 
                 model_name: str = "qwen2.5-VL", verbose: bool = False) -> None:
        """
        Initialize AgentThink class for processing driving scenario data.
        
        Args:
            token (str): Token identifier for the data
            split (str): Data split type ('train' or 'val')
            data_path (str): Path to tool results data
            drivelmm_json_file (str): Path to DriveLMM JSON file
            model_name (str): Name of the model being used
            verbose (bool): Whether to show detailed logs
        """
        self.token = token
        self.split = split
        self.data_path = data_path
        self.model_name = model_name
        self.verbose = verbose
        
        # Load data from pickle file
        folder_name = Path("val") if "val" in split else Path("train")
        self.file_name = Path(data_path) / folder_name / Path(f"{self.token}.pkl")
        with open(self.file_name, "rb") as f:
            self.data_dict = pickle.load(f)
        
        # Initialize function agent
        self.func_agent = FuncAgent(self.data_dict)
        
        # Set limits for tool calls
        self.num_call_detection_times = 3
        self.num_call_prediction_times = 1
        self.num_call_occupancy_times = 1
        self.num_call_map_times = 1

    def _preprocess_tool_results(self, json_data):
        """
        Convert agent-driver data into the scene in drivelmm-o1 format.
        
        Args:
            json_data: JSON data to preprocess
            
        Returns:
            Preprocessed JSON data with tool results
        """
        # TODO: Implement matching and alignment between JSON data and tool results
        new_json_data = []
        for sample in json_data:
            sample_idx = sample['idx']
            scene_token = sample_idx.split('_')[0]
            frame_token = sample_idx.split('_')[1]
            
            # Load corresponding tool data
            folder_name = Path("val") if "val" in self.split else Path("train")
            file_name = Path(self.data_path) / folder_name / Path(f"{frame_token}.pkl")
            with open(file_name, "rb") as f:
                data_dict = pickle.load(f)
            
            # Add tool results to sample
            sample['tool_results'] = data_dict
            new_json_data.append(sample)
        
        # Save processed data
        output_file = f'cot_{self.split}_{self.model_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_json_data, f, indent=4)
        
        return new_json_data

    def tool_call(self, response_message):
        """
        Execute a tool call based on the response message.
        
        Args:
            response_message: Message containing tool call information
            
        Returns:
            Tool response dictionary or None if call fails
        """
        try:
            tool_name = response_message['Tool']['function_name']
        except (KeyError, TypeError):
            return None
            
        if tool_name == '' or tool_name == 'none':
            return None
            
        function_args = response_message['Tool']['parameters']
        if len(function_args) > 0:
            if function_args[0] == '':
                function_args = {}
        else:
            return None
            
        # Process function arguments based on tool type
        if isinstance(function_args, list):
            if 'occupancy' in tool_name:
                locations = function_args[0]
                timestep = function_args[1]
                if isinstance(locations, list):
                    locations = tuple(locations)
                function_args = {'locations': [locations], 'timestep': timestep}
            elif 'location' in tool_name:
                locations = function_args[0]
                if isinstance(locations, list):
                    locations = tuple(locations)
                function_args = {'locations': [locations]}
            else:
                if 'open' not in tool_name:
                    obj_list = function_args[0]
                    function_args = {'object_ids': obj_list}
                else:
                    tool_name = 'get_open_world_vocabulary_detection'
                    obj_list = function_args[0]
                    function_args = {'object_names': obj_list}
        
        # Get the function to call
        try:
            function_to_call = getattr(self.func_agent, tool_name)
        except AttributeError:
            return None
        
        # Execute the function call
        if not callable(function_to_call):
            print(f"Function {tool_name} is not callable!")
            return None
        else:
            try:
                tool_returns = function_to_call(**function_args)
            except Exception:
                return None
                
            tool_prompt, tool_result_data = tool_returns
            
            if tool_prompt is None:
                tool_prompt = ""
                
            # Create tool response dictionary
            tool_response = {
                "name": tool_name,
                "args": function_args,
                "prompt": tool_prompt,
            }
            
            if self.verbose:
                print(f"Tool: {tool_name}")
                print(f"Args: {function_args}")
                print(f"Prompt: {tool_prompt}")
                
            return tool_response

    def get_tool_results(self, sample, ego_prompts=None):
        """
        Collect information from driving scenarios using chain-of-thought reasoning with function calls.
        
        Args:
            sample: Data sample to process
            ego_prompts: Optional ego prompts to include
            
        Returns:
            Tuple of (full_messages, system_message, tool_responses)
        """
        # Initialize system message
        init_system_message = get_system_prompt()
        full_messages = []
        tool_responses = []
        
        # Combine system message with ego prompts
        system_message = init_system_message + "\n" + ego_prompts + "\n"
        
        if self.verbose:
            print("System Message:", system_message)
            print("Detection Prompt:", get_detection_prompt())
        
        # Process chain of thought data
        cot_data = sample['cot_data']
        tool_chain = cot_data['Chain']
        
        # Execute tool calls iteratively
        cur_num_det_tool_call = 0
        for chain_node in tool_chain:
            try:
                tool_name = chain_node['Tool']['function_name']
            except (KeyError, TypeError):
                continue
                
            # Limit detection tool calls
            if 'detection' in tool_name:
                cur_num_det_tool_call += 1
                if cur_num_det_tool_call > self.num_call_detection_times:
                    continue
                    
            # Execute tool call
            tool_response = self.tool_call(chain_node)
            
            # Add tool response to messages
            if tool_response is not None:
                full_messages.append({
                    'role': 'function',
                    'name': tool_response['name'],
                    'content': tool_response['prompt'],
                })
                
            tool_responses.append(tool_response)
            
        return full_messages, system_message, tool_responses


def main(drivelmm_json_file="/path/to/final_cot_test_gpt-4.1-mini.json"):
    """
    Main function to process DriveLMM JSON data with AgentThink.
    
    Args:
        drivelmm_json_file: Path to DriveLMM JSON file
        
    Returns:
        Processed JSON data
    """
    # Load JSON data
    with open(drivelmm_json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    
    # Process each sample in the JSON data
    new_json_data = []
    for index, sample in enumerate(tqdm(json_data, desc="Processing JSON samples")):
        sample_idx = sample['idx']
        scene_token = sample_idx.split('_')[0]
        frame_token = sample_idx.split('_')[1]
        
        # Initialize agent and get ego prompts
        agent = AgentThink(
            token=frame_token, 
            split='val', 
            data_path="/path/to/tool_results",
            drivelmm_json_file=drivelmm_json_file,
            model_name='Qwen2.5-VL'
        )
        
        cur_data_dict = agent.data_dict
        ego_prompts = get_ego_prompts(cur_data_dict)
        
        # Get tool results
        full_messages, system_prompts, tool_responses = agent.get_tool_results(
            sample=sample, 
            ego_prompts=ego_prompts
        )
        
        # Update sample with tool results
        sample['tool_result'] = tool_responses
        sample['system_prompts'] = system_prompts
        new_json_data.append(sample)
    
    # Save processed data
    output_file = f'{agent.data_path}/cot_{agent.split}_{agent.model_name}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, indent=4)
        
    return new_json_data


if __name__ == "__main__":
    # Example usage
    main()