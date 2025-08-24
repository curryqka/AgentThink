import argparse
import itertools
import json
import os
import random
import math
import re
import time
from functools import partial

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm

import sys 
# sys.path.append(f"{os.getcwd()}/third_party/ms-swift-main")
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
swift_path = os.path.join(root_dir, "third_party", "ms-swift-main")
if swift_path not in sys.path:
    sys.path.append(swift_path)
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

ds_collections = {
    'DriveLMMo1': {
        # 'root': './data/DriveLMMo1_TEST.jsonl',
        'root': './data/DriveLMMo1_TEST_tool_results.jsonl',
        # 'root': './DriveLMM-o1-main/data/DriveLMMo1_TEST_tool_results.jsonl',
        'max_new_tokens': 2000,
        'min_new_tokens': 1,
        'split': 'validation',
        'image_root': './data/image2concat'
    }
}

def collate_fn(batches, tokenizer):
    # pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    images = [_['images'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    reasons = [_['reason'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    return images, questions, answers, reasons, data_ids


class DriveLMMo1Dataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt, image_path, point_path=None, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, tool_result_json:str=None):
        self.data_path = root
        with open(root, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
            # data_val = json.load(f)
        # merge all dataset
        # self.data = concatenate_datasets(sub_dataset_list)
        
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.image_path = image_path
        self.point_path = point_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx]
        data_id = data['id']
        question = data['conversations'][0]['value'].strip()
        
        image_file = os.path.join(self.image_path, data['image'])
        image = Image.open(image_file).convert('RGB')
        answer = data['conversations'][1]['value'].strip()
        reason_gt = data['conversations'][2]['value'].strip()
        if 'tool_results' in self.data_path:
            tool_result = data['tool_result']
            system_prompt = data['system_prompts']
            reason = f"{system_prompt}\nTo answer the question, please refer to the tool recomendation results which show in the following dict: (Note: the numerical results are all based on the ego-car coordination axis.)\n{tool_result}"
        if self.dynamic_image_size:
            pil_image = dynamic_preprocess(image, image_size=self.input_size,
                                           use_thumbnail=self.use_thumbnail,
                                           max_num=self.max_num)
            images = pil_image
        else:
            images = [image]
        
        return {
            'question': self.prompt+'\n<image>\n'+question,
            'images': image_file,
            'answer': answer,
            'reason': reason,
            'data_id': data_id
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def load_model(pretrained_model):
    """Load model and tokenizer"""
    model = pretrained_model
    template_type = None  # None: default template_type
    default_system = None  # None: default_system

    # Load models and conversation
    model, tokenizer = get_model_tokenizer(model)
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    return engine, model, tokenizer



def retry_torch_distributed_barrier(max_retries=3, delay_seconds=5):
    """
    Attempts to execute torch.distributed.barrier() with a retry mechanism
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay_seconds (int): Delay in seconds between retry attempts
    """
    retries = 0
    while retries < max_retries:
        try:
            torch.distributed.barrier()
            # Exit the function upon successful execution
            return
        except Exception as e:
            retries += 1
            print(f"torch.distributed.barrier() failed (retry {retries}/{max_retries}): {str(e)}")
            print(f"Retrying after {delay_seconds} seconds...")
            time.sleep(delay_seconds)
    
    # Raise exception if barrier still fails after max retries
    raise RuntimeError(f"torch.distributed.barrier() failed after {max_retries} retries")

def evaluate_chat_model():
    random.seed(args.seed)
    prompt = "When answering the question based on the provided image, follow a structured and logical reasoning process. Organize your response using the format, ensuring each step builds upon the previous one and clearly explains how the image(s) contribute to the solution. Your answer should be structured as Reasoning Steps: (step by step reasoning) Final Answer: (final answer) \n Question: "
    
    for ds_name in args.datasets:
        dataset = DriveLMMo1Dataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
            prompt=prompt,
            image_path=ds_collections[ds_name]['image_root'],
            # image_meta = ds_collections[ds_name]["image_meta"],
            # input_size=image_size,
            dynamic_image_size=args.dynamic,
            # use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (images, questions, answers, reasons, data_ids) in tqdm(enumerate(dataloader)):
            # pixel_values = pixel_values.to(torch.bfloat16).cuda()
                        
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
           
            reason_prompt = reasons[0]
            infer_requests = [
                InferRequest(messages=[
                    {'role': 'system', 'content': "You are the helpful assistant!"},
                    {'role': 'user', 'content': f"<image>{questions[0]}\n{reason_prompt}"}
                ],
                images=images),
            ]
           
            resp_list = engine.infer(infer_requests, RequestConfig(max_tokens=12000, temperature=args.temperature))
            pred = resp_list[0].choices[0].message.content
           
            preds = [pred]

            for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'id': data_id
                })
        
        # torch.distributed.barrier()
        retry_torch_distributed_barrier(max_retries=15, delay_seconds=5)

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            # time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            # time_prefix = "qwen"
            
            results_file = f'{ds_name}_{args.output_name}.json'
            output_path = os.path.join(args.out_dir, results_file)

            # breakpoint()
            with open(output_path, 'w') as f:
                json.dump(merged_outputs, f, indent=4)
            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='DriveLMMo1')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--output_name', type=str, default='qwen_32B_swift')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', default=False)
    parser.add_argument('--max-num', type=int, default=12)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # model, tokenizer = load_model_and_tokenizer()
    # engine, model, tokenizer = load_model("qwen_vla/Qwen2.5-VL-32B-Instruct")
    engine, model, tokenizer = load_model(args.checkpoint)
  

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
  
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()

