# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Shared inference utilities for the AgentThink evaluation scripts.

The three evaluation entry points (``inference.py``, ``inference_withtool.py``
and ``inference_agentthink.py``) used to duplicate ~300 lines of dataset,
distributed and generation code.  This module keeps a single implementation so
that the three inference modes differ only in their prompt and input format.

The scripts can be launched either with ``torchrun`` (multi-GPU / multi-node)
or with plain ``python`` (single GPU); distributed mode is detected from the
environment variables set by ``torchrun``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_TEST_FILE = DEFAULT_DATA_ROOT / "DriveLMMo1_TEST.jsonl"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT / "image2concat"
DEFAULT_TOOL_RESULT_TEST_FILE = DEFAULT_DATA_ROOT / "DriveLMMo1_TEST_tool_results.jsonl"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _format_tool_result(tool_result: Any, style: str = "prompts") -> str:
    """Render stored tool outputs for the model prompt.

    Args:
        tool_result: ``tool_result`` field of a JSONL record.  It is normally a
            list of ``{"name", "args", "prompt"}`` dictionaries produced by
            :mod:`scripts.tools.gen_tool_result_data`.
        style: ``"prompts"`` concatenates the natural-language ``prompt`` of each
            call (recommended).  ``"repr"`` reproduces the string representation
            used by the original v1 scripts.
    """
    if tool_result is None:
        return ""
    if style == "repr" or not isinstance(tool_result, (list, tuple)):
        return str(tool_result)

    prompts = []
    for item in tool_result:
        if isinstance(item, dict):
            prompt = item.get("prompt") or ""
            if prompt.strip():
                prompts.append(prompt.strip())
        elif item:
            prompts.append(str(item).strip())
    return "\n".join(prompts)


class DriveLMMo1Dataset(torch.utils.data.Dataset):
    """DriveLMM-o1 JSONL dataset used for evaluation.

    Each record must contain ``id``, ``image``, ``conversations`` (question,
    final answer, reference steps) and, for tool-augmented inference,
    ``tool_result`` and ``system_prompts``.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        prompt: str,
        image_root: str | Path,
        *,
        with_tool_results: bool = False,
        tool_result_style: str = "prompts",
        limit: Optional[int] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.prompt = prompt
        self.with_tool_results = with_tool_results
        self.tool_result_style = tool_result_style

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Test file not found: {self.jsonl_path}. "
                "See docs/pipeline.md for how to build it from the raw data."
            )

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            self.data = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            self.data = self.data[:limit]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        data = self.data[idx]
        data_id = data["id"]
        question = data["conversations"][0]["value"].strip()
        answer = data["conversations"][1]["value"].strip()

        image_file = self.image_root / data["image"]
        if not image_file.exists():
            raise FileNotFoundError(
                f"Image not found: {image_file}. "
                "Check --image-root and the `image` field of the test JSONL."
            )

        reason = ""
        if self.with_tool_results:
            system_prompt = (data.get("system_prompts") or "").strip()
            tool_result = _format_tool_result(
                data.get("tool_result"), style=self.tool_result_style
            )
            reason = (
                f"{system_prompt}\n"
                "To answer the question, please refer to the tool recommendation "
                "results which show in the following dict: (Note: the numerical "
                "results are all based on the ego-car coordination axis.)\n"
                f"{tool_result}"
            )

        return {
            # The single ``<image>`` placeholder is added by ``generate()``;
            # repeating it here would send two placeholders for one image.
            "question": f"{self.prompt}\n{question}",
            "images": str(image_file),
            "answer": answer,
            "reason": reason,
            "data_id": data_id,
        }


def collate_fn(batches: Sequence[dict], tokenizer: Any = None):
    """Batch size 1 collation (AgentThink generation only supports 1)."""
    return (
        [item["images"] for item in batches],
        [item["question"] for item in batches],
        [item["answer"] for item in batches],
        [item["reason"] for item in batches],
        [item["data_id"] for item in batches],
    )


class InferenceSampler(torch.utils.data.Sampler):
    """Deterministic contiguous shard of the dataset for ``rank``."""

    def __init__(self, size: int) -> None:
        self._size = int(size)
        if self._size <= 0:
            raise ValueError("InferenceSampler requires a non-empty dataset")
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(
            size, self._world_size, self._rank
        )

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int) -> range:
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self) -> int:
        return len(self._local_indices)


# ---------------------------------------------------------------------------
# Distributed / model helpers
# ---------------------------------------------------------------------------

def init_distributed() -> None:
    """Initialise ``torch.distributed`` for ``torchrun`` *or* plain ``python``."""
    if torch.distributed.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
    else:
        # Single-process fallback so `python evaluation/inference.py ...` works
        # without torchrun.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:29500",
                world_size=1,
                rank=0,
            )
        except (RuntimeError, ValueError):
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:29500",
                world_size=1,
                rank=0,
            )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))


def retry_torch_distributed_barrier(max_retries: int = 15, delay_seconds: int = 5) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            torch.distributed.barrier()
            return
        except Exception as exc:  # pragma: no cover - depends on cluster state
            print(
                f"torch.distributed.barrier() failed "
                f"(retry {attempt}/{max_retries}): {exc}"
            )
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"torch.distributed.barrier() failed after {max_retries} retries"
    )


def load_engine(checkpoint: str, max_batch_size: int = 1):
    """Load a checkpoint with ms-swift and wrap it in a ``PtEngine``.

    ms-swift is imported lazily so that this module can be imported (and unit
    tested) on machines without the full inference stack.
    """
    swift_path = REPO_ROOT / "third_party" / "ms-swift-main"
    if swift_path.exists() and str(swift_path) not in sys.path:
        sys.path.append(str(swift_path))

    from swift.llm import PtEngine, get_model_tokenizer, get_template

    model, tokenizer = get_model_tokenizer(checkpoint)
    template = get_template(
        model.model_meta.template, tokenizer, default_system=None
    )
    engine = PtEngine.from_model_template(
        model, template, max_batch_size=max_batch_size
    )
    return engine, model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    engine: Any,
    images: Sequence[str],
    questions: Sequence[str],
    reasons: Sequence[str],
    *,
    max_new_tokens: int = 2000,
    temperature: float = 0.0,
    system_message: Optional[str] = None,
) -> List[str]:
    """Run one batch through ms-swift and return the decoded responses."""
    from swift.llm import InferRequest, RequestConfig

    preds = []
    for image, question, reason in zip(images, questions, reasons):
        content = f"<image>{question}"
        if reason:
            content = f"{content}\n{reason}"
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        request = InferRequest(messages=messages, images=[image] if isinstance(image, str) else image)
        response = engine.infer(
            [request], RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        )
        preds.append(response[0].choices[0].message.content)
    return preds


def evaluate_model(
    engine: Any,
    dataset: DriveLMMo1Dataset,
    args: argparse.Namespace,
    *,
    system_message: Optional[str] = None,
    postprocess: Optional[Callable[[str], dict]] = None,
) -> List[dict]:
    """Run generation over a dataset and return the output records."""
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=None),
    )

    outputs: List[dict] = []
    for images, questions, answers, reasons, data_ids in tqdm(
        dataloader, desc="Generating", disable=torch.distributed.get_rank() != 0
    ):
        preds = generate(
            engine,
            images,
            questions,
            reasons,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            system_message=system_message,
        )
        for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
            record = {
                "question": question,
                "answer": pred,
                "gt_answers": answer,
                "id": data_id,
            }
            if postprocess is not None:
                record.update(postprocess(pred))
            outputs.append(record)
    return outputs


def gather_and_save(outputs: List[dict], args: argparse.Namespace, dataset_name: str) -> None:
    """Merge shards from all ranks and write ``<dataset>_<output_name>.json``."""
    retry_torch_distributed_barrier()

    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        merged: List[Optional[str]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged, json.dumps(outputs))
        outputs = list(
            itertools.chain.from_iterable(json.loads(chunk) for chunk in merged)
        )

    if torch.distributed.get_rank() != 0:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{dataset_name}_{args.output_name}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=4, ensure_ascii=False)
    print(f"Results saved to {result_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out_dir: str = "results",
    default_test_file: Path = DEFAULT_TEST_FILE,
) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", type=str, required=True, help="Merged model checkpoint or HF id.")
    parser.add_argument("--test-file", type=str, default=str(default_test_file), help="Evaluation JSONL file.")
    parser.add_argument("--image-root", type=str, default=str(DEFAULT_IMAGE_ROOT), help="Directory of stitched six-view images.")
    parser.add_argument("--out-dir", type=str, default=default_out_dir)
    parser.add_argument("--output-name", type=str, required=True, help="Suffix of the result JSON file.")
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples (debugging).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tool-result-style", choices=["prompts", "repr"], default="prompts",
                        help="How to render stored tool outputs for AgentThink inference.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("Only batch size 1 is supported by the upstream generation loop.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
