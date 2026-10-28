# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Minimal Qwen2.5-VL interface shared by the demo scripts.

The two demo entry points used to duplicate this ~180-line class and, worse,
instantiate the model inside the per-sample loop (reloading weights for every
sample).  The interface is now created once per process.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def pil_to_base64(pil_image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def inference_with_retry(
    inference_func: Callable,
    *args: Any,
    max_retries: int = 3,
    retry_delay: int = 3,
    **kwargs: Any,
) -> str:
    """Call ``inference_func`` with simple retry/backoff semantics."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return inference_func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on GPU state
            last_error = exc
            print(f"Inference error: {exc}. Retry {attempt}/{max_retries}...")
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"Inference failed after {max_retries} retries") from last_error


class Qwen25VLInterface:
    """Thin wrapper around ``Qwen2_5_VLForConditionalGeneration``."""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device or next(self.model.parameters()).device

    def inference(
        self,
        pil_image: Image.Image,
        prompt: str,
        max_tokens: int = 4096,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Generate an answer for one image/prompt pair."""
        image_url = f"data:image;base64,{pil_to_base64(pil_image)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs: Dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature > 0:
            generation_kwargs.update(do_sample=True, temperature=temperature)

        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def infer_with_retry(
        self,
        pil_image: Image.Image,
        prompt: str,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        return inference_with_retry(
            self.inference, pil_image, prompt, max_tokens, max_retries=3, retry_delay=3, **kwargs
        )


# Backwards-compatible alias for the name used in v1.
Qwen2_5VLInterface = Qwen25VLInterface


def execute_tool_call(
    func_agent: Any,
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    verbose: bool = False,
) -> Optional[dict]:
    """Execute ``tool_name`` on ``func_agent`` and return its prompt.

    Returns ``None`` when the tool is unknown or raises, so callers can skip a
    broken call without aborting the whole demo.
    """
    tool_function = getattr(func_agent, tool_name, None)
    if tool_function is None or not callable(tool_function):
        print(f"Error: tool '{tool_name}' is not available")
        return None

    try:
        tool_prompt, _ = tool_function(**tool_args)
    except Exception as exc:
        print(f"Error executing tool '{tool_name}': {exc}")
        return None

    tool_prompt = tool_prompt or ""
    if verbose:
        print(f"Tool: {tool_name}\nArguments: {tool_args}\nPrompt: {tool_prompt}")
    return {"name": tool_name, "args": tool_args, "prompt": tool_prompt}


def build_tool_prompt(
    func_agent: Any,
    tool_chain,
    *,
    verbose: bool = False,
) -> str:
    """Execute a stored tool chain and concatenate the natural-language outputs."""
    prompt = ""
    for node in tool_chain or []:
        if not node:
            continue
        response = execute_tool_call(
            func_agent, node["name"], node.get("args") or {}, verbose=verbose
        )
        if response:
            prompt += response["prompt"]
    return prompt
