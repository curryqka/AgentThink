#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""LLM-as-Judge evaluation on DriveLMM-o1 (paper Sec. 3.3, Appendix C).

Scores every model response on the twelve DriveLMM-o1 reasoning metrics and
reports the seven columns of Table 3 in the paper (Risk Assess., Rule Adh.,
Scene Aware., Relevance, Missing, Reason., MCQ).

Example:
    export OPENAI_API_KEY=...
    python evaluation/evaluation_script.py \
        --input results/DriveLMMo1_AgentThink.json \
        --dataset data/DriveLMMo1/DriveLMMo1_TEST.json \
        --output results/agentthink_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.prompts import (  # noqa: E402
    JUDGE_METRIC_KEYS,
    JUDGE_RESPONSE_FORMAT,
    JUDGE_SYSTEM_PROMPT,
    PAPER_METRIC_COLUMNS,
)
from scripts.tools.tool_chain import extract_json_object  # noqa: E402

#: Question-type suffixes (last ``_``-separated field of ``idx``) that are
#: multiple-choice questions in DriveLMM-o1.
DEFAULT_MCQ_TYPES = (2, 3, 4, 5, 8)

ANSWER_MARKERS = (
    "The final answer is:",
    "**Final Answer:**",
    "**Final Answer**:",
    "Final Answer",
    "Answer",
    "Why take this action?:",
    "**Final Answer**",
    "**Final Decision**:",
    "Final Step:",
    "<CONCLUSION>",
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """Return the substring after the first final-answer marker."""
    if not text:
        return ""
    for marker in ANSWER_MARKERS:
        if marker in text:
            return text.split(marker)[-1].strip()
    return ""


def extract_options(text: str) -> List[tuple[str, str]]:
    """Parse ``A) ...`` style options. Returns ``[("none", "none")]`` if absent."""
    matches = re.findall(r"([A-F])\)\s*(.+)", text or "")
    if matches:
        return [(label, option) for label, option in matches]
    if "none of the " in (text or "").lower():
        return [("F", "None of the option.")]
    return [("none", "none")]


def question_type(idx: str) -> Optional[int]:
    try:
        return int(str(idx).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def parse_judge_response(content: str) -> Optional[Dict[str, float]]:
    """Parse the judge JSON and recompute ``Overall Score`` as the mean."""
    if not content:
        return None
    candidate = extract_json_object(content) or content
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores = {}
    for key in JUDGE_METRIC_KEYS:
        value = data.get(key)
        if isinstance(value, (int, float)):
            scores[key] = float(value)
    if len(scores) != len(JUDGE_METRIC_KEYS):
        return None
    scores["Overall Score"] = mean(scores.values())
    return scores


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------

def build_client(base_url: Optional[str] = None):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running the judge, e.g.\n"
            "  export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))


def judge_reasoning(
    client,
    model: str,
    question: str,
    ground_truth: str,
    response: str,
) -> Optional[Dict[str, float]]:
    """Score one response; retries twice on transient or malformed output."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"LLM Response: {response}"
            ),
        },
    ]
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=JUDGE_RESPONSE_FORMAT,
                max_tokens=500,
                temperature=0.0,
            )
            scores = parse_judge_response(completion.choices[0].message.content)
            if scores is not None:
                return scores
        except Exception as exc:  # network / rate limit / schema errors
            if attempt == 2:
                print(f"Judge failed after 3 attempts: {exc}")
    return None


# ---------------------------------------------------------------------------
# Dataset / sample handling
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return {str(item["idx"]): item for item in records}


def load_predictions(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_sample(
    item: dict,
    ground_truth: Dict[str, dict],
    client,
    args: argparse.Namespace,
) -> Optional[dict]:
    idx = item.get("id") or item.get("idx") or ""
    idx = str(idx)
    if not idx or idx not in ground_truth:
        return None

    reference = ground_truth[idx]
    question = item.get("question") or reference.get("question", "")
    response = item.get("llm-response") or item.get("answer", "")
    gt_steps = reference.get("steps", "")
    gt_final = reference.get("final_answer", "")
    ground_truth_text = (
        f"**Step-by-Step Reasoning**: {gt_steps} **Final Answer**: {gt_final}"
    )

    scores = judge_reasoning(
        client, args.judge_model, question, ground_truth_text, response
    )
    if scores is None:
        return None

    # MCQ accuracy, following the DriveLMM-o1 protocol.
    mcq = -1
    if question_type(idx) in args.mcq_types:
        gt_label = extract_options(gt_final)[0][0]
        if gt_label != "none":
            pred_label = extract_options(extract_final_answer(response))[0][0]
            mcq = int(pred_label == gt_label)

    record = {
        "id": idx,
        "question": question,
        "ground_truth": ground_truth_text,
        "llm_response": response,
        "mcq": mcq,
    }
    record.update(scores)

    if args.tool_metrics:
        from evaluation.tool_metrics import judge_tool_usage

        tool_scores = judge_tool_usage(
            client,
            args.judge_model,
            question=question,
            reference=ground_truth_text,
            response=response,
        )
        if tool_scores:
            record["tool_metrics"] = tool_scores
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(records: List[dict]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_samples": len(records)}
    for column, key in PAPER_METRIC_COLUMNS.items():
        values = [r[key] for r in records if key in r]
        summary[column] = round(mean(values), 2) if values else None

    answered = [r["mcq"] for r in records if r.get("mcq", -1) != -1]
    summary["MCQ"] = round(100.0 * mean(answered), 2) if answered else None
    summary["num_mcq"] = len(answered)

    tool_records = [r["tool_metrics"] for r in records if r.get("tool_metrics")]
    if tool_records:
        summary["Tool Usage Appro."] = round(
            mean(r["Tool Usage Appropriateness"] for r in tool_records), 2
        )
        summary["Tool Chain Coh."] = round(
            mean(r["Tool Chain Coherence"] for r in tool_records), 2
        )
        summary["Percep-Guided Tool Align."] = round(
            mean(r["Perception-Guided Tool Alignment"] for r in tool_records), 2
        )
        summary["Overall Tool Score"] = round(
            mean(r["Overall Tool Score"] for r in tool_records), 2
        )
    return summary


def write_outputs(records: List[dict], summary: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.parent / f"{output.name}.json"
    summary_path = output.parent / f"{output.name}.summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=4, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4, ensure_ascii=False)

    # Flatten the tool metrics for the CSV export.
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "tool_metrics"}
        row.update(record.get("tool_metrics") or {})
        rows.append(row)
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        csv_path = output.parent / f"{output.name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\nEvaluated {summary['num_samples']} samples "
          f"({summary['num_mcq']} MCQ).")
    header = ["Metric"] + list(PAPER_METRIC_COLUMNS) + ["MCQ"]
    print("  ".join(f"{name:>12}" for name in header))
    values = [summary.get(column, "n/a") for column in PAPER_METRIC_COLUMNS]
    print("  ".join(f"{str(value):>12}" for value in values + [summary.get("MCQ", "n/a")]))
    if "Overall Tool Score" in summary:
        print(
            "Tool metrics: "
            f"usage={summary['Tool Usage Appro.']}  "
            f"chain={summary['Tool Chain Coh.']}  "
            f"alignment={summary['Percep-Guided Tool Align.']}  "
            f"overall={summary['Overall Tool Score']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveLMM-o1 LLM-as-Judge evaluation")
    parser.add_argument("--input", required=True, help="Prediction JSON produced by evaluation/inference*.py.")
    parser.add_argument("--dataset", required=True, help="Ground-truth DriveLMMo1_TEST.json.")
    parser.add_argument("--output", default="results/evaluation", help="Output prefix (no extension).")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N predictions.")
    parser.add_argument(
        "--mcq-types",
        type=lambda value: tuple(int(v) for v in value.split(",")),
        default=DEFAULT_MCQ_TYPES,
        help="Question-type suffixes treated as MCQ (default: 2,3,4,5,8).",
    )
    parser.add_argument("--tool-metrics", action="store_true",
                        help="Also compute the three tool-use metrics of Table 2.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible endpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    input_path = Path(args.input)
    if not dataset_path.exists():
        raise SystemExit(f"Ground-truth file not found: {dataset_path}")
    if not input_path.exists():
        raise SystemExit(f"Prediction file not found: {input_path}")

    ground_truth = load_ground_truth(dataset_path)
    predictions = load_predictions(input_path)
    if args.limit:
        predictions = predictions[: args.limit]

    valid = [
        item
        for item in predictions
        if str(item.get("id") or item.get("idx") or "") in ground_truth
    ]
    skipped = len(predictions) - len(valid)
    if skipped:
        print(f"Skipping {skipped} predictions without a ground-truth match.")
    if not valid:
        raise SystemExit("No prediction ids match the ground-truth file.")

    client = build_client(args.base_url)
    records: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(score_sample, item, ground_truth, client, args): item
            for item in valid
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            record = future.result()
            if record is not None:
                records.append(record)

    if not records:
        raise SystemExit("All judge calls failed; check the API key and model name.")

    records.sort(key=lambda record: record["id"])
    summary = summarize(records)
    write_outputs(records, summary, Path(args.output))
    print_summary(summary)
    print(f"\nResults written to {args.output}.json / .csv / .summary.json")


if __name__ == "__main__":
    main()
