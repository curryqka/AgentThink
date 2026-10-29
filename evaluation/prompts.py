# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Prompt templates shared by data generation, inference and evaluation.

Keeping every model-visible instruction in one place makes it easy to verify
that the released pipeline matches the paper (Sec. 3.1 and Sec. 3.3).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Inference prompts
# ---------------------------------------------------------------------------

#: Baseline prompt used by ``evaluation/inference.py`` (plain CoT, no tools).
PLAIN_COT_PROMPT = (
    "When answering the question based on the provided image, follow a structured "
    "and logical reasoning process. Organize your response using the format, ensuring "
    "each step builds upon the previous one and clearly explains how the image(s) "
    "contribute to the solution. Your answer should be structured as "
    "Reasoning Steps: (step by step reasoning) Final Answer: (final answer)\n"
    "Question: "
)

#: Prompt used by ``evaluation/inference_withtool.py``. The model proposes a
#: structured tool-augmented reasoning chain (Tool / Sub / Guess_Answer / ...),
#: exactly the schema introduced in Sec. 3.1 of the paper.
TOOL_COT_PROMPT = (
    "When addressing the question presented in the image, please adhere to a "
    "structured and logical reasoning process. Ensure that your chain of reasoning "
    "consists of no more than six steps to maintain clarity and conciseness.\n"
    "For each atomic step of the reasoning chain, choose the appropriate tool and "
    'its parameters (if blank, the value of "parameters" should be [""]). '
    "The available tools are listed below.\n\n"
    "{tool_catalog}\n\n"
    "Generate a sub-question for each action or reasoning step (perception, "
    'prediction or planning information) as the value of "Sub". Also generate an '
    'answer for each Sub as the value of "Guess_Answer" when you are confident. '
    'Extract the key terms related to the guess answer and list them as the value '
    'of "key_words"; for each keyword consider 2-5 synonyms or alternative '
    'expressions. If you cannot answer a sub-question, set "Missing_flag" to '
    '"False"; otherwise set it to "True". "next_action" must be either '
    '"continue reasoning" or "conclude". Continue generating steps until the '
    'reasoning chain is complete. Finally, put the answer in "final_answer" and '
    'its keywords in "final_answer_keywords".\n\n'
    "Output format example:\n"
    "{\n"
    '    "Question": "",\n'
    '    "Chain": [\n'
    "        {\n"
    '            "Tool": {"function_name": "open_vocabulary_detector", '
    '"parameters": ["", ""]},\n'
    '            "Sub": "",\n'
    '            "Guess_Answer": "",\n'
    '            "key_words": ["word1", "word2"],\n'
    '            "Missing_flag": "",\n'
    '            "next_action": "continue reasoning"\n'
    "        },\n"
    "        {\n"
    '            "Tool": {"function_name": "depth_estimator", "parameters": ["", ""]},\n'
    '            "Sub": "",\n'
    '            "Guess_Answer": "",\n'
    '            "key_words": ["word1", "word2"],\n'
    '            "Missing_flag": "",\n'
    '            "next_action": "conclude"\n'
    "        }\n"
    "    ],\n"
    '    "final_answer_keywords": ["word1", "word2"],\n'
    '    "final_answer": ""\n'
    "}\n\n"
    'STRICTLY FOLLOW THE JSON RESPONSE FORMAT. THE RESPONSE SHOULD START WITH "{". '
    'DO NOT START WITH "```json" OR ANYTHING ELSE.'
)

#: Prefix appended to the question when tool results have already been retrieved
#: (full AgentThink inference, ``evaluation/inference_agentthink.py``).
TOOL_RESULT_TEMPLATE = (
    "{system_prompt}\n"
    "To answer the question, please refer to the tool recommendation results which "
    "show in the following dict: (Note: the numerical results are all based on the "
    "ego-car coordination axis.)\n{tool_result}"
)

#: System prompt used during GRPO (paper Appendix B.2).
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in "
    "the mind and then provide the user with the answer."
)

#: System prompt of the baseline ``inference.py`` / ``inference_agentthink.py``.
EVAL_SYSTEM_MESSAGE = "You are the helpful assistant!"


def build_tool_cot_prompt(tool_catalog: str) -> str:
    """Fill :data:`TOOL_COT_PROMPT` with the catalog of available tools.

    ``str.replace`` is used instead of ``str.format`` because the template
    contains a JSON example with literal braces.
    """
    return TOOL_COT_PROMPT.replace("{tool_catalog}", tool_catalog)


# ---------------------------------------------------------------------------
# LLM-as-Judge prompts (DriveLMM-o1 protocol, paper Appendix C)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """
You are an autonomous driving reasoning evaluator. Your task is to assess the alignment, coherence, and quality of reasoning steps in text responses for safety-critical driving scenarios.

You will evaluate the model-generated reasoning using the following metrics:

1. Faithfulness-Step (1-10)
Measures how well the model's reasoning steps align with the ground truth.
9-10: All steps correctly match or closely reflect the reference.
7-8: Most steps align, with minor deviations.
5-6: Some steps align, but several are incorrect or missing.
3-4: Few steps align; most are inaccurate or missing.
1-2: Majority of steps are incorrect.

2. Informativeness-Step (1-10)
Measures completeness of reasoning:
9-10: Captures almost all critical information.
7-8: Covers most key points, with minor omissions.
5-6: Missing significant details.
3-4: Only partial reasoning present.
1-2: Poor extraction of relevant reasoning.

3. Risk Assessment Accuracy (1-10)
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

5. Scene Awareness & Object Understanding (1-10)
Measures how well the response interprets objects, their positions, and actions.
9-10: Clearly understands all relevant objects and their relationships.
7-8: Minor misinterpretations but mostly correct.
5-6: Some key objects misunderstood or ignored.
3-4: Many errors in object recognition and reasoning.
1-2: Misidentifies or ignores key objects.

6. Repetition-Token (1-10)
Identifies unnecessary repetition in reasoning.
9-10: No redundancy, very concise.
7-8: Minor repetition but still clear.
5-6: Noticeable redundancy.
3-4: Frequent repetition that disrupts reasoning.
1-2: Excessive redundancy, making reasoning unclear.

7. Hallucination (1-10)
Detects irrelevant or invented reasoning steps not aligned with ground truth.
9-10: No hallucinations, all reasoning is grounded.
7-8: One or two minor hallucinations.
5-6: Some fabricated details.
3-4: Frequent hallucinations.
1-2: Majority of reasoning is hallucinated.

8. Semantic Coverage-Step (1-10)
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

#: Keys produced by the judge. ``Overall Score`` is the mean of the other keys.
JUDGE_METRIC_KEYS = [
    "Faithfulness-Step",
    "Informativeness-Step",
    "Risk Assessment Accuracy",
    "Traffic Rule Adherence",
    "Scene Awareness & Object Understanding",
    "Repetition-Token",
    "Hallucination",
    "Semantic Coverage-Step",
    "Commonsense Reasoning",
    "Missing Step",
    "Relevance",
    "Missing Details",
]

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvaluationScores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {key: {"type": "number"} for key in JUDGE_METRIC_KEYS}
            | {"Overall Score": {"type": "number"}},
            "required": JUDGE_METRIC_KEYS + ["Overall Score"],
            "additionalProperties": False,
        },
    },
}

#: Mapping from judge keys to the columns reported in the paper (Table 3).
#: ``Reason`` is the commonsense reasoning score and ``Missing`` is the
#: missing-details score; ``Overall`` is the mean of all twelve judge metrics.
PAPER_METRIC_COLUMNS = {
    "Risk Assess.": "Risk Assessment Accuracy",
    "Rule Adh.": "Traffic Rule Adherence",
    "Scene Aware.": "Scene Awareness & Object Understanding",
    "Relevance": "Relevance",
    "Missing": "Missing Details",
    "Reason.": "Commonsense Reasoning",
    "Overall": "Overall Score",
}


# ---------------------------------------------------------------------------
# Tool-use metrics (paper Table 2, Sec. 3.3)
# ---------------------------------------------------------------------------

TOOL_JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator of tool-augmented reasoning for autonomous driving.
You will be given a driving question, the reference reasoning, and a model
response that may contain tool invocations of the form Tool(name, parameters).

Score the response on the following three metrics. Each metric is an integer
from 1 to 10.

1. Tool Usage Appropriateness
Assesses whether tools are logically selected and meaningfully used to support
individual reasoning steps.
9-10: Every tool call is necessary, well-timed, and supports the step it serves.
7-8: Mostly appropriate, with one questionable or redundant call.
5-6: Several calls are unnecessary, mistimed, or unrelated to the step.
3-4: Tool use is largely arbitrary and does not support the reasoning.
1-2: No meaningful tool use, or tools are selected at random.

2. Tool Chain Coherence
Evaluates whether the sequence of tool invocations is clear, logically ordered,
and efficiently contributes to reasoning.
9-10: The chain is minimal, dependency-ordered, and leads directly to the answer.
7-8: Mostly coherent with minor ordering issues or redundancy.
5-6: The chain is partially ordered and contains noticeable detours.
3-4: The chain is disorganized or repetitive.
1-2: No coherent chain of tool calls.

3. Perception-Guided Tool Alignment
Measures how well tool usage aligns with multimodal inputs, including visual
observations and scene context.
9-10: Every call is grounded in specific visual evidence and scene context.
7-8: Mostly grounded, with minor mismatches.
5-6: Some calls are only loosely related to the observed scene.
3-4: Tool calls rarely reflect the visual input.
1-2: Tool use ignores the multimodal input.

Return only a JSON object with the keys "Tool Usage Appropriateness",
"Tool Chain Coherence", "Perception-Guided Tool Alignment" and
"Overall Tool Score" (the mean of the three).
"""

TOOL_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ToolEvaluationScores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Tool Usage Appropriateness": {"type": "number"},
                "Tool Chain Coherence": {"type": "number"},
                "Perception-Guided Tool Alignment": {"type": "number"},
                "Overall Tool Score": {"type": "number"},
            },
            "required": [
                "Tool Usage Appropriateness",
                "Tool Chain Coherence",
                "Perception-Guided Tool Alignment",
                "Overall Tool Score",
            ],
            "additionalProperties": False,
        },
    },
}
