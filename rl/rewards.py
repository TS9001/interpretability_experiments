"""Reward functions for GRPO training on GSM8K.

Each reward function follows the GRPOTrainer signature:
    (completions, **kwargs) -> list[float]
"""

import re


def extract_final_answer(text: str) -> float | None:
    """Extract the numeric answer from a completion.

    Looks for #### <number> pattern first, then falls back to the last number in the text.
    """
    # Try #### pattern (GSM8K standard format)
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))

    # Fallback: last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    return None


def correctness_reward(completions: list[list[dict]], answer: list[str], **kwargs) -> list[float]:
    """Binary reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise."""
    rewards = []
    for completion, gt in zip(completions, answer):
        text = completion[0]["content"]
        predicted = extract_final_answer(text)
        gt_value = extract_final_answer(gt)

        if predicted is not None and gt_value is not None and abs(predicted - gt_value) < 1e-6:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Soft reward for following GSM8K answer format.

    +0.5 for containing #### marker
    +0.5 for having step-by-step reasoning (multiple lines with numbers)
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        score = 0.0

        if "####" in text:
            score += 0.5

        # Check for step-by-step: at least 2 lines containing numbers/equations
        lines_with_math = [l for l in text.split("\n") if re.search(r"\d+\s*[+\-*/×÷=]\s*\d+", l)]
        if len(lines_with_math) >= 2:
            score += 0.5

        rewards.append(score)
    return rewards
