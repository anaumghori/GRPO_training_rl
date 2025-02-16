from math_verify import LatexExtractionConfig, parse, verify
import re

def format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion follows either a strict or loose format.

    - Strict format (1.0 reward): Requires newlines around <reasoning> and <answer>.
    - Loose format (0.5 reward): Allows a more flexible structure without strict newlines.
    """

    strict_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    loose_pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$"

    completion_contents = [completion[0]["content"] for completion in completions]

    reward_list = []
    for content in completion_contents:
        if re.match(strict_pattern, content):
            reward_list.append(1.0)  # Strict format matched
        elif re.match(loose_pattern, content):
            reward_list.append(0.5)  # Loose format matched
        else:
            reward_list.append(0.0)  # No valid format

    return reward_list

def accuracy_reward(completions, **kwargs) -> list[float]:
    """Evaluates completions by checking if they match the correct solution."""
    solutions = kwargs.get("solution", [])
    completion_contents = [c[0]["content"] for c in completions]
    rewards = []

    for content, solution in zip(completion_contents, solutions):
        parsed_solution = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        parsed_content = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

        if not parsed_solution:
            rewards.append(1.0)  # Default reward when no valid solution exists
            continue
        try:
            rewards.append(float(verify(parsed_content, parsed_solution)))
        except Exception:
            rewards.append(0.0)  # Return 0 if verification fails

    return rewards
