"""GSM8K prompt builder.

Constructs prompts with format instructions that guide the model to
produce step-by-step reasoning followed by a final answer marker.
"""

from __future__ import annotations

from rlite.core.types import Task

# System / instruction prefix added before every question.
# Instructs the model to reason step-by-step and use a parsable answer format.
GSM8K_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer on a new line in the format: #### {answer}\n\n"
)

# Optional few-shot examples (one is sufficient for format teaching).
GSM8K_FEWSHOT = """\
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: The grove started with 15 trees. After planting, there are 21 trees. So they must have planted 21 - 15 = 6 trees. #### 6

"""


def build_prompt(task: Task, use_fewshot: bool = True) -> str:
    """Build a prompt string for a GSM8K task.

    Args:
        task: The ``Task`` containing ``input["question"]``.
        use_fewshot: Include a one-shot example to teach the answer format.

    Returns:
        Prompt text ready for tokenisation.
    """
    question = task.input["question"]
    parts = [GSM8K_SYSTEM_PROMPT]
    if use_fewshot:
        parts.append(GSM8K_FEWSHOT)
    parts.append(f"Question: {question}\nAnswer:")
    return "".join(parts)
