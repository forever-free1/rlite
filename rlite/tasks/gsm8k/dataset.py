"""GSM8K dataset loader and task plugin.

Loads the GSM8K dataset from HuggingFace, converts each row into a ``Task``,
and registers the ``GSM8KTask`` plugin with the registry.
"""

from __future__ import annotations

from typing import Iterable

from datasets import load_dataset

from rlite.core.types import Task
from rlite.logging import logger
from rlite.plugins.base import TaskPlugin
from rlite.registry import register_task
from rlite.tasks.gsm8k.reward import extract_answer


def load_gsm8k_dataset(split: str = "train", max_samples: int | None = None) -> list[Task]:
    """Load GSM8K tasks from HuggingFace.

    Args:
        split: ``"train"`` or ``"test"``.
        max_samples: If set, limit the number of tasks returned.

    Returns:
        List of ``Task`` objects ready for prompt building.
    """
    logger.info("Loading GSM8K dataset (split=%s)...", split)
    ds = load_dataset("gsm8k", "main", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        question = row["question"]
        raw_answer = row["answer"]
        numeric_answer = extract_answer(raw_answer) or ""
        tasks.append(
            Task(
                task_id=f"gsm8k_{split}_{i}",
                input={"question": question, "raw_answer": raw_answer},
                target={"answer": numeric_answer},
                metadata={"source": "gsm8k", "split": split, "index": i},
            )
        )
    logger.info("Loaded %d GSM8K tasks", len(tasks))
    return tasks


@register_task("gsm8k")
class GSM8KTask(TaskPlugin):
    """Task plugin for GSM8K math reasoning."""

    name = "gsm8k"

    def load_dataset(self, split: str = "train", max_samples: int | None = None) -> Iterable[Task]:
        return iter(load_gsm8k_dataset(split=split, max_samples=max_samples))

    def build_prompt(self, task: Task) -> str:
        from rlite.tasks.gsm8k.prompt import build_prompt
        return build_prompt(task)
