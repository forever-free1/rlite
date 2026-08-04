"""Explicit dispatcher for policy objectives."""

from __future__ import annotations

from collections.abc import Callable

from rlite.algos.dapo import dapo_loss
from rlite.algos.grpo import grpo_loss
from rlite.algos.gspo import gspo_loss

ALGORITHMS: dict[str, Callable] = {
    "grpo": grpo_loss,
    "dapo": dapo_loss,
    "gspo": gspo_loss,
}


def get_objective(name: str) -> Callable:
    try:
        return ALGORITHMS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown objective {name!r}; choose one of: {choices}") from exc


get_algorithm = get_objective
