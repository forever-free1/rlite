"""reward.py — Reward plugin re-exports and helpers.

Concrete reward plugins live in ``rlite.tasks.<task_name>.reward``
and register themselves via ``@register_reward`` at import time.
"""

from rlite.plugins.base import RewardPlugin
from rlite.registry import register_reward, reward_registry

__all__ = ["RewardPlugin", "reward_registry", "register_reward"]
