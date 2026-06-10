"""Plugin registry: register and look up task/reward/metric plugins by name."""

from __future__ import annotations

from typing import Any, TypeVar

from rlite.logging import logger

T = TypeVar("T")


class RegistryError(Exception):
    """Raised when a plugin cannot be found or registered."""


class Registry:
    """A simple name→class registry with validation.

    Each registry maps string names to plugin *classes* (not instances).
    Instances are created via ``create(name, **kwargs)``.
    """

    def __init__(self, name: str, base_class: type | None = None):
        self.name = name
        self.base_class = base_class
        self._entries: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        """Register a plugin class under *name*."""
        if self.base_class is not None and not issubclass(cls, self.base_class):
            raise RegistryError(
                f"{cls.__name__} is not a subclass of {self.base_class.__name__}"
            )
        if name in self._entries and self._entries[name] is not cls:
            logger.warning(
                "Registry '%s': overwriting '%s' (%s → %s)",
                self.name,
                name,
                self._entries[name].__name__,
                cls.__name__,
            )
        self._entries[name] = cls
        logger.debug("Registry '%s': registered '%s' → %s", self.name, name, cls.__name__)

    def get(self, name: str) -> type:
        """Look up a registered plugin class by name."""
        if name not in self._entries:
            raise RegistryError(
                f"'{name}' not found in registry '{self.name}'. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a plugin instance by name, forwarding *kwargs* to its constructor."""
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list[str]:
        """Return registered plugin names."""
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# ---------------------------------------------------------------------------
# global registries
# ---------------------------------------------------------------------------

# These are initialised empty here; concrete plugins register themselves
# at import time (or explicitly via register()).

task_registry = Registry("task")
reward_registry = Registry("reward")
metric_registry = Registry("metric")


def register_task(name: str):
    """Decorator: register a TaskPlugin class."""
    def _decorator(cls: type) -> type:
        task_registry.register(name, cls)
        return cls
    return _decorator


def register_reward(name: str):
    """Decorator: register a RewardPlugin class."""
    def _decorator(cls: type) -> type:
        reward_registry.register(name, cls)
        return cls
    return _decorator


def register_metric(name: str):
    """Decorator: register a MetricPlugin class."""
    def _decorator(cls: type) -> type:
        metric_registry.register(name, cls)
        return cls
    return _decorator
