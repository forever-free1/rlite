"""metric.py — Metric plugin re-exports and helpers.

Concrete metric plugins live in ``rlite.tasks.<task_name>.metrics``
and register themselves via ``@register_metric`` at import time.
"""

from rlite.plugins.base import MetricPlugin
from rlite.registry import metric_registry, register_metric

__all__ = ["MetricPlugin", "metric_registry", "register_metric"]
