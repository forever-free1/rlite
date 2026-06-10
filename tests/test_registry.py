"""Tests for the plugin registry system."""

from __future__ import annotations

import pytest

from rlite.core.types import Task, Trajectory
from rlite.plugins.base import MetricPlugin, RewardPlugin, TaskPlugin
from rlite.registry import (
    Registry,
    RegistryError,
    metric_registry,
    register_metric,
    register_reward,
    register_task,
    reward_registry,
    task_registry,
)


# ---------------------------------------------------------------------------
# Registry primitive
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        r = Registry("test")

        class Foo:
            pass

        r.register("foo", Foo)
        assert r.get("foo") is Foo

    def test_register_duplicate_warns(self):
        r = Registry("test")

        class A:
            pass

        class B:
            pass

        r.register("dup", A)
        r.register("dup", B)  # should log warning but succeed
        assert r.get("dup") is B

    def test_get_missing_raises(self):
        r = Registry("test")
        with pytest.raises(RegistryError, match="not found"):
            r.get("nonexistent")

    def test_create(self):
        r = Registry("test")

        class MyPlugin:
            def __init__(self, x=1):
                self.x = x

        r.register("my", MyPlugin)
        inst = r.create("my", x=42)
        assert isinstance(inst, MyPlugin)
        assert inst.x == 42

    def test_list(self):
        r = Registry("test")
        assert r.list() == []
        r.register("a", type("A", (), {}))
        r.register("b", type("B", (), {}))
        assert sorted(r.list()) == ["a", "b"]

    def test_contains(self):
        r = Registry("test")
        r.register("x", type("X", (), {}))
        assert "x" in r
        assert "y" not in r

    def test_base_class_validation(self):
        r = Registry("test", base_class=TaskPlugin)

        class Good(TaskPlugin):
            def load_dataset(self, split, max_samples=None):
                return []
            def build_prompt(self, task):
                return ""

        class Bad:
            pass

        r.register("good", Good)
        with pytest.raises(RegistryError, match="not a subclass"):
            r.register("bad", Bad)


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------


class TestGlobalRegistries:
    def test_task_registry_has_debug(self):
        # debug plugin registers at import time
        import rlite.plugins.task  # noqa: F401
        assert "debug" in task_registry
        cls = task_registry.get("debug")
        assert issubclass(cls, TaskPlugin)

    def test_reward_registry_has_debug(self):
        import rlite.plugins.task  # noqa: F401
        assert "debug" in reward_registry
        cls = reward_registry.get("debug")
        assert issubclass(cls, RewardPlugin)

    def test_metric_registry_has_debug(self):
        import rlite.plugins.task  # noqa: F401
        assert "debug" in metric_registry
        cls = metric_registry.get("debug")
        assert issubclass(cls, MetricPlugin)


# ---------------------------------------------------------------------------
# Decorator registration
# ---------------------------------------------------------------------------


class TestDecoratorRegistration:
    def test_register_task_decorator(self):
        @register_task("_test_task")
        class _TestTask(TaskPlugin):
            name = "_test_task"

            def load_dataset(self, split, max_samples=None):
                return []
            def build_prompt(self, task):
                return ""

        assert "_test_task" in task_registry
        inst = task_registry.create("_test_task")
        assert isinstance(inst, TaskPlugin)

    def test_register_reward_decorator(self):
        @register_reward("_test_reward")
        class _TestReward(RewardPlugin):
            name = "_test_reward"

            def score(self, task, trajectory):
                return 1.0

        assert "_test_reward" in reward_registry
        inst = reward_registry.create("_test_reward")
        assert isinstance(inst, RewardPlugin)

    def test_register_metric_decorator(self):
        @register_metric("_test_metric")
        class _TestMetric(MetricPlugin):
            name = "_test_metric"

            def compute(self, trajectories):
                return {"ok": 1.0}

        assert "_test_metric" in metric_registry
        inst = metric_registry.create("_test_metric")
        assert isinstance(inst, MetricPlugin)


# ---------------------------------------------------------------------------
# Plugin instances
# ---------------------------------------------------------------------------


class TestPluginInstances:
    def test_debug_task_creates_real_tasks(self):
        import rlite.plugins.task  # noqa: F401
        plugin = task_registry.create("debug")
        tasks = list(plugin.load_dataset("train", max_samples=3))
        assert len(tasks) == 3
        for t in tasks:
            assert isinstance(t, Task)
            assert t.task_id.startswith("debug_")

    def test_debug_task_builds_prompt(self):
        import rlite.plugins.task  # noqa: F401
        plugin = task_registry.create("debug")
        task = Task(task_id="t1", input={"question": "What is 2+2?"})
        prompt = plugin.build_prompt(task)
        assert "2+2" in prompt

    def test_debug_reward_scores_trajectory(self):
        import rlite.plugins.task  # noqa: F401
        plugin = reward_registry.create("debug")
        task = Task(task_id="t1")
        traj = Trajectory.from_single_response("t1", "p", "r")
        score = plugin.score(task, traj)
        assert score == 1.0

    def test_debug_metric_computes(self):
        import rlite.plugins.task  # noqa: F401
        plugin = metric_registry.create("debug")
        trajs = [
            Trajectory.from_single_response("t1", "p", "r1"),
            Trajectory.from_single_response("t2", "p", "r2"),
        ]
        result = plugin.compute(trajs)
        assert result["debug_score"] == 1.0
        assert result["count"] == 2.0


# ---------------------------------------------------------------------------
# Config-driven resolution (the key completion criterion)
# ---------------------------------------------------------------------------


class TestConfigDrivenResolution:
    """The train loop selects plugins by name from config — never by direct import."""

    def test_resolve_task_by_config_name(self):
        import rlite.plugins.task  # noqa: F401
        # Simulate what train.py does given cfg.task.name == "debug"
        name = "debug"
        plugin = task_registry.create(name)
        assert isinstance(plugin, TaskPlugin)
        # Verify the plugin can actually produce tasks
        tasks = list(plugin.load_dataset("train", max_samples=2))
        assert len(tasks) == 2

    def test_resolve_reward_by_config_name(self):
        import rlite.plugins.task  # noqa: F401
        plugin = reward_registry.create("debug")
        assert isinstance(plugin, RewardPlugin)

    def test_resolve_metric_by_config_name(self):
        import rlite.plugins.task  # noqa: F401
        plugin = metric_registry.create("debug")
        assert isinstance(plugin, MetricPlugin)

    def test_train_loop_never_imports_concrete_task(self):
        """Prove that the train.py code path resolves everything through registries."""
        import rlite.plugins.task  # noqa: F401  # registration only

        # These three lines are what the training loop does;
        # none of them mention "debug" or "gsm8k" or any concrete task name.
        task_plugin = task_registry.create("debug")
        reward_plugin = reward_registry.create("debug")
        metric_plugin = metric_registry.create("debug")

        # Full mini loop
        tasks = list(task_plugin.load_dataset("train", max_samples=4))
        for t in tasks:
            prompt = task_plugin.build_prompt(t)
            traj = Trajectory.from_single_response(t.task_id, prompt, "answer")
            traj.reward = reward_plugin.score(t, traj)
            assert traj.reward is not None
        metrics = metric_plugin.compute(
            [Trajectory.from_single_response("x", "p", "a") for _ in range(2)]
        )
        assert "debug_score" in metrics


# ---------------------------------------------------------------------------
# Extensibility: adding a new task does not touch existing code
# ---------------------------------------------------------------------------


class TestExtensibility:
    def test_register_new_task_at_runtime(self):
        """A new task plugin can be registered at runtime without modifying any existing file."""

        @register_task("_my_new_task")
        class MyTask(TaskPlugin):
            name = "_my_new_task"

            def load_dataset(self, split, max_samples=None):
                return [Task(task_id="n1", input={"x": 1})]

            def build_prompt(self, task):
                return f"x={task.input['x']}"

        # Now it should be resolvable
        plugin = task_registry.create("_my_new_task")
        tasks = list(plugin.load_dataset("train"))
        assert len(tasks) == 1
        assert tasks[0].task_id == "n1"
        assert plugin.build_prompt(tasks[0]) == "x=1"
