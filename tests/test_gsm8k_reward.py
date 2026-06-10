"""Tests for GSM8K reward plugin and answer extraction."""

from __future__ import annotations

import pytest

from rlite.core.types import Task, Trajectory
from rlite.tasks.gsm8k.reward import (
    GSM8KReward,
    _numbers_match,
    extract_answer,
)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_hash_format(self):
        """#### followed by number."""
        assert extract_answer("Some reasoning\n#### 72") == "72"

    def test_hash_format_with_space(self):
        assert extract_answer("Step 1: ...\n####  42") == "42"

    def test_boxed_format(self):
        """\\boxed{number} LaTeX format."""
        assert extract_answer("Therefore \\boxed{15} is the answer.") == "15"

    def test_last_number_fallback(self):
        """When no marker is present, pick the last number."""
        assert extract_answer("The answer is 42.") == "42"

    def test_decimal_number(self):
        assert extract_answer("#### 3.14") == "3.14"

    def test_negative_number(self):
        assert extract_answer("Result is #### -5") == "-5"

    def test_comma_separated_number(self):
        """GSM8K sometimes has comma-formatted numbers."""
        assert extract_answer("Total is #### 1,234") == "1234"

    def test_comma_in_boxed(self):
        assert extract_answer("\\boxed{12,345}") == "12345"

    def test_negative_decimal(self):
        assert extract_answer("Answer: #### -3.5") == "-3.5"

    def test_multiple_numbers_extracts_last(self):
        """With no marker, the last number in the text is used."""
        # "x=5, y=10" → extract "10"
        result = extract_answer("We have 5 apples and 10 oranges.")
        assert result == "10"

    def test_hash_priority_over_last_number(self):
        """#### marker takes priority even if there are later numbers."""
        text = "#### 72\nThen we check 100 times."
        assert extract_answer(text) == "72"

    def test_boxed_priority_over_last_number(self):
        text = "The value is \\boxed{7}, but we also see 99."
        assert extract_answer(text) == "7"

    def test_empty_string(self):
        assert extract_answer("") is None

    def test_no_numbers(self):
        assert extract_answer("Just some text with no numbers.") is None

    def test_none_input(self):
        assert extract_answer(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Number matching
# ---------------------------------------------------------------------------


class TestNumbersMatch:
    def test_exact_integers(self):
        assert _numbers_match("72", "72")

    def test_float_equivalence(self):
        assert _numbers_match("3.0", "3")
        assert _numbers_match("3.140", "3.14")

    def test_negative_match(self):
        assert _numbers_match("-5", "-5")

    def test_different_values(self):
        assert not _numbers_match("5", "6")

    def test_non_numeric_fallback(self):
        """When values aren't numeric, fall back to string comparison."""
        assert not _numbers_match("abc", "def")
        assert _numbers_match("abc", "abc")


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------


class TestGSM8KReward:
    @pytest.fixture
    def reward_plugin(self):
        return GSM8KReward()

    @pytest.fixture
    def task(self):
        return Task(
            task_id="gsm8k_test_0",
            input={"question": "What is 1+1?"},
            target={"answer": "2"},
        )

    def _make_traj(self, task_id: str, response: str) -> Trajectory:
        return Trajectory.from_single_response(
            task_id=task_id,
            prompt="Question: What is 1+1?\nAnswer:",
            response=response,
        )

    def test_correct_with_hash_format(self, reward_plugin, task):
        """Correct answer + proper format → 1.0 + 0.5 = 1.5"""
        traj = self._make_traj(task.task_id, "We have 1+1 which is 2.\n#### 2")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(1.5)

    def test_correct_with_boxed_format(self, reward_plugin, task):
        """Correct answer + boxed format → 1.0 + 0.5 = 1.5"""
        traj = self._make_traj(task.task_id, "Let's solve: 1+1=2, so \\boxed{2}")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(1.5)

    def test_correct_no_format(self, reward_plugin, task):
        """Correct answer extracted from last number, no format marker → 1.0"""
        traj = self._make_traj(task.task_id, "The answer should be 2.")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(1.0)

    def test_wrong_answer_with_format(self, reward_plugin, task):
        """Wrong answer with format → 0.0 + 0.5 = 0.5"""
        traj = self._make_traj(task.task_id, "I think #### 3")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(0.5)

    def test_wrong_answer_no_format(self, reward_plugin, task):
        """Wrong answer, no format → 0.0"""
        traj = self._make_traj(task.task_id, "The answer is probably 3.")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(0.0)

    def test_no_extractable_answer(self, reward_plugin, task):
        """No number at all → invalid penalty -0.5"""
        traj = self._make_traj(task.task_id, "I don't know the answer.")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(-0.5)

    def test_empty_response(self, reward_plugin, task):
        traj = self._make_traj(task.task_id, "")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(-0.5)

    def test_decimal_target(self, reward_plugin):
        task = Task(
            task_id="decimal_test",
            input={"question": "What is 10/4?"},
            target={"answer": "2.5"},
        )
        traj = self._make_traj(task.task_id, "10/4 = 2.5\n#### 2.5")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(1.5)

    def test_negative_answer(self, reward_plugin):
        task = Task(
            task_id="neg_test",
            input={"question": "5 - 8 = ?"},
            target={"answer": "-3"},
        )
        traj = self._make_traj(task.task_id, "5-8 = -3\n#### -3")
        score = reward_plugin.score(task, traj)
        assert score == pytest.approx(1.5)

    def test_custom_weights(self):
        """Custom reward weights are applied correctly."""
        reward = GSM8KReward(
            exact_match_weight=2.0,
            format_reward_weight=0.0,
            invalid_penalty=-1.0,
        )
        task = Task(task_id="t", target={"answer": "5"})
        traj = Trajectory.from_single_response("t", "p", "#### 5")
        score = reward.score(task, traj)
        assert score == pytest.approx(2.0)  # 2.0 exact + 0.0 format

    def test_validate_detects_degenerate(self, caplog):
        reward = GSM8KReward()
        reward.validate([0.5, 0.5, 0.5])  # all identical
        # Warning should be logged
        assert "degenerate" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Dataset answer extraction
# ---------------------------------------------------------------------------


class TestDatasetAnswerExtraction:
    def test_gsm8k_answer_parsing(self):
        """Simulate real GSM8K answer format: explanation + #### number.
        Uses extract_answer from reward.py (which dataset.py also delegates to)."""
        gsm8k_style = (
            "Natalia sold 48 clips in April and 24 in May. "
            "48 + 24 = 72 clips total. #### 72"
        )
        assert extract_answer(gsm8k_style) == "72"

    def test_gsm8k_comma_answer(self):
        assert extract_answer("Total: #### 1,200") == "1200"


# ---------------------------------------------------------------------------
# Integration: reward + metrics pipeline
# ---------------------------------------------------------------------------


class TestGSM8KPipeline:
    def test_full_reward_then_metrics(self):
        """Simulate a mini rollout → reward → metrics cycle."""
        reward = GSM8KReward()
        from rlite.tasks.gsm8k.metrics import GSM8KMetric

        metric = GSM8KMetric()

        tasks = [
            Task(task_id="a", target={"answer": "10"}),
            Task(task_id="b", target={"answer": "20"}),
            Task(task_id="c", target={"answer": "30"}),
            Task(task_id="d", target={"answer": "40"}),
        ]

        responses = [
            "Step by step: ...\n#### 10",  # correct + format
            "#### 25",  # wrong + format
            "I think the answer is 30.",  # correct, no format
            "...",  # no extractable answer
        ]

        trajectories = [
            Trajectory.from_single_response(t.task_id, "p", r)
            for t, r in zip(tasks, responses)
        ]

        for traj, task in zip(trajectories, tasks):
            traj.reward = reward.score(task, traj)

        scores = [t.reward for t in trajectories]
        assert scores[0] == pytest.approx(1.5)  # correct + format
        assert scores[1] == pytest.approx(0.5)  # wrong + format
        assert scores[2] == pytest.approx(1.0)  # correct, no format
        assert scores[3] == pytest.approx(-0.5)  # invalid

        metrics = metric.compute(trajectories)
        assert metrics["exact_match"] == pytest.approx(0.5)   # 2 out of 4 (rewards >= 1.0)
        assert metrics["invalid_format_rate"] == pytest.approx(0.25)  # 1 out of 4
        assert metrics["avg_response_length"] > 0
