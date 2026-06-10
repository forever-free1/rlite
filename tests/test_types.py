"""Tests for core data types: Task, Step, Trajectory, RolloutReq, RolloutResp."""

from __future__ import annotations

import pytest

from rlite.core.types import MetricResult, RewardResult, Step, Task, Trajectory
from rlite.core.rollout_types import RolloutReq, RolloutResp


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TestTask:
    def test_create_minimal(self):
        t = Task(task_id="test_001")
        assert t.task_id == "test_001"
        assert t.input == {}
        assert t.target is None
        assert t.metadata == {}

    def test_create_full(self):
        t = Task(
            task_id="gsm8k_001",
            input={"question": "What is 1+1?"},
            target={"answer": "2"},
            metadata={"source": "gsm8k"},
        )
        assert t.task_id == "gsm8k_001"
        assert t.input["question"] == "What is 1+1?"
        assert t.target["answer"] == "2"
        assert t.metadata["source"] == "gsm8k"

    def test_raw_dict_to_task(self):
        """Verify that a raw dataset sample can be converted to a Task."""
        raw = {"question": "x + 1 = 3, x = ?", "answer": "2"}
        t = Task(
            task_id="math_001",
            input={"question": raw["question"]},
            target={"answer": raw["answer"]},
        )
        assert isinstance(t, Task)
        assert t.input["question"] == raw["question"]


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_create_basic(self):
        s = Step(prompt="hello", response="world")
        assert s.prompt == "hello"
        assert s.response == "world"
        assert s.token_ids is None
        assert s.logprobs is None

    def test_with_token_ids(self):
        s = Step(
            prompt="hi",
            response="there",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
        )
        assert s.token_ids == [1, 2, 3]
        assert s.logprobs == [-0.1, -0.2, -0.3]


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_from_single_response(self):
        traj = Trajectory.from_single_response(
            task_id="gsm8k_001",
            prompt="Question: 1+1?\nAnswer:",
            response="The answer is 2.",
        )
        assert traj.task_id == "gsm8k_001"
        assert len(traj.steps) == 1
        assert traj.steps[0].prompt == "Question: 1+1?\nAnswer:"
        assert traj.steps[0].response == "The answer is 2."
        assert traj.final_response == "The answer is 2."
        assert traj.reward is None
        assert traj.advantage is None

    def test_multi_step(self):
        traj = Trajectory(
            task_id="code_001",
            steps=[
                Step(prompt="fix this:", response="def foo(): pass"),
                Step(prompt="tests:", response="all pass"),
            ],
            final_response="all pass",
        )
        assert len(traj.steps) == 2
        assert traj.final_response == "all pass"

    def test_reward_assignable(self):
        traj = Trajectory.from_single_response("t1", "p", "r")
        traj.reward = 0.8
        traj.advantage = 0.3
        assert traj.reward == 0.8
        assert traj.advantage == 0.3


# ---------------------------------------------------------------------------
# RewardResult
# ---------------------------------------------------------------------------


class TestRewardResult:
    def test_default(self):
        r = RewardResult(score=1.0)
        assert r.score == 1.0
        assert r.details == {}
        assert r.valid is True

    def test_with_details(self):
        r = RewardResult(
            score=1.5,
            details={"exact_match": 1.0, "format": 0.5},
        )
        assert r.details["exact_match"] == 1.0
        assert r.details["format"] == 0.5

    def test_invalid(self):
        r = RewardResult(score=0.0, valid=False)
        assert not r.valid


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_compute(self):
        m = MetricResult(
            metrics={"exact_match": 0.75, "format_ok": 0.9},
            n_samples=100,
        )
        assert m.metrics["exact_match"] == 0.75
        assert m.n_samples == 100


# ---------------------------------------------------------------------------
# RolloutReq
# ---------------------------------------------------------------------------


class TestRolloutReq:
    def test_create(self):
        tasks = [
            Task(task_id="a", input={"q": "?"}, target={"a": "1"}),
            Task(task_id="b", input={"q": "??"}, target={"a": "2"}),
        ]
        req = RolloutReq(
            batch_id="batch_001",
            tasks=tasks,
            prompts=["Q: ?", "Q: ??"],
            n_samples=4,
            temperature=1.0,
            top_p=0.95,
            max_tokens=256,
            policy_version=1,
        )
        assert req.batch_id == "batch_001"
        assert len(req.prompts) == 2
        assert req.n_samples == 4
        assert req.policy_version == 1


# ---------------------------------------------------------------------------
# RolloutResp
# ---------------------------------------------------------------------------


class TestRolloutResp:
    def test_create(self):
        trajs = [
            Trajectory.from_single_response("a", "p1", "r1"),
            Trajectory.from_single_response("b", "p2", "r2"),
        ]
        resp = RolloutResp(
            batch_id="batch_001",
            trajectories=trajs,
            policy_version=1,
        )
        assert resp.batch_id == "batch_001"
        assert len(resp.trajectories) == 2
        assert resp.policy_version == 1

    def test_reward_plugins_receive_valid_trajectories(self):
        """RolloutResp → RewardPlugin: each trajectory has task_id + response."""
        trajs = [
            Trajectory.from_single_response("t1", "prompt", "answer1"),
            Trajectory.from_single_response("t2", "prompt", "answer2"),
        ]
        resp = RolloutResp("b1", trajectories=trajs)
        for t in resp.trajectories:
            assert t.task_id is not None
            assert t.final_response is not None


# ---------------------------------------------------------------------------
# End-to-end data flow
# ---------------------------------------------------------------------------


class TestDataFlow:
    def test_raw_to_grpo_batch_structure(self):
        """Simulate: raw dataset → Task → prompts → Trajectory → reward → batch."""
        # 1. raw → Task
        raw_samples = [
            {"q": "1+1", "a": "2"},
            {"q": "2+2", "a": "4"},
            {"q": "3+3", "a": "6"},
        ]
        tasks = [
            Task(
                task_id=f"math_{i}",
                input={"question": s["q"]},
                target={"answer": s["a"]},
            )
            for i, s in enumerate(raw_samples)
        ]
        assert len(tasks) == 3

        # 2. build prompts
        prompts = [f"Q: {t.input['question']}\nA:" for t in tasks]

        # 3. simulate K=2 responses per prompt
        trajectories: list[Trajectory] = []
        for t in tasks:
            for k in range(2):
                traj = Trajectory.from_single_response(
                    task_id=t.task_id,
                    prompt=prompts[0],
                    response=f"response_{t.task_id}_sample{k}",
                )
                trajectories.append(traj)
        assert len(trajectories) == 6  # 3 tasks × 2 samples

        # 4. assign rewards (simulated)
        for traj in trajectories:
            traj.reward = float(len(traj.final_response))  # dummy reward
            traj.advantage = 0.0

        # 5. group by task_id (GRPO grouping)
        groups: dict[str, list[Trajectory]] = {}
        for traj in trajectories:
            groups.setdefault(traj.task_id, []).append(traj)
        assert len(groups) == 3
        for g in groups.values():
            assert len(g) == 2  # K=2 per group

        # 6. compute group statistics (not NaN)
        for g in groups.values():
            rewards = [t.reward for t in g]
            mu = sum(rewards) / len(rewards)
            # all rewards are the same length, so std=0 → advantage = 0
            import math

            std = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
            assert not (math.isnan(mu) or math.isnan(std))

    def test_nan_protection_all_wrong(self):
        """All-wrong group (all rewards equal) should not NaN."""
        trajs = [
            Trajectory.from_single_response(f"t_{i}", "p", f"r_{i}") for i in range(4)
        ]
        for t in trajs:
            t.reward = 0.0  # all same

        rewards = [t.reward for t in trajs]
        mu = sum(rewards) / len(rewards)
        import math

        std = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
        # std=0 is fine — advantage formula must check for this
        assert math.isclose(mu, 0.0)
        assert math.isclose(std, 0.0)


# ---------------------------------------------------------------------------
# Serialisation / round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_task_round_trip(self):
        t = Task(
            task_id="x",
            input={"a": 1, "b": [1, 2]},
            target={"ans": 42},
        )
        # dataclass fields survive
        assert t.input["a"] == 1
        assert t.target["ans"] == 42

    def test_trajectory_round_trip(self):
        traj = Trajectory.from_single_response(
            "tid", "hello?", "world!", token_ids=[0, 1, 2]
        )
        assert traj.task_id == "tid"
        assert traj.steps[0].token_ids == [0, 1, 2]
