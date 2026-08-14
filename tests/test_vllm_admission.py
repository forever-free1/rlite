"""Pure-Python checks for leader-gated request admission."""

from __future__ import annotations

from types import SimpleNamespace

from rlite.core.rollout_types import RolloutReq
from rlite.core.types import Task
from rlite.rollout.vllm_engine import VLLMRolloutEngine


class FakeEngine:
    def __init__(self):
        self.pending = []
        self.added = []

    def add_request(self, request_id, prompt, sampling_params, **kwargs):
        self.pending.append(request_id)
        self.added.append(request_id)

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        current, self.pending = self.pending, []
        return [
            SimpleNamespace(
                request_id=request_id,
                finished=True,
                num_cached_tokens=0,
                outputs=[SimpleNamespace(token_ids=[1], text=request_id)],
            )
            for request_id in current
        ]


def test_leaders_are_admitted_before_followers():
    fake = FakeEngine()
    rollout = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
    rollout.llm = SimpleNamespace(llm_engine=fake)
    request = RolloutReq(
        batch_id="unit",
        tasks=[Task("a"), Task("b")],
        prompts=["A", "B"],
        n_samples=3,
        policy_version=7,
    )

    grouped = rollout._generate_leader(request, object(), None)

    assert len(grouped) == 2
    assert all(len(group) == 3 for group in grouped)
    assert [request_id.split("-s")[1].split("-")[0] for request_id in fake.added[:2]] == ["0", "0"]
    assert sorted(request_id.split("-s")[1].split("-")[0] for request_id in fake.added[2:]) == ["1", "1", "2", "2"]
