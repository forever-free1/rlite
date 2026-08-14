"""Configuration system: YAML-backed dataclass config for rlite."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    name: str = "debug"
    dataset_path: str = ""
    split: str = "train"
    max_samples: int | None = None


@dataclass
class RewardConfig:
    name: str = "debug"
    # reward-specific overrides (e.g. weights for format / answer)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutConfig:
    engine: str = "hf"  # "hf" | "vllm"
    n_samples: int = 4  # K responses per prompt
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    model_name: str = ""
    dtype: str = "bfloat16"
    # vLLM-specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    enable_prefix_caching: bool = True
    group_admission: str = "native"  # "native" | "leader"


@dataclass
class AlgoConfig:
    name: str = "grpo"  # "grpo" | "dapo" | "gspo"
    eps: float = 0.2  # clip epsilon
    eps_high: float = 0.28  # DAPO asymmetric upper clip
    kl_coef: float = 0.0  # optional KL penalty
    dynamic_sampling_max_rounds: int = 8


@dataclass
class TrainerConfig:
    method: str = "lora"  # "lora" | "deepspeed_lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    train_steps: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    micro_batch_size_per_gpu: int = 1
    max_tokens_per_micro_batch: int | None = 1024
    save_steps: int = 50
    eval_steps: int = 50
    log_steps: int = 10
    seed: int = 42
    output_dir: str = "./outputs"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    wandb_project: str = ""
    wandb_run_name: str = ""
    log_dir: str = "./logs"


# ---------------------------------------------------------------------------
# root config
# ---------------------------------------------------------------------------


@dataclass
class RLiteConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # convenience: the path this config was loaded from
    _config_path: str | None = field(default=None, repr=False)

    def validate(self) -> None:
        """Reject configurations whose advertised behaviour is unsupported."""
        if self.algo.name not in {"grpo", "dapo", "gspo"}:
            raise ValueError("algo.name must be 'grpo', 'dapo', or 'gspo'")
        if self.algo.eps <= 0 or self.algo.eps_high <= 0:
            raise ValueError("algorithm clip ranges must be positive")
        if self.algo.dynamic_sampling_max_rounds <= 0:
            raise ValueError("dynamic_sampling_max_rounds must be positive")
        if self.rollout.engine not in {"hf", "vllm"}:
            raise ValueError("rollout.engine must be 'hf' or 'vllm'")
        if self.trainer.method != "lora":
            raise ValueError("Only trainer.method='lora' is currently implemented")
        if self.task.split == "train" and self.rollout.n_samples < 2:
            raise ValueError("Group-relative algorithms require rollout.n_samples >= 2")
        if self.trainer.batch_size <= 0 or self.trainer.train_steps <= 0:
            raise ValueError("trainer batch_size and train_steps must be positive")
        if self.trainer.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.trainer.micro_batch_size_per_gpu <= 0:
            raise ValueError("micro_batch_size_per_gpu must be positive")
        if (self.trainer.max_tokens_per_micro_batch is not None
                and self.trainer.max_tokens_per_micro_batch <= 0):
            raise ValueError("max_tokens_per_micro_batch must be positive when set")
        if not 0.0 <= self.rollout.top_p <= 1.0:
            raise ValueError("rollout.top_p must be in [0, 1]")
        if self.rollout.group_admission not in {"native", "leader"}:
            raise ValueError("rollout.group_admission must be 'native' or 'leader'")
        if self.rollout.max_model_len is not None and self.rollout.max_model_len <= 0:
            raise ValueError("rollout.max_model_len must be positive when set")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> RLiteConfig:
    """Load an RLiteConfig from a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = RLiteConfig(
        task=TaskConfig(**raw.get("task", {})),
        reward=RewardConfig(**raw.get("reward", {})),
        rollout=RolloutConfig(**raw.get("rollout", {})),
        algo=AlgoConfig(**raw.get("algo", {})),
        trainer=TrainerConfig(**raw.get("trainer", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        _config_path=str(path.resolve()),
    )
    cfg.validate()
    return cfg


def save_config(config: RLiteConfig, path: str | Path) -> None:
    """Save current config as YAML (useful for experiment tracking)."""
    path = Path(path)
    _dict = {
        "task": config.task.__dict__,
        "reward": config.reward.__dict__,
        "rollout": config.rollout.__dict__,
        "algo": {
            "name": config.algo.name,
            "eps": config.algo.eps,
            "eps_high": config.algo.eps_high,
            "kl_coef": config.algo.kl_coef,
            "dynamic_sampling_max_rounds": config.algo.dynamic_sampling_max_rounds,
        },
        "trainer": config.trainer.__dict__,
        "logging": config.logging.__dict__,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_dict, f, default_flow_style=False, allow_unicode=True)
