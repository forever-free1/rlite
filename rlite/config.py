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


@dataclass
class DAPODynamicSamplingConfig:
    enabled: bool = False


@dataclass
class DAPOClipHigherConfig:
    enabled: bool = False
    eps_low: float = 0.2
    eps_high: float = 0.28


@dataclass
class DAPOTokenLevelLossConfig:
    enabled: bool = False


@dataclass
class DAPOOverlongPenaltyConfig:
    enabled: bool = False
    max_len: int = 1024
    penalty: float = -0.2


@dataclass
class DAPOConfig:
    enabled: bool = False
    dynamic_sampling: DAPODynamicSamplingConfig = field(
        default_factory=DAPODynamicSamplingConfig
    )
    clip_higher: DAPOClipHigherConfig = field(default_factory=DAPOClipHigherConfig)
    token_level_loss: DAPOTokenLevelLossConfig = field(
        default_factory=DAPOTokenLevelLossConfig
    )
    overlong_penalty: DAPOOverlongPenaltyConfig = field(
        default_factory=DAPOOverlongPenaltyConfig
    )


@dataclass
class AlgoConfig:
    name: str = "grpo"  # "grpo"
    eps: float = 0.2  # clip epsilon
    kl_coef: float = 0.0  # optional KL penalty
    dapo: DAPOConfig = field(default_factory=DAPOConfig)


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


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _dict_to_dapo_config(d: dict[str, Any] | None) -> DAPOConfig:
    if d is None:
        return DAPOConfig()
    return DAPOConfig(
        enabled=d.get("enabled", False),
        dynamic_sampling=DAPODynamicSamplingConfig(
            **d.get("dynamic_sampling", {})
        ),
        clip_higher=DAPOClipHigherConfig(**d.get("clip_higher", {})),
        token_level_loss=DAPOTokenLevelLossConfig(
            **d.get("token_level_loss", {})
        ),
        overlong_penalty=DAPOOverlongPenaltyConfig(
            **d.get("overlong_penalty", {})
        ),
    )


def load_config(path: str | Path) -> RLiteConfig:
    """Load an RLiteConfig from a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = RLiteConfig(
        task=TaskConfig(**raw.get("task", {})),
        reward=RewardConfig(**raw.get("reward", {})),
        rollout=RolloutConfig(**raw.get("rollout", {})),
        algo=AlgoConfig(
            **{k: v for k, v in raw.get("algo", {}).items() if k != "dapo"},
            dapo=_dict_to_dapo_config(raw.get("algo", {}).get("dapo")),
        ),
        trainer=TrainerConfig(**raw.get("trainer", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        _config_path=str(path.resolve()),
    )
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
            "kl_coef": config.algo.kl_coef,
            "dapo": {
                "enabled": config.algo.dapo.enabled,
                "dynamic_sampling": config.algo.dapo.dynamic_sampling.__dict__,
                "clip_higher": config.algo.dapo.clip_higher.__dict__,
                "token_level_loss": config.algo.dapo.token_level_loss.__dict__,
                "overlong_penalty": config.algo.dapo.overlong_penalty.__dict__,
            },
        },
        "trainer": config.trainer.__dict__,
        "logging": config.logging.__dict__,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_dict, f, default_flow_style=False, allow_unicode=True)
