# rlite —— 插件化 LoRA-GRPO/DAPO 低资源 LLM RL 后训练框架实现计划

## 0. 一句话定位

`rlite` 是一个面向低资源场景的插件化 LLM RL 后训练框架，核心能力是：

> **通过可插拔 Task / Reward / Metric 插件，在不修改 GRPO/DAPO + LoRA 训练内核的情况下，训练数学推理、格式遵循、代码修复、工具调用等可验证能力。**

英文定位：

> **rlite: A plugin-based LoRA-first GRPO/DAPO framework for verifiable LLM skills.**

---

## 1. 项目目标

### 1.1 核心目标

实现一个小而完整的 LLM RL 后训练系统，重点不是“大而全”，而是：

1. **LoRA-first**：默认训练 LoRA adapter，而不是 full-parameter RL。
2. **GRPO/DAPO-only**：只实现 GRPO 和 DAPO-style 插件，不扩展 PPO、DPO、RLOO 等其它算法。
3. **Plugin-based**：任务、prompt、reward、metrics 都可插拔。
4. **Low-resource**：默认支持 1.5B / 3B / 7B 模型在 1–4 张 GPU 上训练。
5. **vLLM rollout**：支持 HF generate debug 路径和 vLLM 高吞吐 rollout 路径。
6. **Adapter sync**：重点做 LoRA adapter 同步，而不是全量权重同步。
7. **Readable system**：代码边界清楚，适合作为简历项目和开源项目展示。

---

## 2. 不做什么

为了避免项目发散，v1 明确不做：

```text
PPO
critic model
reward model training
full-parameter RL
ZeRO-3 full weight gather
NCCL full model broadcast
disaggregate rollout/training
fully async RL
DRPO
Flow-DPPO
VLM RL
diffusion RL
multi-node training
复杂 TensorRef data plane
完整 agent SDK
```

这些可以作为未来方向，但不进入当前主线。

---

## 3. 总体系统逻辑

训练闭环如下：

```text
TaskPlugin
  ↓
PromptBuilder
  ↓
RolloutReq
  ↓
HF / vLLM Rollout Engine
  ↓
RolloutResp
  ↓
RewardPlugin
  ↓
MetricPlugin
  ↓
GRPO / DAPO-style Algorithm
  ↓
LoRA Trainer
  ↓
Adapter Sync
  ↓
Next Rollout
```

固定内核：

```text
GRPO / DAPO
LoRA Trainer
HF / vLLM Rollout
Adapter Sync
Logging
```

可插拔部分：

```text
Dataset
TaskBuilder
PromptBuilder
RewardPlugin
MetricPlugin
```

核心原则：

> 换能力时，只换 TaskPlugin / RewardPlugin / MetricPlugin，不改 GRPO/DAPO/LoRA 训练主循环。

---

## 4. 参考框架分工

| 参考框架     | `rlite` 学什么                                                 | 不学什么                          |
| -------- | ----------------------------------------------------------- | ----------------------------- |
| rLLM     | Task / Episode / Trajectory / Step，Agent / Env 解耦           | 不做完整 agent SDK                |
| UniRL    | RolloutReq / RolloutResp / RewardService / typed runtime 思想 | 不做多模态统一 RL，不做复杂 data plane    |
| EasyR1   | GRPO/DAPO recipe、LoRA 低资源训练配置                               | 不做 VLM 主线                     |
| verl     | GRPO/DAPO batch 组织和 loss 参考                                 | 不做 production-ready worker 系统 |
| OpenRLHF | Ray + vLLM + 训练/推理分离思想                                      | 不做完整 RLHF 系统                  |
| vLLM     | 高吞吐 rollout、adapter 加载/重载                                   | 不改 vLLM 内核                    |
| PEFT     | LoRA adapter 训练和保存                                          | 不做复杂 adapter zoo              |
| TRL      | 最小 GRPO trainer 设计思路                                        | 不直接封装成 TRL wrapper            |
| Ray      | 后期 actor 化和 colocate 编排                                     | v1 不做复杂多节点/异步                 |

---

## 5. 核心抽象

## 5.1 Task 层抽象

用于表达“训练什么能力”。

```python
@dataclass
class Task:
    task_id: str
    input: dict
    target: Any
    metadata: dict
```

示例：

数学任务：

```python
Task(
    task_id="gsm8k_001",
    input={"question": "Natalia sold clips..."},
    target={"answer": "72"},
    metadata={"source": "gsm8k"}
)
```

代码修复任务：

```python
Task(
    task_id="code_001",
    input={
        "buggy_code": "...",
        "tests": "..."
    },
    target={"expected": "all_tests_pass"},
    metadata={"source": "mbpp_bugfix"}
)
```

---

## 5.2 Trajectory 层抽象

用于表达模型对某个 task 的一次采样结果。

```python
@dataclass
class Step:
    prompt: str
    response: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    metadata: dict

@dataclass
class Trajectory:
    task_id: str
    steps: list[Step]
    final_response: str
    reward: float | None
    advantage: float | None
    metadata: dict
```

GSM8K 是 one-step trajectory：

```text
Task: 一道数学题
Step: 模型生成一次答案
Reward: 答案是否正确 + 格式奖励
Trajectory: 这次回答的完整记录
```

---

## 5.3 Rollout 层抽象

参考 UniRL 的 typed runtime 思想，但做最轻版本。

```python
@dataclass
class RolloutReq:
    batch_id: str
    tasks: list[Task]
    n_samples: int
    temperature: float
    top_p: float
    max_tokens: int
    policy_version: int
    metadata: dict

@dataclass
class RolloutResp:
    batch_id: str
    trajectories: list[Trajectory]
    policy_version: int
    metadata: dict
```

第一版不做复杂 `TensorRef`，直接返回可训练 batch 所需数据。

---

## 5.4 插件接口

### TaskPlugin

负责数据加载和 task 构造。

```python
class TaskPlugin:
    name: str

    def load_dataset(self, split: str) -> Iterable[Task]:
        ...

    def build_prompt(self, task: Task) -> str:
        ...
```

### RewardPlugin

负责给模型输出打分。

```python
class RewardPlugin:
    name: str

    def score(self, task: Task, trajectory: Trajectory) -> float:
        ...

    def validate(self, rewards: list[float]) -> None:
        ...
```

### MetricPlugin

负责评估能力。

```python
class MetricPlugin:
    name: str

    def compute(self, episodes: list[Trajectory]) -> dict:
        ...
```

### AlgorithmPlugin

v1 只支持：

```text
GRPO
GRPO + DAPO-style plugins
```

---

## 6. 支持的任务插件

## 6.1 MVP：GSM8K 数学推理插件

能力目标：

```text
数学推理
答案格式遵循
final answer extraction
```

插件组成：

```text
tasks/gsm8k/dataset.py
tasks/gsm8k/prompt.py
tasks/gsm8k/reward.py
tasks/gsm8k/metrics.py
```

奖励：

```text
exact match reward
format reward
invalid answer penalty
```

指标：

```text
exact_match
invalid_format_rate
avg_response_length
```

参考：

```text
EasyR1: math GRPO recipe
rLLM: task/env 组织
verl: batch + reward 数据流
```

---

## 6.2 Strong Version：JSON / 格式遵循插件

能力目标：

```text
结构化输出
schema following
格式遵循
```

插件组成：

```text
tasks/json_following/dataset.py
tasks/json_following/prompt.py
tasks/json_following/reward.py
tasks/json_following/metrics.py
```

奖励：

```text
JSON parse success
schema validation
required fields coverage
no extra text penalty
```

指标：

```text
valid_json_rate
schema_pass_rate
field_coverage
```

为什么适合：

```text
reward 明确
算力低
容易验证插件化是否成功
```

---

## 6.3 Advanced Version：代码修复插件

能力目标：

```text
代码生成
代码修复
单元测试驱动反馈
```

插件组成：

```text
tasks/code_repair/dataset.py
tasks/code_repair/prompt.py
tasks/code_repair/reward.py
tasks/code_repair/metrics.py
tasks/code_repair/sandbox.py
```

奖励：

```text
syntax valid reward
unit test pass ratio
timeout penalty
unsafe code penalty
```

指标：

```text
syntax_valid_rate
test_pass_rate
pass@1
avg_tests_passed
```

参考：

```text
rLLM: environment / tool execution 思想
AReaL: agentic rollout 思想，只借鉴概念，不实现 fully async
```

---

## 7. GRPO / DAPO 算法设计

## 7.1 GRPO 主算法

每个 prompt/task 采样 K 个 response：

```text
task_i
  → response_i1, reward_i1
  → response_i2, reward_i2
  → ...
  → response_iK, reward_iK
```

组内 advantage：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

训练目标：

```text
ratio = exp(new_logprob - old_logprob)

loss = -min(
  ratio * advantage,
  clip(ratio, 1 - eps, 1 + eps) * advantage
)
```

必须支持：

```text
group reward normalization
old logprob
new logprob
ratio clip
response mask
optional KL
invalid response penalty
nonzero advantage ratio
```

参考：

```text
verl: GRPO batch/loss 组织
EasyR1: GRPO recipe
TRL: 最小 trainer 思路
```

---

## 7.2 DAPO-style 插件

DAPO 不作为独立复杂算法，而是作为 GRPO 的插件集合。

只做四个：

```text
Dynamic Sampling
Clip-Higher
token-level loss
overlong penalty
```

配置：

```yaml
algo:
  name: grpo
  dapo:
    enabled: true
    dynamic_sampling: true
    clip_higher: true
    token_level_loss: true
    overlong_penalty: true
```

每个插件都必须能独立开关。

### Dynamic Sampling

过滤全对或全错 group，提高有效 advantage 比例。

指标：

```text
nonzero_advantage_ratio
filtered_group_ratio
```

### Clip-Higher

对正 advantage 样本使用更高 clip 上界。

指标：

```text
loss_curve
clip_fraction
```

### Token-level Loss

从 sequence-level 聚合切换到 token-level mask。

指标：

```text
token_loss
response_length
training_stability
```

### Overlong Penalty

控制过长输出。

指标：

```text
avg_response_length
overlong_rate
```

参考：

```text
DAPO: 组件思想
EasyR1: DAPO 配置风格
verl: 插件化算法组织
```

---

## 8. LoRA-first 训练策略

## 8.1 默认训练方式

默认冻结 base model，只训练 LoRA adapter：

```text
base model: frozen
LoRA adapter: trainable
RL loss: update adapter only
```

默认模型：

```text
Qwen2.5-1.5B-Instruct
Qwen2.5-Math-1.5B
Qwen2.5-Coder-1.5B，作为 code repair demo 可选
```

默认配置：

```yaml
trainer:
  method: lora
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

---

## 8.2 训练后端

MVP：

```text
transformers + PEFT + Accelerate
```

Strong Version：

```text
DeepSpeed ZeRO-2 + PEFT LoRA
```

不做：

```text
ZeRO-3 full-parameter RL
full weight gather
full model broadcast
```

---

## 8.3 Adapter Sync

第一版：

```text
train LoRA
→ save adapter checkpoint
→ rollout engine reload adapter
```

第二版：

```text
train LoRA
→ extract adapter state_dict
→ update rollout engine adapter
```

第三版：

```text
Ray RolloutActor / TrainerActor
→ adapter broadcast
→ rollout actor reload adapter
```

核心原则：

> 同步 adapter，而不是同步 full model。

---

## 9. Rollout 设计

## 9.1 HF Rollout

目的：

```text
debug
验证算法闭环
最小依赖
```

适合 Phase 早期。

接口：

```python
class HFRolloutEngine:
    def generate(self, req: RolloutReq) -> RolloutResp:
        ...
```

参考：

```text
transformers generate
TRL debug trainer 思想
```

---

## 9.2 vLLM Rollout

目的：

```text
提高生成吞吐
模拟现代 RLVR 系统中的 rollout engine
```

接口：

```python
class VLLMRolloutEngine:
    def generate(self, req: RolloutReq) -> RolloutResp:
        ...

    def reload_adapter(self, path: str) -> None:
        ...
```

优先实现：

```text
vLLM generate
LoRA adapter checkpoint reload
```

后续再做：

```text
adapter state_dict in-place update
```

参考：

```text
vLLM rollout
OpenRLHF 训练/推理分离思想
EasyR1 vLLM recipe
```

---

## 10. 推荐目录结构

```text
rlite/
  configs/
    debug.yaml
    gsm8k_grpo_lora_hf.yaml
    gsm8k_grpo_lora_vllm.yaml
    gsm8k_dapo_lora.yaml
    json_following_grpo_lora.yaml
    code_repair_grpo_lora.yaml
    ray_colocate_lora.yaml

  rlite/
    __init__.py

    config.py
    registry.py
    logging.py
    metrics.py

    core/
      types.py
      rollout_types.py
      batch.py

    plugins/
      base.py
      task.py
      reward.py
      metric.py

    tasks/
      gsm8k/
        dataset.py
        prompt.py
        reward.py
        metrics.py

      json_following/
        dataset.py
        prompt.py
        reward.py
        metrics.py

      code_repair/
        dataset.py
        prompt.py
        reward.py
        metrics.py
        sandbox.py

    rollout/
      base.py
      hf_engine.py
      vllm_engine.py

    algos/
      advantages.py
      losses.py
      grpo.py
      dapo.py

    trainers/
      base.py
      lora_trainer.py
      deepspeed_lora_trainer.py

    sync/
      adapter_checkpoint.py
      adapter_inplace.py

    ray/
      actors.py
      placement.py
      driver.py

    train.py
    eval.py

  tests/
    test_types.py
    test_registry.py
    test_gsm8k_reward.py
    test_json_reward.py
    test_grpo.py
    test_dapo.py
    test_lora_trainer.py
    test_adapter_sync.py

  scripts/
    run_gsm8k_grpo_lora_hf.sh
    run_gsm8k_grpo_lora_vllm.sh
    run_gsm8k_dapo_lora.sh
    run_json_following.sh
    run_code_repair.sh

  docs/
    architecture.md
    plugin_system.md
    grpo.md
    dapo.md
    lora_sync.md
    experiments.md

  README.md
  PLAN.md
  pyproject.toml
  requirements.txt
```

---

# Phase Plan

---

## Phase 0 — Scope Lock：收敛项目边界

### 目标

明确 `rlite` 只做：

```text
插件化任务系统
LoRA 低资源训练
GRPO
DAPO-style plugins
HF / vLLM rollout
adapter sync
```

### 不做

```text
PPO
critic
reward model
full-parameter RL
ZeRO-3 full sync
fully async
VLM
multi-node
```

### 参考

| 模块    | 参考                        |
| ----- | ------------------------- |
| 项目定位  | EasyR1 / rLLM / UniRL     |
| 不做项边界 | OpenRLHF / verl 的复杂能力作为反例 |

### 产出

```text
docs/scope.md
```

### 完成标准

项目 README 中明确写出：

> rlite is a plugin-based LoRA-GRPO/DAPO framework for verifiable LLM skills.

---

## Phase 1 — 脚手架 + 配置 + Registry

### 目标

建立项目骨架和插件注册机制。

### 实现模块

```text
config.py
registry.py
train.py
eval.py
configs/debug.yaml
```

### 要做

1. `pyproject.toml`；
2. YAML config；
3. dataclass config；
4. plugin registry；
5. train/eval CLI；
6. debug 空跑。

### 参考

| 模块            | 参考              |
| ------------- | --------------- |
| 配置系统          | EasyR1 recipe   |
| typed config  | UniRL recipe 思想 |
| train/eval 分离 | rLLM CLI 思想     |

### 配置示例

```yaml
task:
  name: gsm8k

reward:
  name: gsm8k_rule

rollout:
  engine: hf
  n_samples: 4

algo:
  name: grpo

trainer:
  method: lora
```

### 验证

```bash
python -m rlite.train --config configs/debug.yaml
python -m rlite.eval --config configs/debug.yaml
```

### 完成标准

1. 可以注册 task/reward/metric；
2. debug config 能跑通；
3. CLI 不报 import/config 错误。

---

## Phase 2 — 核心数据结构

### 目标

定义所有模块共享的数据 contract。

### 实现模块

```text
core/types.py
core/rollout_types.py
tests/test_types.py
```

### 要做

实现：

```text
Task
Step
Trajectory
RolloutReq
RolloutResp
RewardResult
MetricResult
```

### 参考

| 抽象                       | 参考                     |
| ------------------------ | ---------------------- |
| Task / Step / Trajectory | rLLM                   |
| RolloutReq / RolloutResp | UniRL                  |
| RewardResult             | UniRL RewardService 思想 |

### 完成标准

1. raw dataset sample 能转成 Task；
2. generated response 能转成 Trajectory；
3. RolloutResp 能被 RewardPlugin 接收；
4. Rewarded trajectories 能转成 GRPO batch；
5. 单测通过。

---

## Phase 3 — Plugin System：Task / Reward / Metric

### 目标

把任务、奖励、评估都做成可插拔组件。

### 实现模块

```text
plugins/base.py
plugins/task.py
plugins/reward.py
plugins/metric.py
registry.py
tests/test_registry.py
```

### 接口

```python
class TaskPlugin:
    def load_dataset(self, split: str) -> Iterable[Task]:
        ...

    def build_prompt(self, task: Task) -> str:
        ...

class RewardPlugin:
    def score(self, task: Task, trajectory: Trajectory) -> RewardResult:
        ...

class MetricPlugin:
    def compute(self, trajectories: list[Trajectory]) -> dict:
        ...
```

### 参考

| 模块             | 参考             |
| -------------- | -------------- |
| Agent / Env 解耦 | rLLM           |
| RewardService  | UniRL          |
| recipe 选择插件    | EasyR1 / UniRL |

### 完成标准

1. 能通过配置选择 task/reward/metric；
2. 训练主循环不 import 具体任务；
3. 新任务只需要新建 `tasks/<name>/`。

---

## Phase 4 — GSM8K 插件：第一个可验证能力

### 目标

实现第一个完整 TaskPlugin，用于验证数学推理能力。

### 实现模块

```text
tasks/gsm8k/dataset.py
tasks/gsm8k/prompt.py
tasks/gsm8k/reward.py
tasks/gsm8k/metrics.py
tests/test_gsm8k_reward.py
```

### 奖励

```text
exact match reward
format reward
invalid answer penalty
```

### 答案抽取支持

```text
#### answer
\boxed{}
最后一个数字
负数
小数
千分位
```

### 指标

```text
exact_match
invalid_format_rate
avg_response_length
```

### 参考

| 模块              | 参考     |
| --------------- | ------ |
| math reward     | EasyR1 |
| task/env 结构     | rLLM   |
| reward validate | UniRL  |

### 验证

```bash
pytest tests/test_gsm8k_reward.py
python -m rlite.eval --config configs/gsm8k_eval.yaml
```

### 完成标准

1. GSM8K eval-only 跑通；
2. reward 解析单测全通过；
3. base model 有 baseline 指标。

---

## Phase 5 — GRPO 纯函数核心

### 目标

先把算法写对，不碰模型训练。

### 实现模块

```text
algos/advantages.py
algos/losses.py
algos/grpo.py
tests/test_grpo.py
```

### 要做

1. group reward normalization；
2. advantage 计算；
3. old/new logprob ratio；
4. clipped objective；
5. response mask；
6. optional KL；
7. all-correct/all-wrong group 防 NaN；
8. nonzero advantage ratio 统计。

### 参考

| 模块         | 参考            |
| ---------- | ------------- |
| GRPO loss  | verl / EasyR1 |
| trainer 思路 | TRL           |
| batch 结构   | verl          |

### 验证

```bash
pytest tests/test_grpo.py
```

### 完成标准

1. 手算小例子对齐；
2. loss 数值稳定；
3. 极端 reward group 不 NaN；
4. 输出 metrics 包含 nonzero advantage ratio。

---

## Phase 6 — HF LoRA-GRPO：最小训练闭环

### 目标

不用 vLLM，不用 Ray，先用 HF generate 跑通 LoRA-GRPO。

### 实现模块

```text
rollout/hf_engine.py
trainers/lora_trainer.py
train.py
scripts/run_gsm8k_grpo_lora_hf.sh
```

### 训练闭环

```text
sample tasks
→ build prompts
→ HF generate K responses
→ RewardPlugin score
→ GRPO advantage
→ LoRA train_step
→ eval
→ log
```

### 参考

| 模块          | 参考            |
| ----------- | ------------- |
| LoRA 训练     | PEFT          |
| 最小 RL loop  | TRL           |
| GRPO recipe | EasyR1 / verl |

### 必须记录

```text
train/reward_mean
train/loss
eval/exact_match
train/invalid_format_rate
train/nonzero_advantage_ratio
system/train_step_time
```

### 验证

```bash
bash scripts/run_gsm8k_grpo_lora_hf.sh
```

### 完成标准

1. one-batch overfit 成功；
2. GSM8K subset 上 reward 有上升趋势；
3. LoRA checkpoint 可保存/恢复；
4. 训练主循环不依赖具体 GSM8K 代码，只依赖插件接口。

这是第一个核心里程碑。

---

## Phase 7 — vLLM Rollout + LoRA Adapter Reload

### 目标

把 rollout 从 HF generate 换成 vLLM，提高生成吞吐。

### 实现模块

```text
rollout/vllm_engine.py
sync/adapter_checkpoint.py
configs/gsm8k_grpo_lora_vllm.yaml
scripts/run_gsm8k_grpo_lora_vllm.sh
```

### 训练闭环

```text
vLLM generate
→ RewardPlugin score
→ GRPO loss
→ update LoRA
→ save adapter checkpoint
→ vLLM reload adapter
→ next rollout
```

### 参考

| 模块                 | 参考       |
| ------------------ | -------- |
| vLLM rollout       | vLLM     |
| 训练/推理分离            | OpenRLHF |
| LoRA + vLLM recipe | EasyR1   |

### 必须记录

```text
system/rollout_tokens_per_sec
system/samples_per_sec
system/adapter_reload_time
eval/exact_match
train/reward_mean
```

### 完成标准

1. vLLM rollout 跑通；
2. adapter reload 后模型行为发生变化；
3. reward 曲线趋势与 HF 版本一致；
4. rollout throughput 高于 HF generate。

这是第二个核心里程碑。

---

## Phase 8 — DAPO-style Plugins

### 目标

在 GRPO 上加入 DAPO-style 训练稳定性插件。

### 实现模块

```text
algos/dapo.py
configs/gsm8k_dapo_lora.yaml
tests/test_dapo.py
scripts/run_gsm8k_dapo_lora.sh
```

### 插件

```text
Dynamic Sampling
Clip-Higher
token-level loss
overlong penalty
```

### 配置

```yaml
algo:
  name: grpo
  dapo:
    enabled: true
    dynamic_sampling:
      enabled: true
    clip_higher:
      enabled: true
      eps_low: 0.2
      eps_high: 0.28
    token_level_loss:
      enabled: true
    overlong_penalty:
      enabled: true
      max_len: 1024
      penalty: -0.2
```

### 参考

| 模块      | 参考     |
| ------- | ------ |
| DAPO 组件 | DAPO   |
| 配置风格    | EasyR1 |
| 算法组织    | verl   |

### 完成标准

1. 每个插件可单独开关；
2. Dynamic Sampling 能提高 nonzero advantage ratio；
3. overlong penalty 能降低 overlong rate；
4. 有 GRPO vs DAPO-style ablation。

这是第三个核心里程碑。

---

## Phase 9 — 第二任务插件：JSON / 格式遵循

### 目标

证明 `rlite` 不是 GSM8K 专用脚本，而是可插拔训练系统。

### 实现模块

```text
tasks/json_following/dataset.py
tasks/json_following/prompt.py
tasks/json_following/reward.py
tasks/json_following/metrics.py
configs/json_following_grpo_lora.yaml
scripts/run_json_following.sh
```

### 奖励

```text
JSON parse success
schema validation
required fields coverage
no extra text penalty
```

### 指标

```text
valid_json_rate
schema_pass_rate
field_coverage
```

### 参考

| 模块               | 参考     |
| ---------------- | ------ |
| plugin task 设计   | rLLM   |
| RewardService 思想 | UniRL  |
| 低资源 recipe       | EasyR1 |

### 完成标准

1. 不改训练主循环；
2. 只新增 task/reward/metric 插件；
3. LoRA-GRPO 可以训练 JSON/schema following；
4. valid_json_rate 或 schema_pass_rate 有提升。

这是证明“即插即用”的关键里程碑。

---

## Phase 10 — Adapter In-place Sync

### 目标

优化 Phase 7 的 adapter checkpoint reload，尝试直接同步 adapter state_dict。

### 实现模块

```text
sync/adapter_inplace.py
tests/test_adapter_sync.py
docs/lora_sync.md
```

### 同步路径

```text
LoRA Trainer
→ extract adapter state_dict
→ send to rollout engine
→ update / reload adapter
→ next rollout
```

### 参考

| 模块                            | 参考       |
| ----------------------------- | -------- |
| weight sync 思想                | OpenRLHF |
| sync 模块化                      | UniRL    |
| adapter state_dict            | PEFT     |
| rollout engine adapter reload | vLLM     |

### 完成标准

1. 不保存 full model；
2. 只同步 adapter；
3. adapter sync time 低于 checkpoint reload；
4. 同步失败时 fallback 到 checkpoint reload。

这是系统亮点里程碑。

---

## Phase 11 — Ray Colocate：可选工程化

### 目标

把 rollout 和 trainer 分离为 actor，展示工程编排能力。

### 实现模块

```text
ray/actors.py
ray/placement.py
ray/driver.py
configs/ray_colocate_lora.yaml
```

### Actor 设计

```text
RolloutActor:
  - vLLM generate
  - reload/update LoRA adapter

TrainerActor:
  - LoRA train_step
  - save/extract adapter

Driver:
  - sample task
  - call rollout
  - call reward
  - call algo
  - call trainer
  - call adapter sync
  - log metrics
```

### 参考

| 模块                | 参考       |
| ----------------- | -------- |
| Ray actor 编排      | OpenRLHF |
| placement group   | Ray      |
| driver/runtime 分离 | UniRL    |

### 完成标准

1. Ray colocate 跑通；
2. 与非 Ray 版本 reward 趋势一致；
3. 核心算法代码不依赖 Ray；
4. README 里有 actor 架构图。

这是高级加分项，不是 MVP 必须项。

---

## Phase 12 — 第三任务插件：代码修复

### 目标

加入一个更像真实工程场景的 verifiable skill。

### 实现模块

```text
tasks/code_repair/dataset.py
tasks/code_repair/prompt.py
tasks/code_repair/reward.py
tasks/code_repair/metrics.py
tasks/code_repair/sandbox.py
configs/code_repair_grpo_lora.yaml
scripts/run_code_repair.sh
```

### 奖励

```text
syntax valid reward
unit test pass ratio
timeout penalty
unsafe code penalty
```

### 指标

```text
syntax_valid_rate
test_pass_rate
pass@1
avg_tests_passed
```

### 参考

| 模块                  | 参考          |
| ------------------- | ----------- |
| environment/task 设计 | rLLM        |
| agentic RL 思想       | AReaL，仅借鉴概念 |
| reward service      | UniRL       |

### 完成标准

1. 不改训练主循环；
2. 只新增 code_repair 插件；
3. test_pass_rate 有提升；
4. 可以作为 README 第二/第三 demo。

---

## Phase 13 — 文档、实验、简历包装

### 目标

整理成可以展示的简历项目。

### README 必须包含

1. 项目定位；
2. 为什么 LoRA-first；
3. 为什么 GRPO/DAPO-only；
4. 插件系统说明；
5. 架构图；
6. 一行启动命令；
7. GSM8K reward 曲线；
8. JSON/schema following 插件证明；
9. vLLM vs HF rollout throughput；
10. adapter checkpoint reload vs adapter sync 对比；
11. 和 rLLM / UniRL / EasyR1 / OpenRLHF 的差异。

### 实验表

| 实验                                | 目的               |
| --------------------------------- | ---------------- |
| HF LoRA-GRPO on GSM8K             | 验证最小 RL 闭环       |
| vLLM LoRA-GRPO on GSM8K           | 验证 rollout 加速    |
| LoRA-GRPO vs LoRA-DAPO            | 验证 DAPO-style 插件 |
| GSM8K vs JSON plugin              | 验证任务插件化          |
| checkpoint reload vs adapter sync | 验证系统优化           |
| non-Ray vs Ray colocate           | 验证 actor 化，选做    |
| code repair demo                  | 验证真实工程任务，选做      |

### 指标

```text
train/reward_mean
train/reward_std
eval/exact_match
eval/valid_json_rate
eval/schema_pass_rate
eval/test_pass_rate
train/loss
train/kl
train/response_length
train/invalid_format_rate
train/nonzero_advantage_ratio
train/filtered_group_ratio
system/rollout_tokens_per_sec
system/samples_per_sec
system/train_step_time
system/adapter_reload_time
system/adapter_sync_time
system/gpu_memory_peak
```

---

# Milestones

## MVP：必须完成

```text
Phase 0: scope lock
Phase 1: scaffold + config + registry
Phase 2: core data structures
Phase 3: plugin system
Phase 4: GSM8K plugin
Phase 5: GRPO pure function
Phase 6: HF LoRA-GRPO
```

完成 MVP 后，项目已经可以证明：

```text
我实现了一个插件化 LoRA-GRPO RL 后训练闭环。
```

---

## Strong Version：建议完成

```text
Phase 7: vLLM rollout + adapter reload
Phase 8: DAPO-style plugins
Phase 9: JSON/schema following plugin
Phase 10: adapter in-place sync
```

完成 Strong Version 后，项目已经适合写进简历。

---

## Advanced Version：加分项

```text
Phase 11: Ray colocate
Phase 12: code repair plugin
Phase 13: docs + benchmark
```

完成 Advanced Version 后，项目会变成一个比较完整的系统型后训练项目。

---

# 最小可交付版本

如果时间有限，最小可交付版本是：

```text
Plugin registry
+ GSM8K TaskPlugin
+ RewardPlugin
+ GRPO pure function
+ PEFT LoRA Trainer
+ HF rollout
+ reward/eval 曲线
+ README
```

这已经可以作为简历项目雏形。

---

# 最推荐的最终项目标题

中文：

> rlite：面向可验证能力训练的插件化 LoRA-GRPO/DAPO 后训练框架

英文：

> rlite: A Plugin-based LoRA-GRPO/DAPO Framework for Verifiable LLM Skill Training

---

# 简历描述草稿

`rlite`: Designed and implemented a plugin-based LoRA-first RL post-training framework for verifiable LLM skills. The framework supports pluggable Task/Reward/Metric modules, GRPO and DAPO-style training, PEFT LoRA optimization, HF/vLLM rollout backends, adapter-level synchronization, and optional Ray-based colocated actor orchestration. Built reproducible low-resource experiments on Qwen2.5-1.5B for GSM8K reasoning and JSON/schema-following tasks, tracking reward, exact match, valid-output rate, nonzero advantage ratio, rollout throughput, memory usage, and adapter sync overhead.

---

# 核心卖点

1. **LoRA-first**：默认低资源训练，而不是 full-parameter RL。
2. **GRPO/DAPO-only**：算法主线清晰，不做大杂烩。
3. **Plugin-based**：换能力只换 Task/Reward/Metric，不改训练内核。
4. **Verifiable skills**：优先支持可自动打分的能力训练。
5. **vLLM rollout**：支持现代 RLVR 的高吞吐采样路径。
6. **Adapter sync**：系统亮点集中在 LoRA adapter 同步，而不是高风险 full model sync。
7. **Readable**：代码结构清晰，适合简历展示和开源。
