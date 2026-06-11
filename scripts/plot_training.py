"""Plot training curves from rlite training log."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Paths
LOG_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/train.log")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parse log
steps: list[int] = []
losses: list[float] = []
rewards: list[float] = []
nonzero_advs: list[float] = []
kls: list[float] = []

eval_steps: list[int] = []
exact_matches: list[float] = []
invalid_rates: list[float] = []
avg_lens: list[int] = []

with open(LOG_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        # [step    1/100] loss= 0.0000 | reward= 0.562 | nonzero_adv=0.750 | kl=0.0000 | time= 14.9s
        m = re.search(r"\[step\s+(\d+)/\d+\].*?loss=\s*([-\d.]+).*?reward=\s*([-\d.]+).*?nonzero_adv=\s*([-\d.]+).*?kl=\s*([-\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            rewards.append(float(m.group(3)))
            nonzero_advs.append(float(m.group(4)))
            kls.append(float(m.group(5)))

        # [eval    25] exact_match=0.300 | invalid_rate=0.000 | avg_len=856
        m = re.search(r"\[eval\s+(\d+)\].*?exact_match=([-\d.]+).*?invalid_rate=([-\d.]+).*?avg_len=(\d+)", line)
        if m:
            eval_steps.append(int(m.group(1)))
            exact_matches.append(float(m.group(2)))
            invalid_rates.append(float(m.group(3)))
            avg_lens.append(int(m.group(4)))

# Helper for dual y-axis plots
def dual_plot(x1, y1, label1, x2, y2, label2, title, xlabel, ylabel1, ylabel2, filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x1, y1, "b-o", markersize=4, label=label1)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel1, color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper left")

    if x2 and y2:
        ax2 = ax1.twinx()
        ax2.plot(x2, y2, "r-s", markersize=4, label=label2)
        ax2.set_ylabel(ylabel2, color="r", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.legend(loc="upper right")

    ax1.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")

# ------------------------------------------------------------------
# Plot 1: Loss and Reward per step
# ------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 5))
color1 = "tab:blue"
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Loss", color=color1, fontsize=12)
ax1.plot(steps, losses, "-o", color=color1, markersize=3, label="Loss")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
color2 = "tab:orange"
ax2.set_ylabel("Mean Reward", color=color2, fontsize=12)
ax2.plot(steps, rewards, "-s", color=color2, markersize=3, label="Mean Reward")
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax1.set_title("GSM8K LoRA-GRPO Training: Loss & Reward", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_reward.png", dpi=150)
plt.close()
print(f"Saved loss_reward.png")

# ------------------------------------------------------------------
# Plot 2: Exact Match over eval steps
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(eval_steps, exact_matches, "g-o", markersize=6, linewidth=2)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Exact Match Rate", fontsize=12)
ax.set_title("GSM8K LoRA-GRPO: Evaluation Exact Match Rate", fontsize=14)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
for x, y in zip(eval_steps, exact_matches):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "exact_match.png", dpi=150)
plt.close()
print(f"Saved exact_match.png")

# ------------------------------------------------------------------
# Plot 3: Nonzero Advantage & KL
# ------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(steps, nonzero_advs, "b-o", markersize=3, label="Nonzero Advantage Ratio")
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Nonzero Advantage Ratio", color="b")
ax1.tick_params(axis="y", labelcolor="b")
ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="50%")

ax2 = ax1.twinx()
ax2.plot(steps, kls, "r-s", markersize=3, label="KL Divergence")
ax2.set_ylabel("KL Divergence", color="r")
ax2.tick_params(axis="y", labelcolor="r")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.set_title("GSM8K LoRA-GRPO: Advantage Ratio & KL", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "advantage_kl.png", dpi=150)
plt.close()
print(f"Saved advantage_kl.png")

# ------------------------------------------------------------------
# Plot 4: Combined dashboard (2x2)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(steps, losses, "tab:blue", alpha=0.8)
axes[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.3)
axes[0, 0].set_title("Policy Loss")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Loss")

# Reward
axes[0, 1].plot(steps, rewards, "tab:orange", alpha=0.8)
axes[0, 1].set_title("Mean Reward")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Reward")

# Exact Match
axes[1, 0].plot(eval_steps, exact_matches, "g-o", markersize=4)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_title("Exact Match Rate")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Rate")
axes[1, 0].grid(True, alpha=0.3)

# Nonzero Advantage
axes[1, 1].plot(steps, nonzero_advs, "tab:purple", alpha=0.8)
axes[1, 1].set_title("Nonzero Advantage Ratio")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Ratio")
axes[1, 1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

fig.suptitle("GSM8K LoRA-GRPO Training Dashboard", fontsize=16)
fig.tight_layout()
fig.savefig(OUT_DIR / "dashboard.png", dpi=150)
plt.close()
print(f"Saved dashboard.png")

# ------------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------------
print("\n=== Training Summary ===")
print(f"Steps recorded: {len(steps)}")
print(f"Eval points: {len(eval_steps)}")
if exact_matches:
    print(f"Final exact_match: {exact_matches[-1]:.3f}")
    print(f"Best exact_match:  {max(exact_matches):.3f} at step {eval_steps[exact_matches.index(max(exact_matches))]}")
if losses:
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Final reward: {rewards[-1]:.3f}")
print("Done!")
