"""Create reproducible comparison figures from an rlite experiment CSV."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"GRPO": "#2563eb", "DAPO": "#ea580c", "GSPO": "#16a34a"}


def read_rows(path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            series[row["algorithm"]].append(
                (int(row["step"]), float(row["exact_match"]))
            )
    return {name: sorted(values) for name, values in series.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiments/gsm8k_seed42.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = read_rows(args.input)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, values in series.items():
        x, y = zip(*values)
        ax.plot(x, y, marker="o", markersize=4, linewidth=2,
                color=COLORS.get(name), label=name)
        best_step, best_value = max(values, key=lambda item: item[1])
        ax.scatter([best_step], [best_value], s=75, color=COLORS.get(name), zorder=3)
    ax.set(xlabel="Training step", ylabel="Exact match", title="GSM8K held-out evaluation")
    ax.set_xlim(left=0)
    ax.set_ylim(0.55, 0.75)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "algorithm_comparison.png", dpi=200)
    plt.close(fig)

    names, baseline, best, final = [], [], [], []
    for name, values in series.items():
        names.append(name)
        baseline.append(values[0][1])
        best.append(max(value for _, value in values))
        final.append(values[-1][1])
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    width = 0.24
    ax.bar([i - width for i in x], baseline, width, label="Baseline", color="#94a3b8")
    ax.bar(x, best, width, label="Best", color="#0f766e")
    ax.bar([i + width for i in x], final, width, label="Last reported", color="#7c3aed")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0.5, 0.75)
    ax.set_ylabel("Exact match")
    ax.set_title("Baseline, best and last reported checkpoint")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "result_summary.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
