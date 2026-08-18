"""Create publication-ready loss, component, stability, memory, and speed figures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.schema import DataContractError
from scripts.batch_lr_experiment import (
    BASELINE_GROUP,
    COMPONENT_FIELDS,
    GROUPS,
    latest_completed_run,
    read_json,
    read_metrics,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "runs" / "batch_lr_optimization"
COLORS = {
    2: "#2563EB",
    4: "#CA8A04",
    8: "#EA580C",
    16: "#6B7F2A",
    32: "#C24175",
}
LINESTYLES = {5e-6: ":", 1e-5: "-", 2e-5: "--"}
MARKERS = {5e-6: "D", 1e-5: "o", 2e-5: "s"}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#111827",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _ema(values: Iterable[float], alpha: float = 0.4) -> list[float]:
    output: list[float] = []
    for value in values:
        output.append(float(value) if not output else alpha * float(value) + (1 - alpha) * output[-1])
    return output


def _load_seed42_runs(
    root: Path, safe_groups: list[str], shortlist: list[str], init_sha256: str
) -> dict[str, tuple[Path, list[dict[str, float]]]]:
    runs: dict[str, tuple[Path, list[dict[str, float]]]] = {}
    for group in safe_groups:
        epochs = 15 if group == BASELINE_GROUP or group in shortlist else 5
        run = latest_completed_run(
            root,
            group,
            42,
            epochs,
            initialization_sha256=init_sha256,
        )
        runs[group] = (run, read_metrics(run))
    return runs


def _series_style(group: str) -> dict[str, Any]:
    spec = GROUPS[group]
    return {
        "color": COLORS[spec.batch_size],
        "linestyle": LINESTYLES[spec.learning_rate],
        "marker": MARKERS[spec.learning_rate],
        "linewidth": 1.8 if group != BASELINE_GROUP else 2.5,
        "markersize": 4.0,
        "markevery": 1,
    }


def _plot_all_groups(
    root: Path, runs: dict[str, tuple[Path, list[dict[str, float]]]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharex=True)
    for axis, field, title in zip(
        axes,
        ("train_loss", "val_loss"),
        ("Training objective loss", "Validation objective loss"),
        strict=True,
    ):
        endings: list[tuple[float, str]] = []
        for group, (_, rows) in runs.items():
            epochs = [int(row["epoch"]) + 1 for row in rows]
            values = [row[field] for row in rows]
            axis.plot(epochs, values, label=group, **_series_style(group))
            endings.append((values[-1], group))
        for offset_index, (_, group) in enumerate(sorted(endings)):
            rows = runs[group][1]
            axis.annotate(
                group,
                xy=(int(rows[-1]["epoch"]) + 1, rows[-1][field]),
                xytext=(7, (offset_index - len(endings) / 2) * 2.0),
                textcoords="offset points",
                color=COLORS[GROUPS[group].batch_size],
                fontsize=8,
                fontweight="bold" if group == BASELINE_GROUP else "normal",
            )
        axis.axvline(5, color="#64748B", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Completed epoch")
        axis.set_ylabel("W2 objective loss")
        axis.grid(True, axis="y")
        axis.set_xlim(left=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Batch size and learning-rate loss curves", fontsize=15, fontweight="bold", y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=min(9, len(labels)),
        frameon=False,
    )
    fig.text(
        0.5,
        -0.01,
        "Raw epoch means; vertical line marks the five-epoch screening boundary. Color = batch size, line style = learning rate.",
        ha="center",
        color="#475569",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    _save(fig, output)


def _plot_small_multiples(
    runs: dict[str, tuple[Path, list[dict[str, float]]]],
    preflight: dict[str, Any],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=False)
    for axis, group in zip(axes.flat, GROUPS, strict=True):
        spec = GROUPS[group]
        if group not in runs:
            status = preflight["groups"][group]
            peak = status.get("peak_gpu_memory_bytes")
            peak_text = f"{peak / 2**30:.2f} GiB allocated" if peak else "no peak available"
            axis.text(
                0.5,
                0.55,
                f"{status['status'].upper()}\n{peak_text}",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#9F1239",
                fontsize=12,
                fontweight="bold",
            )
            axis.text(
                0.5,
                0.32,
                status.get("reason", "no loss curve"),
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#475569",
                fontsize=8,
                wrap=True,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            rows = runs[group][1]
            epochs = [int(row["epoch"]) + 1 for row in rows]
            axis.plot(epochs, [row["train_loss"] for row in rows], color="#2563EB", marker="o", label="train")
            axis.plot(epochs, [row["val_loss"] for row in rows], color="#EA580C", marker="s", linestyle="--", label="validation")
            axis.grid(True, axis="y")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("W2 loss")
            axis.legend(frameon=False, fontsize=8)
        axis.set_title(f"{group}: batch {spec.batch_size}, lr {spec.learning_rate:.0e}")
    fig.suptitle("Per-group training and validation loss", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01, "Shared metric definition; panels use individual y ranges to expose within-group variation.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    _save(fig, output)


def _plot_components(
    runs: dict[str, tuple[Path, list[dict[str, float]]]], output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for axis, component in zip(axes.flat, COMPONENT_FIELDS, strict=True):
        field = f"val_{component}"
        for group, (_, rows) in runs.items():
            axis.plot(
                [int(row["epoch"]) + 1 for row in rows],
                [row[field] for row in rows],
                label=group,
                **_series_style(group),
            )
        axis.set_title(component.replace("_", " ").title())
        axis.set_xlabel("Completed epoch")
        axis.set_ylabel("Raw component loss")
        axis.grid(True, axis="y")
        axis.axvline(5, color="#64748B", linestyle=":", linewidth=1.0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Validation loss components", fontsize=16, fontweight="bold", y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=min(9, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, output)


def _available_seed_runs(
    root: Path, group: str, init_sha256: str
) -> dict[int, list[dict[str, float]]]:
    outputs: dict[int, list[dict[str, float]]] = {}
    for seed in (42, 43, 44):
        try:
            run = latest_completed_run(
                root,
                group,
                seed,
                15,
                initialization_sha256=init_sha256,
            )
            outputs[seed] = read_metrics(run)
        except DataContractError:
            continue
    return outputs


def _plot_best_vs_baseline(
    root: Path,
    runs: dict[str, tuple[Path, list[dict[str, float]]]],
    winner: str,
    accepted: bool,
    init_sha256: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True)
    baseline = runs[BASELINE_GROUP][1]
    winner_runs = _available_seed_runs(root, winner, init_sha256)
    for axis, field, title in zip(
        axes,
        ("train_loss", "val_loss"),
        ("Training loss: candidate vs baseline", "Validation loss: candidate vs baseline"),
        strict=True,
    ):
        epochs = np.array([int(row["epoch"]) + 1 for row in baseline])
        baseline_values = np.array([row[field] for row in baseline])
        axis.plot(epochs, baseline_values, color="#334155", marker="o", linewidth=1.6, alpha=0.7, label="G0 raw")
        axis.plot(epochs, _ema(baseline_values), color="#0F172A", linewidth=2.7, label="G0 EMA")
        candidate_arrays = []
        for index, (seed, rows) in enumerate(sorted(winner_runs.items())):
            values = np.array([row[field] for row in rows])
            candidate_arrays.append(values)
            axis.plot(
                epochs,
                values,
                color="#EA580C",
                linewidth=1.0,
                alpha=0.32,
                label=f"{winner} raw seeds" if index == 0 else None,
            )
        matrix = np.vstack(candidate_arrays)
        mean = matrix.mean(axis=0)
        if matrix.shape[0] > 1:
            axis.fill_between(epochs, matrix.min(axis=0), matrix.max(axis=0), color="#F97316", alpha=0.14, label=f"{winner} seed range")
        axis.plot(epochs, _ema(mean), color="#C2410C", linewidth=2.8, linestyle="--", label=f"{winner} mean EMA")
        axis.set_title(title)
        axis.set_xlabel("Completed epoch")
        axis.set_ylabel("W2 objective loss")
        axis.grid(True, axis="y")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    outcome = "accepted" if accepted else "rejected by quality guardrails"
    fig.suptitle(
        f"Best curve candidate ({winner}, {outcome}) versus retained baseline (G0)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=min(5, len(unique)),
        frameon=False,
    )
    fig.text(0.5, -0.01, "Thin lines are raw values; EMA is display-only and never used for ranking.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 0.79))
    _save(fig, output)


def _plot_memory_throughput(
    runs: dict[str, tuple[Path, list[dict[str, float]]]],
    preflight: dict[str, Any],
    output: Path,
) -> None:
    groups = list(GROUPS)
    memory: list[float] = []
    throughput: list[float] = []
    statuses: list[str] = []
    for group in groups:
        statuses.append(str(preflight["groups"][group]["status"]))
        if group in runs:
            rows = runs[group][1]
            values = [row.get("peak_gpu_memory_reserved_bytes", row.get("peak_gpu_memory_bytes", 0.0)) for row in rows]
            memory.append(max(values) / 2**30)
            speeds = [row.get("samples_per_second", math.nan) for row in rows]
            finite_speeds = [value for value in speeds if math.isfinite(value)]
            throughput.append(float(np.median(finite_speeds)) if finite_speeds else 0.0)
        else:
            memory.append(float(preflight["groups"][group].get("peak_gpu_memory_reserved_bytes") or 0) / 2**30)
            throughput.append(0.0)
    colors = [COLORS[GROUPS[group].batch_size] for group in groups]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = np.arange(len(groups))
    axes[0].barh(y, memory, color=colors, edgecolor="#334155")
    axes[0].set_yticks(y, groups)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Peak reserved memory (GiB)")
    axes[0].set_title("GPU memory by configuration")
    axes[0].grid(True, axis="x")
    for index, value in enumerate(memory):
        axes[0].text(value + 0.05, index, f"{value:.2f} GiB · {statuses[index]}", va="center", fontsize=8)
    axes[1].barh(y, throughput, color=colors, edgecolor="#334155")
    axes[1].set_yticks(y, groups)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Median training samples / second")
    axes[1].set_title("Training throughput")
    axes[1].grid(True, axis="x")
    for index, value in enumerate(throughput):
        axes[1].text(value + max(throughput or [1]) * 0.01, index, f"{value:.1f}" if value else "no training", va="center", fontsize=8)
    fig.suptitle("Memory safety and throughput", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    args = parser.parse_args()
    _configure_style()
    root = args.experiment_root.resolve()
    preflight = read_json(root / "preflight.json")
    training_state = read_json(root / "training_complete.json")
    quality = read_json(root / "quality_decision.json")
    init_sha256 = str(training_state["initialization_sha256"])
    safe_groups = [str(group) for group in training_state["safe_groups"]]
    shortlist = [str(group) for group in training_state["shortlist"]]
    runs = _load_seed42_runs(root, safe_groups, shortlist, init_sha256)
    automatic_winner = quality.get("automatic_winner")
    accepted = automatic_winner is not None
    if accepted:
        comparison_group = str(automatic_winner)
    else:
        curve_candidates = [
            (float(details["curve"]["combined_score"]), str(group))
            for group, details in quality.get("candidates", {}).items()
            if details.get("curve") and details["curve"].get("eligible_curve")
        ]
        comparison_group = (
            min(curve_candidates)[1] if curve_candidates else BASELINE_GROUP
        )
    output = root / "figures"
    _plot_all_groups(root, runs, output / "all_groups_total_loss.png")
    _plot_small_multiples(runs, preflight, output / "all_groups_small_multiples.png")
    _plot_components(runs, output / "component_loss_grid.png")
    _plot_best_vs_baseline(
        root,
        runs,
        comparison_group,
        accepted,
        init_sha256,
        output / "best_vs_baseline.png",
    )
    _plot_memory_throughput(runs, preflight, output / "memory_throughput_comparison.png")
    write_json(
        output / "chart_map.json",
        {
            "all_groups_total_loss": {
                "question": "How do raw train and validation losses evolve across configurations?",
                "family": "trend / highlighted multi-series line",
                "fields": ["epoch", "train_loss", "val_loss", "group"],
                "palette": "relaxed multi-category; color=batch, line style=learning rate",
            },
            "all_groups_small_multiples": {
                "question": "Is each individual curve stable without line overlap?",
                "family": "trend / small multiples",
                "fields": ["epoch", "train_loss", "val_loss", "group"],
            },
            "component_loss_grid": {
                "question": "Which W2 loss component drives each validation trajectory?",
                "family": "trend / component facets",
                "fields": ["epoch", *[f"val_{field}" for field in COMPONENT_FIELDS]],
            },
            "best_vs_baseline": {
                "question": "How does the accepted winner, or otherwise the best rejected curve candidate, compare with G0 and vary by seed?",
                "family": "uncertainty and benchmark / raw lines plus range band",
                "fields": ["epoch", "train_loss", "val_loss", "seed"],
            },
            "memory_throughput_comparison": {
                "question": "What resource and speed trade-off accompanies each group?",
                "family": "comparison / horizontal bars",
                "fields": ["group", "peak_reserved_gib", "samples_per_second"],
            },
        },
    )
    print(f"figures={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
