from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (
    ExperimentSpec,
    PLOTS_DIR,
    aggregated_results,
    block_time_series,
    configure_matplotlib_env,
    ensure_directories,
    load_results,
    run_instrumented_simulation,
)

configure_matplotlib_env()

import matplotlib.pyplot as plt
import pandas as pd


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
        }
    )


def _strategy_label(strategy_name: str) -> str:
    return {
        "honest_baseline": "Honest baseline",
        "adaptive_mixing": "Adaptive mixing",
        "forking": "Forking",
    }[strategy_name]


def _delay_label(delay: int) -> str:
    return "VDF off" if delay == 0 else f"VDF delay={delay}"


def _save(fig: plt.Figure, name: str) -> None:
    ensure_directories()
    path = Path(PLOTS_DIR) / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reward_vs_alpha(df: pd.DataFrame) -> None:
    summary = aggregated_results(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"honest_baseline": "#4c78a8", "adaptive_mixing": "#e45756", "forking": "#54a24b"}

    # markers = {0: "o", 32: "s"}

    for strategy_name in summary["strategy"].unique():
        for delay in sorted(summary["vdf_delay"].unique()):
            subset = summary[(summary["strategy"] == strategy_name) & (summary["vdf_delay"] == delay)]
            ax.plot(
                subset["alpha"],
                subset["adversarial_reward"],
                # marker=markers[delay],
                color=colors[strategy_name],
                linestyle="-" if delay == 0 else "--",
                label=f"{_strategy_label(strategy_name)} | {_delay_label(delay)}",
            )

    ax.plot([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], color="black", linestyle=":", label="Reward = alpha")
    ax.set_title("Adversarial Reward vs Stake Share")
    ax.set_xlabel("Adversarial stake share alpha")
    ax.set_ylabel("Adversarial reward (future proposer share)")
    ax.legend(ncol=2, frameon=True)
    _save(fig, "adversarial_reward_vs_alpha.png")


def plot_slot_gain_vs_alpha(df: pd.DataFrame) -> None:
    summary = aggregated_results(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"honest_baseline": "#4c78a8", "adaptive_mixing": "#e45756", "forking": "#54a24b"}

    for strategy_name in summary["strategy"].unique():
        for delay in sorted(summary["vdf_delay"].unique()):
            subset = summary[(summary["strategy"] == strategy_name) & (summary["vdf_delay"] == delay)]
            ax.plot(
                subset["alpha"],
                subset["slot_gain"],
                marker="o" if delay == 0 else "s",
                color=colors[strategy_name],
                linestyle="-" if delay == 0 else "--",
                label=f"{_strategy_label(strategy_name)} | {_delay_label(delay)}",
            )

    ax.axhline(0.0, color="black", linestyle=":")
    ax.set_title("Slot Gain vs Stake Share")
    ax.set_xlabel("Adversarial stake share alpha")
    ax.set_ylabel("Slot gain = reward - alpha")
    ax.legend(ncol=2, frameon=True)
    _save(fig, "slot_gain_vs_alpha.png")


def plot_fork_rate_vs_alpha(df: pd.DataFrame) -> None:
    summary = aggregated_results(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"honest_baseline": "#4c78a8", "adaptive_mixing": "#e45756", "forking": "#54a24b"}

    for strategy_name in summary["strategy"].unique():
        for delay in sorted(summary["vdf_delay"].unique()):
            subset = summary[(summary["strategy"] == strategy_name) & (summary["vdf_delay"] == delay)]
            ax.plot(
                subset["alpha"],
                subset["fork_rate"],
                marker="o" if delay == 0 else "s",
                color=colors[strategy_name],
                linestyle="-" if delay == 0 else "--",
                label=f"{_strategy_label(strategy_name)} | {_delay_label(delay)}",
            )

    ax.set_title("Fork Rate vs Stake Share")
    ax.set_xlabel("Adversarial stake share alpha")
    ax.set_ylabel("Forks per slot")
    ax.legend(ncol=2, frameon=True)
    _save(fig, "fork_rate_vs_alpha.png")


def plot_vdf_effect(df: pd.DataFrame) -> None:
    summary = aggregated_results(df)
    focus = summary[summary["strategy"].isin(["adaptive_mixing", "forking"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0.1: "#4c78a8", 0.2: "#f58518", 0.3: "#e45756"}

    for strategy_name in sorted(focus["strategy"].unique()):
        for alpha in sorted(focus["alpha"].unique()):
            subset = focus[(focus["strategy"] == strategy_name) & (focus["alpha"] == alpha)].sort_values("vdf_delay")
            ax.plot(
                subset["vdf_delay"],
                subset["adversarial_reward"],
                marker="o" if strategy_name == "adaptive_mixing" else "s",
                linestyle="-" if strategy_name == "adaptive_mixing" else "--",
                color=colors[alpha],
                label=f"{_strategy_label(strategy_name)} | alpha={alpha:.1f}",
            )

    ax.set_title("VDF Effect on Adversarial Reward")
    ax.set_xlabel("VDF delay (slots)")
    ax.set_ylabel("Adversarial reward")
    ax.set_xticks([0, 32])
    ax.legend(ncol=2, frameon=True)
    _save(fig, "vdf_effect_reward_vs_delay.png")


def plot_time_evolution() -> None:
    spec_no_vdf = ExperimentSpec(strategy_name="forking", alpha=0.3, seed=2, vdf_delay=0, num_epochs=3)
    spec_vdf = ExperimentSpec(strategy_name="forking", alpha=0.3, seed=2, vdf_delay=32, num_epochs=3)
    sim_no_vdf, _ = run_instrumented_simulation(spec_no_vdf)
    sim_vdf, _ = run_instrumented_simulation(spec_vdf)

    df_no_vdf = block_time_series(sim_no_vdf, sim_no_vdf.analysis_adversarial_ids)
    df_vdf = block_time_series(sim_vdf, sim_vdf.analysis_adversarial_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, data, title in (
        (axes[0], df_no_vdf, "Forking without VDF"),
        (axes[1], df_vdf, "Forking with VDF"),
    ):
        ax.plot(data["slot"], data["cumulative_honest_blocks"], color="#4c78a8", label="Honest blocks")
        ax.plot(data["slot"], data["cumulative_adversarial_blocks"], color="#e45756", label="Adversarial blocks")
        ax.set_title(title)
        ax.set_xlabel("Slot")
        ax.set_ylabel("Cumulative blocks")
        ax.legend(frameon=True)

    _save(fig, "time_evolution.png")


def generate_plots(df: pd.DataFrame | None = None) -> None:
    _apply_style()
    if df is None:
        df = load_results()
    plot_reward_vs_alpha(df)
    plot_slot_gain_vs_alpha(df)
    plot_fork_rate_vs_alpha(df)
    plot_vdf_effect(df)
    plot_time_evolution()


def main() -> None:
    generate_plots()


if __name__ == "__main__":
    main()
