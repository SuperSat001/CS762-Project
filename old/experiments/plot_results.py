#!/usr/bin/env python3
"""
Plot sweep results: CSVs → publication-quality PNGs + markdown report.

Usage:
  python -m experiments.plot_results --csv outputs/sweep_results.csv --out-dir outputs/

Produces:
  • plot_slot_gain.png          – adversarial slot gain vs stake (World A vs B)
  • plot_target_bias.png        – target-slot bias probability
  • plot_throughput.png         – chain throughput degradation
  • plot_missed_rate.png        – missed slot rate
  • plot_strategy_compare.png   – strategy comparison (honest/greedy/optimal)
  • plot_vdf_delay.png          – VDF delay sensitivity (if multiple D values)
  • report.md                   – markdown summary with assumptions & limitations
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGSIZE = (8, 5)
DPI = 180
WORLD_COLORS = {"A": "#2196F3", "B": "#FF5722"}
STRATEGY_COLORS = {
    "honest":  "#4CAF50",
    "greedy":  "#FF9800",
    "optimal": "#9C27B0",
    "fork":    "#F44336",
}


# ──────────────────────────────────────────────
# Load & filter helpers
# ──────────────────────────────────────────────

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise boolean columns
    if "use_vdf" in df.columns:
        df["use_vdf"] = df["use_vdf"].astype(bool)
    if "world" not in df.columns:
        df["world"] = df["use_vdf"].map({True: "B", False: "A"})
    return df


# ──────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────

def plot_slot_gain(df: pd.DataFrame, out_dir: str) -> str:
    """Adversarial slot gain (%) vs stake, World A vs B."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for world in ["A", "B"]:
        sub = df[df["world"] == world]
        # Pick the dominant strategy for clearest comparison
        for strategy in ["optimal", "greedy", "honest"]:
            s = sub[sub["strategy"] == strategy]
            if s.empty:
                continue
            grp = s.groupby("adversary_fraction").agg(
                gain_mean=("slot_gain_mean", "mean"),
                gain_ci95=("slot_gain_ci95", "mean"),
            ).reset_index()
            label = f"World {world} ({strategy})"
            color = WORLD_COLORS[world]
            ls = "-" if strategy == "optimal" else "--"
            ax.plot(
                grp["adversary_fraction"],
                grp["gain_mean"] * 100,
                color=color, linestyle=ls, marker="o", ms=4, label=label,
            )
            ax.fill_between(
                grp["adversary_fraction"],
                (grp["gain_mean"] - grp["gain_ci95"]) * 100,
                (grp["gain_mean"] + grp["gain_ci95"]) * 100,
                alpha=0.15, color=color,
            )
            break  # one strategy per world for this plot

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Adversary stake fraction (u)")
    ax.set_ylabel("Slot gain over honest baseline (%)")
    ax.set_title("Adversarial Slot Gain: RANDAO vs RANDAO+VDF")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_slot_gain.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_target_bias(df: pd.DataFrame, out_dir: str) -> str:
    """Target-slot bias probability vs stake."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for world in ["A", "B"]:
        sub = df[df["world"] == world]
        for strategy in ["optimal", "greedy", "honest"]:
            s = sub[sub["strategy"] == strategy]
            if s.empty:
                continue
            grp = s.groupby("adversary_fraction").agg(
                bias_mean=("target_slot_bias_mean", "mean"),
                bias_ci95=("target_slot_bias_ci95", "mean"),
            ).reset_index()
            ls = "-" if strategy == "optimal" else "--"
            ax.plot(
                grp["adversary_fraction"],
                grp["bias_mean"],
                color=WORLD_COLORS[world], linestyle=ls, marker="s", ms=4,
                label=f"World {world} ({strategy})",
            )
            ax.fill_between(
                grp["adversary_fraction"],
                grp["bias_mean"] - grp["bias_ci95"],
                grp["bias_mean"] + grp["bias_ci95"],
                alpha=0.12, color=WORLD_COLORS[world],
            )
            break

    # Honest baseline (u = honest probability of target slot)
    u_vals = np.linspace(0.01, 0.50, 50)
    ax.plot(u_vals, u_vals, color="gray", linestyle=":", linewidth=1.2, label="Honest baseline (u)")

    ax.set_xlabel("Adversary stake fraction (u)")
    ax.set_ylabel("Probability adversary wins target slot")
    ax.set_title("Target-Slot Bias: RANDAO vs RANDAO+VDF")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_target_bias.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_throughput(df: pd.DataFrame, out_dir: str) -> str:
    """Chain throughput vs stake."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for world in ["A", "B"]:
        sub = df[df["world"] == world]
        for strategy in ["optimal", "honest"]:
            s = sub[sub["strategy"] == strategy]
            if s.empty:
                continue
            grp = s.groupby("adversary_fraction").agg(
                tput_mean=("throughput_mean", "mean"),
                tput_ci95=("throughput_ci95", "mean"),
            ).reset_index()
            ls = "-" if strategy == "optimal" else "--"
            ax.plot(
                grp["adversary_fraction"],
                grp["tput_mean"] * 100,
                color=WORLD_COLORS[world], linestyle=ls, marker="^", ms=4,
                label=f"World {world} ({strategy})",
            )
            ax.fill_between(
                grp["adversary_fraction"],
                (grp["tput_mean"] - grp["tput_ci95"]) * 100,
                (grp["tput_mean"] + grp["tput_ci95"]) * 100,
                alpha=0.12, color=WORLD_COLORS[world],
            )
            break

    ax.axhline(100, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylim(80, 105)
    ax.set_xlabel("Adversary stake fraction (u)")
    ax.set_ylabel("Chain throughput (%)")
    ax.set_title("Throughput Degradation Under Attack")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_throughput.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_missed_rate(df: pd.DataFrame, out_dir: str) -> str:
    """Missed slot rate vs stake."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for world in ["A", "B"]:
        sub = df[df["world"] == world]
        for strategy in ["optimal", "honest"]:
            s = sub[sub["strategy"] == strategy]
            if s.empty:
                continue
            grp = s.groupby("adversary_fraction").agg(
                miss_mean=("missed_rate_mean", "mean"),
                miss_ci95=("missed_rate_ci95", "mean"),
            ).reset_index()
            ls = "-" if strategy == "optimal" else "--"
            ax.plot(
                grp["adversary_fraction"],
                grp["miss_mean"] * 100,
                color=WORLD_COLORS[world], linestyle=ls, marker="D", ms=4,
                label=f"World {world} ({strategy})",
            )
            break

    ax.set_xlabel("Adversary stake fraction (u)")
    ax.set_ylabel("Missed slot rate (%)")
    ax.set_title("Slot Miss Rate Under Withholding Attack")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_missed_rate.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_strategy_compare(df: pd.DataFrame, out_dir: str) -> str:
    """Compare strategies within World A."""
    sub = df[df["world"] == "A"]
    strategies = sorted(sub["strategy"].unique())
    if len(strategies) < 2:
        return ""  # nothing to compare

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for strat in strategies:
        s = sub[sub["strategy"] == strat]
        grp = s.groupby("adversary_fraction").agg(
            gain_mean=("slot_gain_mean", "mean"),
        ).reset_index()
        ax.plot(
            grp["adversary_fraction"],
            grp["gain_mean"] * 100,
            color=STRATEGY_COLORS.get(strat, "gray"),
            marker="o", ms=4,
            label=strat,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Adversary stake fraction (u)")
    ax.set_ylabel("Slot gain over honest baseline (%)")
    ax.set_title("Strategy Comparison (World A — RANDAO only)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_strategy_compare.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_vdf_delay(df: pd.DataFrame, out_dir: str) -> str:
    """VDF delay sensitivity: gain vs stake for D=1,2,4,..."""
    sub_b = df[df["world"] == "B"]
    delays = sorted(sub_b["vdf_delay_epochs"].unique())
    if len(delays) < 2:
        return ""  # only one delay value

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(delays) - 1)) for i in range(len(delays))]

    for d, color in zip(delays, colors):
        s = sub_b[sub_b["vdf_delay_epochs"] == d]
        for strategy in ["optimal", "greedy", "honest"]:
            ss = s[s["strategy"] == strategy]
            if ss.empty:
                continue
            grp = ss.groupby("adversary_fraction").agg(
                gain_mean=("slot_gain_mean", "mean"),
            ).reset_index()
            ax.plot(
                grp["adversary_fraction"],
                grp["gain_mean"] * 100,
                color=color, marker="o", ms=4, label=f"D={d} ({strategy})",
            )
            break

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Adversary stake fraction (u)")
    ax.set_ylabel("Slot gain over honest baseline (%)")
    ax.set_title("VDF Delay Sensitivity (World B)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_vdf_delay.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────

def write_report(df: pd.DataFrame, artifacts: List[str], out_dir: str) -> str:
    path = os.path.join(out_dir, "report.md")

    # Key numbers
    wa = df[df["world"] == "A"]
    wb = df[df["world"] == "B"]

    wa_opt = wa[wa["strategy"] == "optimal"]
    wb_opt = wb[wb["strategy"] == "optimal"]

    # Max gain at u=0.30
    u_mid = 0.30
    gain_a = wa_opt[np.isclose(wa_opt["adversary_fraction"], u_mid)]["slot_gain_mean"].mean() if not wa_opt.empty else float("nan")
    gain_b = wb_opt[np.isclose(wb_opt["adversary_fraction"], u_mid)]["slot_gain_mean"].mean() if not wb_opt.empty else float("nan")

    lines = [
        "# PoS RANDAO/VDF DES Simulation — Report",
        "",
        "## Overview",
        "This report summarises a discrete-event simulation (DES) of a Proof-of-Stake",
        "beacon chain with RANDAO randomness, comparing an adversarial setting under",
        "plain RANDAO (World A) versus RANDAO augmented with a Verifiable Delay Function",
        "(World B).",
        "",
        "## Key Results",
        "",
        f"| Metric | World A (RANDAO) | World B (RANDAO+VDF) |",
        f"|--------|-----------------|---------------------|",
    ]

    for u in sorted(df["adversary_fraction"].unique()):
        ga = wa_opt[np.isclose(wa_opt["adversary_fraction"], u)]["slot_gain_mean"].mean() if not wa_opt.empty else float("nan")
        gb = wb_opt[np.isclose(wb_opt["adversary_fraction"], u)]["slot_gain_mean"].mean() if not wb_opt.empty else float("nan")
        ta = wa_opt[np.isclose(wa_opt["adversary_fraction"], u)]["target_slot_bias_mean"].mean() if not wa_opt.empty else float("nan")
        tb = wb_opt[np.isclose(wb_opt["adversary_fraction"], u)]["target_slot_bias_mean"].mean() if not wb_opt.empty else float("nan")
        tpa = wa_opt[np.isclose(wa_opt["adversary_fraction"], u)]["throughput_mean"].mean() if not wa_opt.empty else float("nan")
        tpb = wb_opt[np.isclose(wb_opt["adversary_fraction"], u)]["throughput_mean"].mean() if not wb_opt.empty else float("nan")
        lines.append(
            f"| u={u:.2f} slot gain | {ga*100:+.2f}% | {gb*100:+.2f}% |"
            if not (np.isnan(ga) or np.isnan(gb)) else
            f"| u={u:.2f} slot gain | N/A | N/A |"
        )

    lines += [
        "",
        "## Methodology",
        "",
        "- **Simulator**: discrete-event (global min-heap event queue; O(log n) advance).",
        "- **Validators**: equal stake, N=100 (configurable).",
        "- **RANDAO**: XOR accumulator of SHA-256 hashes of slot reveals.  Honest reveals",
        "  are fresh-random per epoch; adversarial reveals are pre-committed.",
        "- **Proposer selection**: stake-weighted SHA-256–based shuffle per slot.",
        "- **Adversary strategies**: Honest (always reveal) / Greedy (Monte Carlo 1-step)",
        "  / Optimal (full 2^k enumeration of reveal subsets).",
        "- **World B VDF**: FastSimVDF (iterated SHA-256, 256 steps); temporal delay",
        "  enforced by `available_at` checks in `VDFPipeline`.  Adversary `vdf_compute_budget`",
        "  limits how many candidate mixes the adversary can score in VDF mode.",
        "",
        "## Assumptions & Limitations",
        "",
        "1. **Simplified network**: no propagation delay, all messages instantaneous.",
        "2. **No attestations**: fork choice is purely block-count based, not stake-weighted.",
        "3. **Mock VDF**: the sequential delay is simulated via `available_at`, not real",
        "   wall-clock computation.  A chiavdf adapter is provided but not exercised here.",
        "4. **Equal stake**: validators have uniform stake; weighted versions would need",
        "   compute_shuffled_index from the Ethereum spec.",
        "5. **Finality**: no finalisation mechanism (Casper FFG not modelled).",
        "6. **VDF grinding model**: the adversary's grinding is limited by `vdf_compute_budget`;",
        "   with budget=1 the adversary cannot grind alternate mixes (real-VDF constraint).",
        "",
        "## Artifact Paths",
        "",
    ]
    for a in artifacts:
        if a:
            lines.append(f"- `{a}`")

    lines += [
        "",
        "## References",
        "",
        "- [2025-037] Schwarz-Schilling et al., *RANDAO Manipulation in PoS Ethereum* (2025)",
        "- Ethereum consensus spec: https://ethereum.github.io/consensus-specs/",
        "- eth2book RANDAO: https://eth2book.info/latest/part2/building_blocks/randomness/",
        "- Boneh et al., *Verifiable Delay Functions* (2018). eprint 2018/601.",
        "- chiavdf: https://github.com/Chia-Network/chiavdf",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report → {path}")
    return path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sweep results")
    p.add_argument("--csv",     required=True, help="Path to sweep_results.csv")
    p.add_argument("--out-dir", default="outputs", help="Output directory for PNGs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_df(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    artifacts: List[str] = []
    artifacts.append(plot_slot_gain(df, args.out_dir))
    artifacts.append(plot_target_bias(df, args.out_dir))
    artifacts.append(plot_throughput(df, args.out_dir))
    artifacts.append(plot_missed_rate(df, args.out_dir))
    artifacts.append(plot_strategy_compare(df, args.out_dir))
    artifacts.append(plot_vdf_delay(df, args.out_dir))

    write_report(df, artifacts, args.out_dir)

    print("\nAll plots saved:")
    for a in artifacts:
        if a:
            print(f"  {a}")


if __name__ == "__main__":
    main()
