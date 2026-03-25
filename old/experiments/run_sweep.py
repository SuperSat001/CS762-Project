#!/usr/bin/env python3
"""
Experiment runner: sweep adversary stake fraction across World A and World B.

Usage examples:
  # Quick test run
  python -m experiments.run_sweep --n-epochs 30 --n-seeds 2 --out-dir outputs/quick

  # Full sweep
  python -m experiments.run_sweep --n-epochs 200 --n-seeds 5 --out-dir outputs/full

  # VDF delay sensitivity
  python -m experiments.run_sweep --vdf-delay-epochs 1 2 4 --n-epochs 100 --n-seeds 3

  # Single stake point (fast)
  python -m experiments.run_sweep --stake-grid 0.30 --strategy optimal --n-epochs 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

# Allow running as `python experiments/run_sweep.py` or `python -m experiments.run_sweep`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.simulator import SimConfig, SimMetrics, run_simulation


# ──────────────────────────────────────────────
# Single configuration runner (with CI over seeds)
# ──────────────────────────────────────────────

def run_one(
    adversary_fraction: float,
    use_vdf: bool,
    vdf_delay_epochs: int,
    vdf_compute_budget: int,
    n_epochs: int,
    n_seeds: int,
    base_seed: int,
    strategy: str,
    n_validators: int,
    target_slot: int,
    burn_in_epochs: int,
    enable_fork_attack: bool,
) -> Dict[str, float]:
    """Run n_seeds simulations for one (stake, world) combo; return aggregated stats."""
    metrics_list: List[SimMetrics] = []

    for seed_offset in range(n_seeds):
        cfg = SimConfig(
            n_validators=n_validators,
            adversary_fraction=adversary_fraction,
            n_epochs=n_epochs,
            seed=base_seed + seed_offset * 997,  # well-separated seeds
            burn_in_epochs=burn_in_epochs,
            use_vdf=use_vdf,
            vdf_delay_epochs=vdf_delay_epochs,
            vdf_delay_slots=vdf_delay_epochs * 32,
            vdf_compute_budget=vdf_compute_budget,
            adversary_strategy=strategy,
            target_slot=target_slot,
            enable_fork_attack=enable_fork_attack,
        )
        m = run_simulation(cfg)
        metrics_list.append(m)

    # Aggregate across seeds
    adv_won    = np.array([m.mean_adv_slots_won   for m in metrics_list])
    baseline   = np.array([m.mean_honest_baseline  for m in metrics_list])
    missed     = np.array([m.mean_missed_rate       for m in metrics_list])
    forked     = np.array([m.mean_forked_honest     for m in metrics_list])
    reorgs_arr = np.array([m.total_reorgs           for m in metrics_list], dtype=float)
    target_b   = np.array([m.target_slot_bias        for m in metrics_list])
    throughput = np.array([m.mean_throughput         for m in metrics_list])
    gain       = np.array([m.slot_gain               for m in metrics_list])

    def ci95(arr: np.ndarray):
        if len(arr) < 2:
            return 0.0
        return 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))

    return {
        "adversary_fraction": adversary_fraction,
        "use_vdf": int(use_vdf),
        "vdf_delay_epochs": vdf_delay_epochs,
        "vdf_compute_budget": vdf_compute_budget,
        "strategy": strategy,
        "n_seeds": n_seeds,
        "n_epochs": n_epochs,

        "adv_slots_won_mean": float(adv_won.mean()),
        "adv_slots_won_ci95": float(ci95(adv_won)),
        "honest_baseline_mean": float(baseline.mean()),
        "slot_gain_mean": float(gain.mean()),
        "slot_gain_ci95": float(ci95(gain)),

        "missed_rate_mean": float(missed.mean()),
        "missed_rate_ci95": float(ci95(missed)),
        "forked_honest_mean": float(forked.mean()),
        "reorg_count_mean": float(reorgs_arr.mean()),
        "target_slot_bias_mean": float(target_b.mean()),
        "target_slot_bias_ci95": float(ci95(target_b)),
        "throughput_mean": float(throughput.mean()),
        "throughput_ci95": float(ci95(throughput)),
    }


# ──────────────────────────────────────────────
# Full sweep
# ──────────────────────────────────────────────

def run_sweep(args: argparse.Namespace) -> List[Dict]:
    stake_grid = sorted(set(args.stake_grid))
    vdf_delays = sorted(set(args.vdf_delay_epochs))

    rows: List[Dict] = []
    total = len(stake_grid) * (1 + len(vdf_delays)) * 2  # rough count
    done  = 0

    for u in stake_grid:
        # ── World A: RANDAO only ──────────────────
        for strategy in args.strategies:
            print(f"  [World A | u={u:.2f} | strategy={strategy}]", end=" ", flush=True)
            t0 = time.perf_counter()
            row = run_one(
                adversary_fraction=u,
                use_vdf=False,
                vdf_delay_epochs=0,
                vdf_compute_budget=2**30,
                n_epochs=args.n_epochs,
                n_seeds=args.n_seeds,
                base_seed=args.base_seed,
                strategy=strategy,
                n_validators=args.n_validators,
                target_slot=args.target_slot,
                burn_in_epochs=args.burn_in_epochs,
                enable_fork_attack=(strategy == "fork"),
            )
            row["world"] = "A"
            rows.append(row)
            done += 1
            print(f"done in {time.perf_counter()-t0:.1f}s  "
                  f"gain={row['slot_gain_mean']:+.3f}")

        # ── World B: RANDAO + VDF ─────────────────
        for vdf_d in vdf_delays:
            for strategy in args.strategies:
                budget = args.vdf_compute_budget
                print(
                    f"  [World B | u={u:.2f} | D={vdf_d} | budget={budget} | strategy={strategy}]",
                    end=" ", flush=True,
                )
                t0 = time.perf_counter()
                row = run_one(
                    adversary_fraction=u,
                    use_vdf=True,
                    vdf_delay_epochs=vdf_d,
                    vdf_compute_budget=budget,
                    n_epochs=args.n_epochs,
                    n_seeds=args.n_seeds,
                    base_seed=args.base_seed,
                    strategy=strategy,
                    n_validators=args.n_validators,
                    target_slot=args.target_slot,
                    burn_in_epochs=args.burn_in_epochs,
                    enable_fork_attack=(strategy == "fork"),
                )
                row["world"] = "B"
                rows.append(row)
                done += 1
                print(f"done in {time.perf_counter()-t0:.1f}s  "
                      f"gain={row['slot_gain_mean']:+.3f}")

    return rows


# ──────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────

def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV → {path}")


def save_json(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved JSON → {path}")


def print_summary_table(rows: List[Dict]) -> None:
    """Print a concise ASCII table of key results."""
    print("\n" + "=" * 90)
    print(f"{'World':6} {'u':6} {'D':4} {'budget':7} {'strategy':10} "
          f"{'adv_won':9} {'baseline':9} {'gain%':7} {'miss%':6} {'thru%':6}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r.get('world','?'):6} "
            f"{r['adversary_fraction']:6.2f} "
            f"{r['vdf_delay_epochs']:4d} "
            f"{r['vdf_compute_budget']:7d} "
            f"{r['strategy']:10s} "
            f"{r['adv_slots_won_mean']:9.3f} "
            f"{r['honest_baseline_mean']:9.3f} "
            f"{r['slot_gain_mean']*100:+7.2f}% "
            f"{r['missed_rate_mean']*100:6.2f}% "
            f"{r['throughput_mean']*100:6.2f}% "
        )
    print("=" * 90)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PoS RANDAO/VDF DES sweep experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stake-grid",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        metavar="U",
        help="Adversary stake fractions to sweep",
    )
    p.add_argument("--n-epochs",      type=int,   default=100,   help="Epochs per run")
    p.add_argument("--n-seeds",       type=int,   default=3,     help="Seeds per config (for CI)")
    p.add_argument("--base-seed",     type=int,   default=42,    help="Base RNG seed")
    p.add_argument("--n-validators",  type=int,   default=100,   help="Total validators")
    p.add_argument("--burn-in-epochs",type=int,   default=5,     help="Burn-in epochs to skip")
    p.add_argument("--target-slot",   type=int,   default=8,     help="Target slot-in-epoch index")
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=["honest", "greedy", "optimal", "fork"],
        default=["honest", "optimal"],
        help="Adversary strategies to compare",
    )
    p.add_argument(
        "--vdf-delay-epochs",
        nargs="+",
        type=int,
        default=[2],
        metavar="D",
        help="VDF delay values (in epochs) for World B",
    )
    p.add_argument(
        "--vdf-compute-budget",
        type=int,
        default=1,
        help="Adversary VDF evaluations per epoch (1=no-grind, 2^k=full)",
    )
    p.add_argument("--out-dir", default="outputs", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nPoS RANDAO/VDF DES Sweep")
    print(f"  stake_grid       = {args.stake_grid}")
    print(f"  strategies       = {args.strategies}")
    print(f"  vdf_delay_epochs = {args.vdf_delay_epochs}")
    print(f"  n_epochs / seed  = {args.n_epochs} / {args.n_seeds}")
    print(f"  n_validators     = {args.n_validators}")
    print()

    t_start = time.perf_counter()
    rows = run_sweep(args)
    elapsed = time.perf_counter() - t_start

    # Save outputs
    csv_path  = os.path.join(args.out_dir, "sweep_results.csv")
    json_path = os.path.join(args.out_dir, "sweep_results.json")
    save_csv(rows, csv_path)
    save_json(rows, json_path)

    print_summary_table(rows)
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Artifacts: {csv_path}, {json_path}")


if __name__ == "__main__":
    main()
