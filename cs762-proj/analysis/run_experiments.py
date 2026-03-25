from __future__ import annotations

import time
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (
    ALPHAS,
    EXPERIMENT_EPOCHS,
    RESULTS_CSV,
    SEEDS,
    STRATEGIES,
    VDF_DELAYS,
    aggregated_results,
    ensure_directories,
    run_instrumented_simulation,
    save_results,
    ExperimentSpec,
)


def run_all_experiments() -> pd.DataFrame:
    ensure_directories()
    total_runs = len(STRATEGIES) * len(ALPHAS) * len(SEEDS) * len(VDF_DELAYS)
    rows = []

    print(f"Running {total_runs} simulations with {EXPERIMENT_EPOCHS} epochs each")
    overall_start = time.perf_counter()

    run_number = 0
    for strategy_name in STRATEGIES:
        for alpha in ALPHAS:
            for seed in SEEDS:
                for vdf_delay in VDF_DELAYS:
                    run_number += 1
                    spec = ExperimentSpec(
                        strategy_name=strategy_name,
                        alpha=alpha,
                        seed=seed,
                        vdf_delay=vdf_delay,
                        num_epochs=EXPERIMENT_EPOCHS,
                    )
                    _, row = run_instrumented_simulation(spec)
                    rows.append(row)
                    print(
                        f"[{run_number:02d}/{total_runs}] "
                        f"strategy={strategy_name:>16} alpha={alpha:.1f} "
                        f"seed={seed} delay={vdf_delay:>2} "
                        f"runtime={row['runtime_seconds']:.3f}s "
                        f"reward={row['adversarial_reward']:.3f} "
                        f"fork_rate={row['fork_rate']:.3f} "
                        f"missed={row['missed_slots']}"
                    )

    total_runtime = time.perf_counter() - overall_start
    df = pd.DataFrame(rows).sort_values(["strategy", "alpha", "seed", "vdf_delay"]).reset_index(drop=True)
    save_results(df)

    print(f"\nSaved results to {RESULTS_CSV}")
    print("\nMean metrics by configuration:")
    print(aggregated_results(df).to_string(index=False))
    print(f"\nTotal experiment runtime: {total_runtime:.2f} seconds")

    if total_runtime > 300:
        raise RuntimeError("Runtime exceeded 300 seconds; optimize before finishing.")

    return df


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()
