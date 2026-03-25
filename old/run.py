#!/usr/bin/env python3
"""
Top-level entry point for the PoS RANDAO/VDF DES simulator.

Quick modes:
  python run.py demo          # short demonstration run
  python run.py sweep         # full stake-grid sweep + plots
  python run.py test          # run pytest suite
  python run.py gate1         # verify Gate 1 (baseline DES + RANDAO)
  python run.py gate2         # verify Gate 2 (attack demo)
  python run.py gate3         # verify Gate 3 (VDF comparison)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from sim.simulator import SimConfig, run_simulation
from sim.chain import SLOTS_PER_EPOCH


# ──────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────

def hr():
    print("─" * 70)


def print_metrics(label: str, m) -> None:
    hr()
    print(f"  {label}")
    hr()
    print(f"  adv_slots_won   : {m.mean_adv_slots_won:.3f}  (baseline {m.mean_honest_baseline:.3f})")
    print(f"  slot_gain       : {m.slot_gain*100:+.2f}%  95% CI [{m.ci_adv_slots[0]:.2f}, {m.ci_adv_slots[1]:.2f}]")
    print(f"  target_slot_bias: {m.target_slot_bias*100:.2f}%  (honest baseline = u*100%)")
    print(f"  missed_rate     : {m.mean_missed_rate*100:.2f}%")
    print(f"  throughput      : {m.mean_throughput*100:.2f}%")
    print(f"  forked_honest   : {m.mean_forked_honest:.3f} blocks/epoch")
    print(f"  total_reorgs    : {m.total_reorgs}")


# ──────────────────────────────────────────────
# Demo mode
# ──────────────────────────────────────────────

def cmd_demo(args) -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  PoS RANDAO/VDF DES Simulator — Demo             ║")
    print("╚══════════════════════════════════════════════════╝")

    u = 0.30
    n_epochs = 60
    seed = 42

    configs = [
        ("World A | honest (baseline)",
         SimConfig(adversary_fraction=u, n_epochs=n_epochs, seed=seed,
                   use_vdf=False, adversary_strategy="honest",
                   n_validators=80, burn_in_epochs=5)),
        ("World A | optimal withhold",
         SimConfig(adversary_fraction=u, n_epochs=n_epochs, seed=seed,
                   use_vdf=False, adversary_strategy="optimal",
                   n_validators=80, burn_in_epochs=5)),
        ("World B | optimal withhold + VDF (D=2, budget=1)",
         SimConfig(adversary_fraction=u, n_epochs=n_epochs, seed=seed,
                   use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
                   vdf_compute_budget=1,
                   adversary_strategy="optimal",
                   n_validators=80, burn_in_epochs=5)),
        ("World B | optimal withhold + VDF (D=2, budget=∞)",
         SimConfig(adversary_fraction=u, n_epochs=n_epochs, seed=seed,
                   use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
                   vdf_compute_budget=2**30,
                   adversary_strategy="optimal",
                   n_validators=80, burn_in_epochs=5)),
    ]

    for label, cfg in configs:
        t0 = time.perf_counter()
        m = run_simulation(cfg)
        print_metrics(label, m)
        print(f"  (elapsed {time.perf_counter()-t0:.2f}s)")

    hr()
    print()


# ──────────────────────────────────────────────
# Gate 1: Baseline verification
# ──────────────────────────────────────────────

def cmd_gate1(args) -> None:
    print("\n=== Gate 1: Baseline DES + RANDAO ===\n")

    failures = []

    # 1a. Honest strategy → near-zero gain
    cfg = SimConfig(adversary_fraction=0.30, n_epochs=100, seed=42,
                    adversary_strategy="honest", n_validators=100, burn_in_epochs=5)
    m = run_simulation(cfg)
    gain_pct = m.slot_gain * 100
    ok = abs(gain_pct) < 10.0
    print(f"[{'PASS' if ok else 'FAIL'}] Honest strategy gain: {gain_pct:+.2f}% (expect ~0)")
    if not ok:
        failures.append("Gate1-a: honest gain too large")

    # 1b. Deterministic reproducibility
    m1 = run_simulation(cfg)
    m2 = run_simulation(cfg)
    ok = m1.mean_adv_slots_won == m2.mean_adv_slots_won
    print(f"[{'PASS' if ok else 'FAIL'}] Deterministic reproducibility")
    if not ok:
        failures.append("Gate1-b: not deterministic")

    # 1c. Proposer schedule for every epoch
    from sim.simulator import PoSSimulator
    sim = PoSSimulator(cfg)
    sim.run()
    missing = [e for e in range(cfg.n_epochs) if sim.chain.get_schedule(e) is None]
    ok = len(missing) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] Proposer schedule present for all {cfg.n_epochs} epochs"
          f"{' (missing: ' + str(missing[:3]) + ')' if missing else ''}")
    if not ok:
        failures.append("Gate1-c: missing schedules")

    # 1d. Event causality
    sim2 = PoSSimulator(cfg)
    original_pop = sim2.eq.pop
    times = []
    def rec_pop():
        item = original_pop()
        times.append(item[0])
        return item
    sim2.eq.pop = rec_pop
    sim2.run()
    violations = [(i, times[i-1], times[i]) for i in range(1, len(times)) if times[i] < times[i-1] - 1e-9]
    ok = len(violations) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] Event queue causality ({len(times)} events, {len(violations)} violations)")
    if not ok:
        failures.append("Gate1-d: causality violations")

    hr()
    if failures:
        print(f"Gate 1 FAILED: {failures}")
        sys.exit(1)
    else:
        print("Gate 1 PASSED ✓")


# ──────────────────────────────────────────────
# Gate 2: Attack demo
# ──────────────────────────────────────────────

def cmd_gate2(args) -> None:
    print("\n=== Gate 2: Attack Behaviour in Baseline ===\n")

    failures = []
    u_values = [0.20, 0.30, 0.40]

    print(f"  {'u':6} {'honest_gain%':13} {'optimal_gain%':14} {'attack_gt_honest':16}")
    hr()

    for u in u_values:
        base = dict(adversary_fraction=u, n_epochs=80, burn_in_epochs=5,
                    n_validators=80, seed=123, use_vdf=False)
        mh = run_simulation(SimConfig(adversary_strategy="honest", **base))
        mo = run_simulation(SimConfig(adversary_strategy="optimal", **base))
        ok = mo.mean_adv_slots_won >= mh.mean_adv_slots_won - 0.5
        gain_diff = (mo.mean_adv_slots_won - mh.mean_adv_slots_won)
        print(f"  {u:.2f}   {mh.slot_gain*100:+10.2f}%    {mo.slot_gain*100:+11.2f}%    "
              f"{'YES ✓' if ok else 'NO ✗'} (diff={gain_diff:+.2f})")
        if not ok:
            failures.append(f"Gate2: attack not > honest at u={u}")

    hr()
    if failures:
        print(f"Gate 2 FAILED: {failures}")
        sys.exit(1)
    else:
        print("Gate 2 PASSED ✓")


# ──────────────────────────────────────────────
# Gate 3: VDF comparison
# ──────────────────────────────────────────────

def cmd_gate3(args) -> None:
    print("\n=== Gate 3: VDF Integration + Comparison ===\n")

    failures = []

    # 3a. VDF temporal causality
    from sim.simulator import PoSSimulator
    cfg = SimConfig(adversary_fraction=0.30, n_epochs=20, seed=42,
                    use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
                    adversary_strategy="honest", n_validators=50, burn_in_epochs=2)
    sim = PoSSimulator(cfg)
    sim.run()
    all_ok = True
    for epoch, avail_at in (sim.vdf._available_at if sim.vdf else {}).items():
        expected_min = (epoch + 1) * SLOTS_PER_EPOCH + cfg.vdf_delay_slots
        if avail_at < expected_min - 1e-6:
            all_ok = False
    print(f"[{'PASS' if all_ok else 'FAIL'}] VDF available_at >= epoch_end + delay_slots")
    if not all_ok:
        failures.append("Gate3-a: VDF available before delay")

    # 3b. World A vs B: budget=1 world B
    print("\n  Comparison: World A (optimal) vs World B (budget=1 optimal):")
    print(f"  {'u':6} {'worldA_gain%':14} {'worldB_gain%':14} {'B_budget_lt_unconstrained':25}")
    hr()

    for u in [0.25, 0.35]:
        base = dict(adversary_fraction=u, n_epochs=60, burn_in_epochs=5,
                    n_validators=60, seed=999)
        mA = run_simulation(SimConfig(use_vdf=False, adversary_strategy="optimal", **base))
        mB1 = run_simulation(SimConfig(use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
                                        vdf_compute_budget=1,
                                        adversary_strategy="optimal", **base))
        mBinf = run_simulation(SimConfig(use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
                                          vdf_compute_budget=2**30,
                                          adversary_strategy="optimal", **base))
        ok = mBinf.mean_adv_slots_won >= mB1.mean_adv_slots_won - 0.3
        print(f"  {u:.2f}   {mA.slot_gain*100:+11.2f}%   {mB1.slot_gain*100:+11.2f}%   "
              f"{'YES ✓' if ok else 'NO ✗'}")
        if not ok:
            failures.append(f"Gate3-b: unconstrained not >= budget=1 at u={u}")

    hr()
    if failures:
        print(f"Gate 3 FAILED: {failures}")
        sys.exit(1)
    else:
        print("Gate 3 PASSED ✓")


# ──────────────────────────────────────────────
# Sweep mode
# ──────────────────────────────────────────────

def cmd_sweep(args) -> None:
    out_dir = args.out_dir or "outputs"
    sweep_args = [
        sys.executable, "-m", "experiments.run_sweep",
        "--stake-grid", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45",
        "--strategies", "honest", "optimal",
        "--vdf-delay-epochs", "2",
        "--n-epochs", str(args.n_epochs),
        "--n-seeds", str(args.n_seeds),
        "--base-seed", "42",
        "--out-dir", out_dir,
    ]
    subprocess.run(sweep_args, cwd=BASE_DIR, check=True)

    # Now plot
    plot_args = [
        sys.executable, "-m", "experiments.plot_results",
        "--csv", os.path.join(out_dir, "sweep_results.csv"),
        "--out-dir", out_dir,
    ]
    subprocess.run(plot_args, cwd=BASE_DIR, check=True)


# ──────────────────────────────────────────────
# Test mode
# ──────────────────────────────────────────────

def cmd_test(args) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=BASE_DIR,
    )
    sys.exit(result.returncode)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="PoS RANDAO/VDF DES Simulator")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("demo",  help="Short demonstration run")
    sub.add_parser("gate1", help="Verify Gate 1: baseline DES + RANDAO")
    sub.add_parser("gate2", help="Verify Gate 2: attack behaviour")
    sub.add_parser("gate3", help="Verify Gate 3: VDF comparison")
    sub.add_parser("test",  help="Run pytest test suite")

    sp = sub.add_parser("sweep", help="Full stake-grid sweep + plots")
    sp.add_argument("--n-epochs", type=int, default=100)
    sp.add_argument("--n-seeds",  type=int, default=3)
    sp.add_argument("--out-dir",  default="outputs")

    args = p.parse_args()

    if   args.cmd == "demo":  cmd_demo(args)
    elif args.cmd == "gate1": cmd_gate1(args)
    elif args.cmd == "gate2": cmd_gate2(args)
    elif args.cmd == "gate3": cmd_gate3(args)
    elif args.cmd == "sweep": cmd_sweep(args)
    elif args.cmd == "test":  cmd_test(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
