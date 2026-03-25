"""
PART 8: Experiments comparing honest, selfish, and forking strategies
with and without VDF.

Run:
    python experiments.py
"""

from __future__ import annotations

import json
from typing import List

from sim.simulator import Simulator, Validator
from sim.strategies import (
    ForkingStrategy,
    HonestStrategy,
    SelfishMixingStrategy,
)
from sim.randao import SLOTS_PER_EPOCH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_honest(n: int, stake: int = 1) -> List[Validator]:
    return [
        Validator(id=i, stake=stake, strategy=HonestStrategy(), secret=i * 1337 + 1)
        for i in range(n)
    ]


def run_experiment(
    label: str,
    validators: List[Validator],
    num_epochs: int = 8,
    use_vdf: bool = False,
    vdf_delay_slots: int = SLOTS_PER_EPOCH,
) -> dict:
    sim = Simulator(
        validators,
        num_epochs=num_epochs,
        use_vdf=use_vdf,
        vdf_delay_slots=vdf_delay_slots,
    )
    metrics = sim.run()
    result = metrics.summary()
    result["label"] = label
    result["use_vdf"] = use_vdf
    return result


def print_result(r: dict) -> None:
    print(f"\n{'-'*60}")
    print(f"  {r['label']}  (VDF={r['use_vdf']})")
    print(f"{'-'*60}")
    print(f"  total_blocks      : {r['total_blocks']}")
    print(f"  adversarial_ratio : {r['adversarial_ratio']:.2%}")
    print(f"  forks             : {r['forks']}")
    print(f"  reorgs            : {r['reorgs']}")
    print(f"  missed_slots      : {r['missed_slots']}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp_baseline(num_epochs: int = 8) -> None:
    """All-honest network — no adversary."""
    v = make_honest(10)
    r = run_experiment("Baseline (all honest)", v, num_epochs=num_epochs)
    print_result(r)


def exp_selfish_no_vdf(num_epochs: int = 8) -> None:
    """1 selfish-mixing adversary (skips all slots), no VDF."""
    v = make_honest(9)
    v.append(Validator(id=9, stake=1,
                       strategy=SelfishMixingStrategy(tail=SLOTS_PER_EPOCH, skip_probability=1.0),
                       secret=9 * 7777))
    r = run_experiment("SelfishMixing - no VDF", v, num_epochs=num_epochs, use_vdf=False)
    print_result(r)


def exp_selfish_with_vdf(num_epochs: int = 8) -> None:
    """1 selfish-mixing adversary (skips all slots), VDF enabled."""
    v = make_honest(9)
    v.append(Validator(id=9, stake=1,
                       strategy=SelfishMixingStrategy(tail=SLOTS_PER_EPOCH, skip_probability=1.0),
                       secret=9 * 7777))
    r = run_experiment("SelfishMixing - with VDF", v, num_epochs=num_epochs, use_vdf=True)
    print_result(r)


def exp_forking_no_vdf(num_epochs: int = 8) -> None:
    """1 forking adversary (threshold=2), no VDF."""
    v = make_honest(9)
    v.append(Validator(id=9, stake=1, strategy=ForkingStrategy(release_threshold=2), secret=9 * 5555))
    r = run_experiment("Forking (thr=2) - no VDF", v, num_epochs=num_epochs, use_vdf=False)
    print_result(r)


def exp_forking_with_vdf(num_epochs: int = 8) -> None:
    """1 forking adversary (threshold=2), VDF enabled."""
    v = make_honest(9)
    v.append(Validator(id=9, stake=1, strategy=ForkingStrategy(release_threshold=2), secret=9 * 5555))
    r = run_experiment("Forking (thr=2) - with VDF", v, num_epochs=num_epochs, use_vdf=True)
    print_result(r)


def exp_vdf_comparison(num_epochs: int = 16) -> None:
    """Side-by-side: adversarial ratio with and without VDF."""
    print("\n" + "=" * 60)
    print("VDF COMPARISON: Selfish Mixing (tail=SLOTS_PER_EPOCH)")
    print("=" * 60)

    results = []
    for use_vdf in (False, True):
        v = make_honest(9)
        v.append(Validator(id=9, stake=1,
                           strategy=SelfishMixingStrategy(tail=SLOTS_PER_EPOCH, skip_probability=1.0),
                           secret=9 * 7777))
        r = run_experiment(
            "SelfishMixing VDF=" + str(use_vdf),
            v,
            num_epochs=num_epochs,
            use_vdf=use_vdf,
        )
        results.append(r)
        print_result(r)

    no_vdf = results[0]
    with_vdf = results[1]
    print(f"\n  Δ missed_slots: {no_vdf['missed_slots']} → {with_vdf['missed_slots']}")
    print(
        f"  Adversarial ratio change: "
        f"{no_vdf['adversarial_ratio']:.2%} → {with_vdf['adversarial_ratio']:.2%}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Ethereum PoS RANDAO Simulator - Experiment Suite")
    print("=" * 60)

    exp_baseline()
    exp_selfish_no_vdf()
    exp_selfish_with_vdf()
    exp_forking_no_vdf()
    exp_forking_with_vdf()
    exp_vdf_comparison()

    print("\nDone.")
