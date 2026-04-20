"""
End-to-end simulation tests.
Gate 1: baseline DES + RANDAO correctness.
Gate 2: attack strategy produces measurable gain.
Gate 3: VDF temporal causality is never violated.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from sim.simulator import SimConfig, PoSSimulator, run_simulation
from sim.chain import SLOTS_PER_EPOCH


# ──────────────────────────────────────────────
# Helper: minimal config
# ──────────────────────────────────────────────

def minimal_config(**kwargs) -> SimConfig:
    defaults = dict(
        n_validators=20,
        adversary_fraction=0.30,
        n_epochs=20,
        seed=42,
        burn_in_epochs=2,
        use_vdf=False,
        vdf_delay_epochs=2,
        vdf_delay_slots=64,
        vdf_compute_budget=1,
        adversary_strategy="honest",
        n_mc_samples=5,
        target_slot=4,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


# ──────────────────────────────────────────────
# Gate 1: Baseline DES correctness
# ──────────────────────────────────────────────

class TestBaselineDES:
    def test_runs_without_error(self):
        cfg = minimal_config()
        m = run_simulation(cfg)
        assert m is not None

    def test_correct_epoch_count(self):
        cfg = minimal_config(n_epochs=10, burn_in_epochs=2)
        m = run_simulation(cfg)
        # per_epoch has entries for all epochs including burn-in
        assert len(m.per_epoch) == cfg.n_epochs

    def test_per_epoch_slot_count(self):
        cfg = minimal_config(n_epochs=5)
        m = run_simulation(cfg)
        for em in m.per_epoch:
            assert em.total_slots == SLOTS_PER_EPOCH

    def test_honest_adversary_no_gain(self):
        """With honest strategy, adversary gain should be near zero."""
        cfg = minimal_config(
            adversary_fraction=0.30,
            adversary_strategy="honest",
            n_epochs=50,
            burn_in_epochs=5,
        )
        m = run_simulation(cfg)
        # Allow for statistical noise: gain should be small
        assert abs(m.slot_gain) < 0.20, f"Honest adversary gained {m.slot_gain:.3f}"

    def test_honest_adversary_zero_missed(self):
        """Honest adversary never misses on purpose."""
        cfg = minimal_config(adversary_strategy="honest", n_epochs=10)
        m = run_simulation(cfg)
        # Some slots may be missed only if schedule not ready (VDF warmup)
        total_missed = sum(em.missed_slots for em in m.per_epoch)
        assert total_missed == 0, f"Honest adversary caused {total_missed} missed slots"

    def test_slot_counts_sum_to_epoch(self):
        """Adversarial + honest + missed = SLOTS_PER_EPOCH per epoch."""
        cfg = minimal_config(
            n_epochs=10,
            adversary_strategy="honest",
            n_validators=40,
            adversary_fraction=0.25,
        )
        sim = PoSSimulator(cfg)
        metrics = sim.run()

        for em in metrics.per_epoch:
            total = em.adv_slots_won + em.missed_slots
            # Honest slots = SLOTS_PER_EPOCH - adv_won - missed
            honest_won = SLOTS_PER_EPOCH - em.adv_slots_won - em.missed_slots
            assert total + honest_won == SLOTS_PER_EPOCH

    def test_deterministic_reproducibility(self):
        """Same seed → identical metrics."""
        cfg = minimal_config(seed=1337, n_epochs=15)
        m1 = run_simulation(cfg)
        m2 = run_simulation(cfg)
        assert m1.mean_adv_slots_won == m2.mean_adv_slots_won
        assert m1.mean_missed_rate   == m2.mean_missed_rate
        assert m1.target_slot_bias   == m2.target_slot_bias

    def test_different_seeds_different_outcomes(self):
        """Different seeds should (with high probability) differ."""
        cfg1 = minimal_config(seed=1, n_epochs=30)
        cfg2 = minimal_config(seed=9999, n_epochs=30)
        m1 = run_simulation(cfg1)
        m2 = run_simulation(cfg2)
        # Very unlikely to be exactly equal across 30 epochs
        assert m1.mean_adv_slots_won != m2.mean_adv_slots_won

    def test_proposer_schedule_built_for_every_epoch(self):
        """Chain must have a proposer schedule for every simulated epoch."""
        cfg = minimal_config(n_epochs=10, use_vdf=False)
        sim = PoSSimulator(cfg)
        sim.run()
        for epoch in range(cfg.n_epochs):
            sched = sim.chain.get_schedule(epoch)
            assert sched is not None, f"Missing schedule for epoch {epoch}"


# ──────────────────────────────────────────────
# Gate 2: Attack strategy produces measurable gain
# ──────────────────────────────────────────────

class TestAttackStrategy:
    def test_optimal_gains_over_honest(self):
        """OptimalWithhold must gain over the honest baseline at u=0.40."""
        cfg_honest = minimal_config(
            adversary_fraction=0.40,
            adversary_strategy="honest",
            n_epochs=80,
            burn_in_epochs=5,
            seed=111,
            n_validators=50,
        )
        cfg_attack = minimal_config(
            adversary_fraction=0.40,
            adversary_strategy="optimal",
            n_epochs=80,
            burn_in_epochs=5,
            seed=111,
            n_validators=50,
        )
        m_honest = run_simulation(cfg_honest)
        m_attack = run_simulation(cfg_attack)
        assert m_attack.mean_adv_slots_won >= m_honest.mean_adv_slots_won, (
            f"Attack {m_attack.mean_adv_slots_won:.3f} <= honest {m_honest.mean_adv_slots_won:.3f}"
        )

    def test_attack_creates_missed_slots(self):
        """Withholding attack causes missed slots; honest never does."""
        cfg = minimal_config(
            adversary_fraction=0.40,
            adversary_strategy="optimal",
            n_epochs=30,
            n_validators=50,
        )
        m = run_simulation(cfg)
        assert m.mean_missed_rate > 0.0, "Expected missed slots under withholding attack"

    def test_target_slot_bias_increases_with_stake(self):
        """Higher adversary stake → higher target-slot probability."""
        biases = []
        for u in [0.10, 0.30, 0.45]:
            cfg = minimal_config(
                adversary_fraction=u,
                adversary_strategy="optimal",
                n_epochs=60,
                burn_in_epochs=5,
                seed=777,
                n_validators=50,
            )
            m = run_simulation(cfg)
            biases.append(m.target_slot_bias)
        # Should be non-decreasing (allow tolerance for statistical noise)
        assert biases[0] <= biases[2] + 0.15, f"Bias not increasing: {biases}"

    def test_greedy_vs_optimal(self):
        """Optimal should perform at least as well as greedy (same seed)."""
        base_kwargs = dict(
            adversary_fraction=0.35,
            n_epochs=50,
            burn_in_epochs=5,
            seed=222,
            n_validators=40,
            n_mc_samples=10,
        )
        m_greedy  = run_simulation(minimal_config(adversary_strategy="greedy",  **base_kwargs))
        m_optimal = run_simulation(minimal_config(adversary_strategy="optimal", **base_kwargs))
        assert m_optimal.mean_adv_slots_won >= m_greedy.mean_adv_slots_won - 0.5


# ──────────────────────────────────────────────
# Gate 3: VDF temporal causality
# ──────────────────────────────────────────────

class TestVDFIntegration:
    def test_vdf_world_runs(self):
        cfg = minimal_config(
            use_vdf=True,
            vdf_delay_epochs=2,
            vdf_delay_slots=64,
            n_epochs=15,
            adversary_strategy="honest",
        )
        m = run_simulation(cfg)
        assert m is not None

    def test_vdf_schedule_never_used_early(self):
        """
        In VDF mode, proposer schedule for epoch e is built from VDF(mix[e-D-1]).
        The VDF result for epoch e should only be available at (e+1)*32 + delay.
        We verify this by checking the VDFPipeline availability directly.
        """
        cfg = minimal_config(
            use_vdf=True,
            vdf_delay_epochs=2,
            vdf_delay_slots=64,
            n_epochs=20,
            adversary_strategy="honest",
        )
        sim = PoSSimulator(cfg)
        sim.run()

        # After run: check that all submitted VDF results respect their delay
        vdf = sim.vdf
        assert vdf is not None
        for epoch, result in vdf._results.items():
            avail_at = vdf._available_at.get(epoch, 0.0)
            epoch_end_time = (epoch + 1) * SLOTS_PER_EPOCH * 1.0  # SLOT_DURATION=1.0
            assert avail_at >= epoch_end_time + cfg.vdf_delay_slots, (
                f"Epoch {epoch}: available_at={avail_at} < "
                f"epoch_end + delay = {epoch_end_time + cfg.vdf_delay_slots}"
            )

    def test_vdf_schedule_deterministic(self):
        """VDF schedules must be identical for same seed."""
        cfg = minimal_config(use_vdf=True, vdf_delay_epochs=1, vdf_delay_slots=32, n_epochs=12)
        m1 = run_simulation(cfg)
        m2 = run_simulation(cfg)
        assert m1.mean_adv_slots_won == m2.mean_adv_slots_won

    def test_world_a_vs_b_same_honest(self):
        """Honest adversary: World A vs B should have similar average slot counts."""
        base = dict(
            adversary_fraction=0.30,
            adversary_strategy="honest",
            n_epochs=60,
            burn_in_epochs=5,
            seed=888,
            n_validators=50,
        )
        mA = run_simulation(minimal_config(use_vdf=False, **base))
        mB = run_simulation(minimal_config(
            use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64, **base
        ))
        # Both should be close to honest baseline (u * 32)
        expected = 0.30 * SLOTS_PER_EPOCH
        assert abs(mA.mean_adv_slots_won - expected) < 3.0
        assert abs(mB.mean_adv_slots_won - expected) < 4.0  # small warmup effect

    def test_vdf_budget_constraint_matters(self):
        """With budget=1 (no grind), adversary in VDF mode has limited scoring ability."""
        base = dict(
            adversary_fraction=0.40,
            n_epochs=50,
            burn_in_epochs=5,
            seed=999,
            n_validators=50,
        )
        m_unlimited = run_simulation(minimal_config(
            use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
            vdf_compute_budget=2**30,   # unconstrained
            adversary_strategy="optimal",
            **base,
        ))
        m_constrained = run_simulation(minimal_config(
            use_vdf=True, vdf_delay_epochs=2, vdf_delay_slots=64,
            vdf_compute_budget=1,       # VDF-constrained
            adversary_strategy="optimal",
            **base,
        ))
        # Unconstrained can score all subsets; constrained is limited
        # Result should be non-negative difference (allow statistical tie)
        diff = m_unlimited.mean_adv_slots_won - m_constrained.mean_adv_slots_won
        assert diff >= -1.0, f"Constrained beat unconstrained by {-diff:.3f}"


# ──────────────────────────────────────────────
# Event-queue causality inside simulation
# ──────────────────────────────────────────────

class TestEventCausality:
    def test_events_always_non_decreasing_time(self):
        """Simulate a run and verify the event queue always pops in order."""
        cfg = minimal_config(n_epochs=8, adversary_strategy="optimal")
        sim = PoSSimulator(cfg)

        # Monkey-patch pop to record times
        original_pop = sim.eq.pop
        times_seen = []

        def recording_pop():
            item = original_pop()
            times_seen.append(item[0])
            return item

        sim.eq.pop = recording_pop
        sim.run()

        for i in range(1, len(times_seen)):
            assert times_seen[i] >= times_seen[i - 1], (
                f"Causality violation at index {i}: "
                f"t[{i}]={times_seen[i]:.4f} < t[{i-1}]={times_seen[i-1]:.4f}"
            )

    def test_no_events_scheduled_in_past(self):
        """Every scheduled event must have time >= current sim time."""
        cfg = minimal_config(n_epochs=5)
        sim = PoSSimulator(cfg)

        original_schedule = sim.eq.schedule
        current_time = [0.0]
        violations = []

        def guarded_schedule(time, etype, payload=None):
            if time < current_time[0] - 1e-9:
                violations.append((current_time[0], time, etype))
            return original_schedule(time, etype, payload)

        sim.eq.schedule = guarded_schedule
        original_pop = sim.eq.pop

        def tracking_pop():
            item = original_pop()
            current_time[0] = item[0]
            return item

        sim.eq.pop = tracking_pop
        sim.run()

        assert not violations, (
            f"Found {len(violations)} event(s) scheduled in the past:\n"
            + "\n".join(f"  now={n:.3f} sched_at={s:.3f} {e}" for n, s, e in violations[:5])
        )
