"""
Main Discrete-Event Simulator (DES) for PoS RANDAO/VDF beacon chain.

Event flow:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  EPOCH_START(e)                                                     │
  │    • adversary computes epoch strategy (reveal/skip decisions)      │
  │    • schedules SLOT_START for all slots in epoch                    │
  │                                                                     │
  │  SLOT_START(s)   [t = s * SLOT_DURATION]                            │
  │    • look up proposer for slot s                                    │
  │    • if honest: schedule BLOCK_PROPOSE(s) at t + PROP_DELAY         │
  │    • if adversary AND decision=propose: schedule BLOCK_PROPOSE(s)   │
  │    • if adversary AND decision=skip: mark slot missed               │
  │                                                                     │
  │  BLOCK_PROPOSE(s)  [t = s * SLOT_DURATION + PROP_DELAY]            │
  │    • create block, apply RANDAO reveal to accumulator               │
  │    • add to canonical chain                                         │
  │                                                                     │
  │  EPOCH_END(e)   [t = (e+1) * SLOTS_PER_EPOCH * SLOT_DURATION]      │
  │    • finalise RANDAO mix[e]                                         │
  │    • In World A: build proposer schedule for epoch e+1              │
  │    • In World B: submit mix[e] to VDF pipeline;                     │
  │        schedule VDF_COMPLETE(e) at t + delay_slots                  │
  │    • schedule EPOCH_START(e+1)                                      │
  │                                                                     │
  │  VDF_COMPLETE(e)                                                    │
  │    • VDF result for epoch e is now available                        │
  │    • build proposer schedule for epoch e + vdf_delay_epochs + 1    │
  │                                                                     │
  │  FORK_ATTEMPT(e)                                                    │
  │    • adversary tries to replace tail blocks with alternate chain    │
  └─────────────────────────────────────────────────────────────────────┘

Metrics collected per epoch (after burn-in):
  • adv_slots_won         – adversarial blocks in canonical chain
  • honest_baseline       – u * SLOTS_PER_EPOCH (expected without attack)
  • missed_slots          – proposer skipped
  • forked_honest         – honest blocks displaced by reorgs
  • reorg_count           – number of successful fork attempts
  • target_slot_hit       – did adversary win the configured target slot?
  • throughput            – non-missed blocks / total slots
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chain import (
    Block,
    ChainState,
    ProposerSchedule,
    Validator,
    SLOTS_PER_EPOCH,
    SLOT_DURATION,
    PROP_DELAY,
)
from .event_queue import (
    EventQueue,
    EVENT_SLOT_START,
    EVENT_BLOCK_PROPOSE,
    EVENT_EPOCH_START,
    EVENT_EPOCH_END,
    EVENT_VDF_COMPLETE,
    EVENT_ADVERSARY_DECIDE,
    EVENT_FORK_ATTEMPT,
)
from .randao import RANDAOState, adversary_reveal, honest_reveal
from .vdf import VDFPipeline, make_vdf_pipeline
from .adversary import AdversaryStrategy, ForkAdversary, make_adversary


# ──────────────────────────────────────────────
# Simulation configuration
# ──────────────────────────────────────────────

@dataclass
class SimConfig:
    n_validators: int = 100
    adversary_fraction: float = 0.30
    n_epochs: int = 100
    seed: int = 42
    burn_in_epochs: int = 5

    # Randomness mode
    use_vdf: bool = False
    vdf_delay_epochs: int = 2           # D: VDF output used for epoch e+D+1
    vdf_delay_slots: int = 64           # delay in sim-time units (slots)
    vdf_compute_budget: int = 1         # adversary VDF evaluations per epoch

    # Adversary
    adversary_strategy: str = "optimal"  # honest / greedy / optimal / fork
    n_mc_samples: int = 20
    target_slot: int = 8                # which slot-in-epoch to track

    # Attack modifiers
    enable_fork_attack: bool = False
    fork_depth: int = 3


# ──────────────────────────────────────────────
# Per-epoch metrics
# ──────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    adv_slots_won: int = 0
    honest_baseline: float = 0.0
    missed_slots: int = 0
    forked_honest: int = 0
    reorg_count: int = 0
    target_slot_hit: bool = False
    total_slots: int = SLOTS_PER_EPOCH
    adv_slots_scheduled: int = 0       # how many adv slots were on schedule


@dataclass
class SimMetrics:
    config: SimConfig
    per_epoch: List[EpochMetrics] = field(default_factory=list)

    # Aggregates (computed after run)
    mean_adv_slots_won: float = 0.0
    mean_honest_baseline: float = 0.0
    mean_missed_rate: float = 0.0
    mean_forked_honest: float = 0.0
    total_reorgs: int = 0
    target_slot_bias: float = 0.0
    mean_throughput: float = 0.0
    slot_gain: float = 0.0            # (adv_won - honest_baseline) / honest_baseline
    ci_adv_slots: Tuple[float, float] = (0.0, 0.0)   # 95% CI

    def finalize(self, burn_in: int) -> None:
        """Compute aggregates from per-epoch data (after burn-in)."""
        post = [e for e in self.per_epoch if e.epoch >= burn_in]
        if not post:
            return

        adv_won    = np.array([e.adv_slots_won    for e in post], dtype=float)
        baseline   = np.array([e.honest_baseline   for e in post], dtype=float)
        missed     = np.array([e.missed_slots      for e in post], dtype=float)
        forked     = np.array([e.forked_honest     for e in post], dtype=float)
        reorgs     = np.array([e.reorg_count       for e in post], dtype=int)
        target_hit = np.array([e.target_slot_hit   for e in post], dtype=float)
        throughput = np.array(
            [(e.total_slots - e.missed_slots) / e.total_slots for e in post],
            dtype=float,
        )

        self.mean_adv_slots_won  = float(adv_won.mean())
        self.mean_honest_baseline = float(baseline.mean())
        self.mean_missed_rate    = float((missed / SLOTS_PER_EPOCH).mean())
        self.mean_forked_honest  = float(forked.mean())
        self.total_reorgs        = int(reorgs.sum())
        self.target_slot_bias    = float(target_hit.mean())
        self.mean_throughput     = float(throughput.mean())

        if self.mean_honest_baseline > 0:
            self.slot_gain = (
                self.mean_adv_slots_won - self.mean_honest_baseline
            ) / self.mean_honest_baseline
        else:
            self.slot_gain = 0.0

        # 95% CI on adv slots won
        n = len(adv_won)
        se = adv_won.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        self.ci_adv_slots = (
            float(adv_won.mean() - 1.96 * se),
            float(adv_won.mean() + 1.96 * se),
        )


# ──────────────────────────────────────────────
# Main simulator
# ──────────────────────────────────────────────

class PoSSimulator:
    """
    Discrete-event simulator for a PoS chain with RANDAO (±VDF) randomness.

    Parameters
    ----------
    config : SimConfig
    """

    def __init__(self, config: SimConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.eq  = EventQueue()

        # Build validator set
        n_adv  = round(config.n_validators * config.adversary_fraction)
        n_hon  = config.n_validators - n_adv
        self.validators: List[Validator] = [
            Validator(validator_id=i, is_adversarial=True)  for i in range(n_adv)
        ] + [
            Validator(validator_id=i + n_adv, is_adversarial=False) for i in range(n_hon)
        ]
        self.adv_ids: set[int] = {v.validator_id for v in self.validators if v.is_adversarial}
        self.hon_ids: set[int] = {v.validator_id for v in self.validators if not v.is_adversarial}

        # RANDAO state
        genesis_seed = config.seed ^ 0xDEADBEEF
        self.randao  = RANDAOState(genesis_seed)

        # Chain state (epoch 0 schedule built from genesis mix)
        genesis_mix  = self.randao.get_mix(-1)
        self.chain   = ChainState(self.validators, genesis_mix)

        # VDF pipeline (World B only)
        self.vdf: Optional[VDFPipeline] = None
        if config.use_vdf:
            self.vdf = make_vdf_pipeline(
                use_real_vdf=False,
                delay_slots=config.vdf_delay_slots,
                vdf_compute_budget=config.vdf_compute_budget,
            )

        # Adversary strategy
        adv_seed = config.seed ^ 0xCAFEBABE
        self.strategy: AdversaryStrategy = make_adversary(
            strategy_name=config.adversary_strategy,
            validators=self.validators,
            adv_seed=adv_seed,
            vdf_compute_budget=config.vdf_compute_budget,
            n_mc_samples=config.n_mc_samples,
        )

        # Per-epoch state
        self._epoch_decisions: Dict[int, Dict[int, bool]] = {}   # epoch -> {slot_in_epoch: propose}
        self._epoch_adv_reveals: Dict[int, Dict[int, bytes]] = {}  # epoch -> {slot_in_epoch: reveal}
        self._epoch_honest_reveals: Dict[int, Dict[int, bytes]] = {}  # epoch -> {slot_in_epoch: reveal}
        self._epoch_seeds: Dict[int, int] = {}

        # Metrics
        self.metrics = SimMetrics(config=config)
        self._current_epoch_metrics: Optional[EpochMetrics] = None

        # Pending VDF schedules (epoch -> target_schedule_epoch)
        self._vdf_pending: Dict[int, int] = {}

    # ──────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────

    def run(self) -> SimMetrics:
        """Run the full simulation and return collected metrics."""
        # Kick off with EPOCH_START for epoch 0
        self.eq.schedule(0.0, EVENT_EPOCH_START, {"epoch": 0})

        while not self.eq.empty():
            t, eid, etype, payload = self.eq.pop()
            self._dispatch(t, etype, payload)

        self.metrics.finalize(self.cfg.burn_in_epochs)
        return self.metrics

    # ──────────────────────────────────────────
    # Dispatcher
    # ──────────────────────────────────────────

    def _dispatch(self, t: float, etype: str, payload: dict) -> None:
        if   etype == EVENT_EPOCH_START:    self._on_epoch_start(t, payload)
        elif etype == EVENT_SLOT_START:     self._on_slot_start(t, payload)
        elif etype == EVENT_BLOCK_PROPOSE:  self._on_block_propose(t, payload)
        elif etype == EVENT_EPOCH_END:      self._on_epoch_end(t, payload)
        elif etype == EVENT_VDF_COMPLETE:   self._on_vdf_complete(t, payload)
        elif etype == EVENT_FORK_ATTEMPT:   self._on_fork_attempt(t, payload)

    # ──────────────────────────────────────────
    # EPOCH_START handler
    # ──────────────────────────────────────────

    def _on_epoch_start(self, t: float, payload: dict) -> None:
        epoch: int = payload["epoch"]

        if epoch >= self.cfg.n_epochs:
            return   # simulation complete

        # Epoch metrics — honest_baseline = expected adversarial slots without attack
        honest_baseline = self.cfg.adversary_fraction * SLOTS_PER_EPOCH
        self._current_epoch_metrics = EpochMetrics(
            epoch=epoch,
            honest_baseline=honest_baseline,
        )

        # Fresh epoch seed for honest reveals (unknown to adversary in advance)
        epoch_seed = int(self.rng.integers(0, 2**63))
        self._epoch_seeds[epoch] = epoch_seed

        # Pre-generate ALL reveals for this epoch (honest and adversarial)
        hon_reveals: Dict[int, bytes] = {}
        adv_reveals: Dict[int, bytes] = {}

        # First: determine who proposes each slot (need schedule for epoch).
        # In VDF warmup, schedule may not exist yet; cascade-search for the most
        # recent available schedule as a fallback (genesis at worst).
        sched = self.chain.get_schedule(epoch)
        if sched is None:
            for fallback_e in range(epoch - 1, -2, -1):
                sched = self.chain.get_schedule(fallback_e)
                if sched is not None:
                    break
            # If still None (shouldn't happen), create a fresh one from genesis
            if sched is None:
                sched = self.chain.set_schedule(epoch, self.randao.get_mix(-1))

        # Generate reveals
        for s in range(SLOTS_PER_EPOCH):
            vid = sched.proposer_for(s)
            if vid in self.adv_ids:
                r = adversary_reveal(self.strategy.adv_seed, vid, epoch)
                adv_reveals[s] = r
            else:
                r = honest_reveal(epoch_seed, vid, epoch)
                hon_reveals[s] = r

        self._epoch_adv_reveals[epoch]    = adv_reveals
        self._epoch_honest_reveals[epoch] = hon_reveals

        # Adversary computes strategy for this epoch
        adv_slots_sorted = sorted(adv_reveals.keys())

        # Target epoch: where the current epoch's RANDAO mix is USED
        if self.vdf is not None:
            target_epoch = epoch + self.cfg.vdf_delay_epochs + 1
        else:
            target_epoch = epoch + 1

        # VDF compute budget for this epoch
        vdf_for_strategy = self.vdf if self.cfg.use_vdf else None

        decisions = self.strategy.decide_epoch_reveals(
            epoch=epoch,
            adv_slots=adv_slots_sorted,
            adv_reveals=[adv_reveals[s] for s in adv_slots_sorted],
            current_mix_at_epoch_start=self.randao.get_mix(epoch - 1) or self.randao.get_mix(-1),
            honest_reveals_this_epoch=hon_reveals,
            target_epoch=target_epoch,
            rng=self.rng,
            vdf_pipeline=vdf_for_strategy,
        )
        self._epoch_decisions[epoch] = decisions

        # Record scheduled adversarial slots
        if self._current_epoch_metrics:
            self._current_epoch_metrics.adv_slots_scheduled = len(adv_slots_sorted)

        # Begin RANDAO accumulation for this epoch
        self.randao.begin_epoch(epoch)

        # Schedule all slots in epoch
        epoch_start_slot = epoch * SLOTS_PER_EPOCH
        for slot_offset in range(SLOTS_PER_EPOCH):
            global_slot = epoch_start_slot + slot_offset
            self.eq.schedule(
                t + slot_offset * SLOT_DURATION,
                EVENT_SLOT_START,
                {"slot": global_slot, "epoch": epoch, "slot_in_epoch": slot_offset},
            )

        # Schedule EPOCH_END
        self.eq.schedule(
            t + SLOTS_PER_EPOCH * SLOT_DURATION,
            EVENT_EPOCH_END,
            {"epoch": epoch},
        )

    # ──────────────────────────────────────────
    # SLOT_START handler
    # ──────────────────────────────────────────

    def _on_slot_start(self, t: float, payload: dict) -> None:
        slot:         int = payload["slot"]
        epoch:        int = payload["epoch"]
        slot_in_epoch: int = payload["slot_in_epoch"]

        # During VDF warmup, fall back to the most recent available schedule.
        sched = self.chain.get_schedule(epoch)
        if sched is None:
            for fallback_e in range(epoch - 1, -2, -1):
                sched = self.chain.get_schedule(fallback_e)
                if sched is not None:
                    break
        if sched is None:
            self._apply_miss(slot, epoch, slot_in_epoch, t)
            return

        proposer_id = sched.proposer_for(slot_in_epoch)
        is_adv = proposer_id in self.adv_ids

        if is_adv:
            decisions = self._epoch_decisions.get(epoch, {})
            propose = decisions.get(slot_in_epoch, True)
        else:
            propose = True   # honest always propose

        if propose:
            reveal = (
                self._epoch_adv_reveals[epoch].get(slot_in_epoch)
                if is_adv
                else self._epoch_honest_reveals[epoch].get(slot_in_epoch)
            )
            self.eq.schedule(
                t + PROP_DELAY,
                EVENT_BLOCK_PROPOSE,
                {
                    "slot": slot,
                    "epoch": epoch,
                    "slot_in_epoch": slot_in_epoch,
                    "proposer_id": proposer_id,
                    "reveal": reveal,
                    "is_adversarial": is_adv,
                },
            )
        else:
            self._apply_miss(slot, epoch, slot_in_epoch, t)

    # ──────────────────────────────────────────
    # BLOCK_PROPOSE handler
    # ──────────────────────────────────────────

    def _on_block_propose(self, t: float, payload: dict) -> None:
        slot:         int   = payload["slot"]
        epoch:        int   = payload["epoch"]
        proposer_id:  int   = payload["proposer_id"]
        reveal:       bytes = payload["reveal"]
        is_adv:       bool  = payload["is_adversarial"]

        block = Block(
            slot=slot,
            epoch=epoch,
            proposer_id=proposer_id,
            parent_hash=self.chain.head_hash(),
            randao_reveal=reveal,
            is_adversarial=is_adv,
            is_missed=False,
        )
        self.chain.add_block(block)
        self.randao.apply_reveal(reveal)

        if self._current_epoch_metrics and is_adv:
            self._current_epoch_metrics.adv_slots_won += 1

        # Check target slot
        if (
            self._current_epoch_metrics
            and payload["slot_in_epoch"] == self.cfg.target_slot
            and is_adv
        ):
            self._current_epoch_metrics.target_slot_hit = True

    # ──────────────────────────────────────────
    # EPOCH_END handler
    # ──────────────────────────────────────────

    def _on_epoch_end(self, t: float, payload: dict) -> None:
        epoch: int = payload["epoch"]

        # Finalise RANDAO mix for this epoch
        final_mix = self.randao.finalize_epoch(epoch)

        if self.vdf is None:
            # World A: build next epoch's schedule immediately
            self.chain.set_schedule(epoch + 1, final_mix)
        else:
            # World B: submit to VDF pipeline; schedule VDF_COMPLETE
            vdf_available_at = t + self.cfg.vdf_delay_slots
            self.vdf.submit(epoch, final_mix, epoch_end_time=t)
            target_sched_epoch = epoch + self.cfg.vdf_delay_epochs + 1
            self._vdf_pending[epoch] = target_sched_epoch
            self.eq.schedule(
                vdf_available_at,
                EVENT_VDF_COMPLETE,
                {"vdf_epoch": epoch, "target_schedule_epoch": target_sched_epoch},
            )

        # Fork attempt (if strategy supports it and there are withheld blocks)
        if isinstance(self.strategy, ForkAdversary):
            decisions = self._epoch_decisions.get(epoch, {})
            adv_slots  = sorted(self._epoch_adv_reveals.get(epoch, {}).keys())
            adv_reveals_list = [self._epoch_adv_reveals[epoch][s] for s in adv_slots]
            withheld = self.strategy.fork_plan(epoch, adv_slots, adv_reveals_list, decisions)
            if withheld:
                self.eq.schedule(
                    t + 0.01,
                    EVENT_FORK_ATTEMPT,
                    {
                        "epoch": epoch,
                        "withheld_slots_in_epoch": withheld,
                        "epoch_end_time": t,
                    },
                )

        # Collect epoch metrics
        if self._current_epoch_metrics:
            em = self._current_epoch_metrics
            em.missed_slots    = len([s for s in self.chain.missed_slots
                                       if s // SLOTS_PER_EPOCH == epoch])
            em.forked_honest   = self.chain.forked_honest_count
            em.reorg_count     = self.chain.reorg_count
            # Reset per-epoch counters for forked/reorg (cumulative → per-epoch)
            self.chain.forked_honest_count = 0
            self.chain.reorg_count        = 0
            self.metrics.per_epoch.append(em)

        # Schedule next epoch
        next_epoch = epoch + 1
        if next_epoch < self.cfg.n_epochs:
            self.eq.schedule(t, EVENT_EPOCH_START, {"epoch": next_epoch})

    # ──────────────────────────────────────────
    # VDF_COMPLETE handler
    # ──────────────────────────────────────────

    def _on_vdf_complete(self, t: float, payload: dict) -> None:
        vdf_epoch:    int = payload["vdf_epoch"]
        target_epoch: int = payload["target_schedule_epoch"]

        result = self.vdf.get_result(vdf_epoch, t)
        if result is None:
            # Should not happen (event was scheduled at available_at)
            return

        # Build the proposer schedule for the target epoch using VDF output
        self.chain.set_schedule(target_epoch, result.output)

    # ──────────────────────────────────────────
    # FORK_ATTEMPT handler
    # ──────────────────────────────────────────

    def _on_fork_attempt(self, t: float, payload: dict) -> None:
        epoch:                int       = payload["epoch"]
        withheld_slots_in_epoch: List[int] = payload["withheld_slots_in_epoch"]

        if not withheld_slots_in_epoch:
            return

        epoch_start_slot = epoch * SLOTS_PER_EPOCH
        # Build alternate blocks for withheld slots
        alternate: List[Block] = []
        for s_in_epoch in withheld_slots_in_epoch:
            global_slot = epoch_start_slot + s_in_epoch
            vid = self.chain.proposer_for(global_slot)
            reveal = self._epoch_adv_reveals[epoch].get(s_in_epoch)
            alt_block = Block(
                slot=global_slot,
                epoch=epoch,
                proposer_id=vid,
                parent_hash=self.chain.head_hash(),
                randao_reveal=reveal,
                is_adversarial=True,
                is_missed=False,
            )
            alternate.append(alt_block)

        fork_start = epoch_start_slot + withheld_slots_in_epoch[0]
        accepted = self.chain.attempt_fork(fork_start, alternate, t)

        if accepted and self._current_epoch_metrics:
            self._current_epoch_metrics.reorg_count += 1

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _apply_miss(
        self, slot: int, epoch: int, slot_in_epoch: int, t: float
    ) -> None:
        self.chain.mark_missed(slot)
        self.randao.skip_slot()
        if self._current_epoch_metrics:
            self._current_epoch_metrics.missed_slots += 1


# ──────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────

def run_simulation(config: SimConfig) -> SimMetrics:
    """Create and run a single simulation; return metrics."""
    sim = PoSSimulator(config)
    return sim.run()
