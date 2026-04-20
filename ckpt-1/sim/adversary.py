"""
Adversary strategies for RANDAO manipulation in PoS blockchains.

Three strategies (inspired by the 2025-037 paper and the existing MDP scripts):

1. HonestStrategy  – baseline, always propose; no RANDAO manipulation.

2. GreedyWithhold  – at each adversarial slot, greedy one-step decision:
     compare expected proposer gain (in the target epoch) for reveal vs skip.
     Uses Monte Carlo sampling over unknown honest reveals.
     This models a computationally bounded adversary.

3. OptimalWithhold – enumerate all 2^k subsets of adversarial reveals and
     pick the one maximising proposer count in the target epoch.
     Full-information lookahead (adversary knows all honest reveals for the
     current epoch, i.e. the epoch's epoch_seed is shared/leaked).
     Models the worst-case adversary in World A.

4. ForkAdversary   – extends OptimalWithhold with an end-of-epoch fork attempt:
     if the adversary withheld blocks near the epoch tail, it tries to publish
     them as an alternate chain displacing honest blocks.

World A (RANDAO only):   target_epoch = current_epoch + 1
World B (RANDAO + VDF):  target_epoch = current_epoch + vdf_delay_epochs + 1
  In World B, the adversary has a vdf_compute_budget for VDF evaluations per
  epoch; with budget=1 they cannot grind alternate mixes.

Key insight from the 2025-037 paper:
  The adversary's attack vector is to select WHICH of their RANDAO reveals
  to include.  Each included reveal changes the XOR accumulator and therefore
  the final epoch mix, which seeds the next proposer schedule.  The adversary
  wants the final mix that maximises their count in the target epoch.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chain import ProposerSchedule, Validator, SLOTS_PER_EPOCH
from .randao import RANDAOState, honest_reveal, randao_contribution, _xor32


# ──────────────────────────────────────────────
# Adversary decision result
# ──────────────────────────────────────────────

@dataclass
class SlotDecision:
    """Decision made by the adversary for a single slot."""
    slot: int
    propose: bool          # True = reveal + propose block; False = miss
    reveal: Optional[bytes]  # the RANDAO reveal bytes (None if not proposing)
    strategy_used: str     # label for metrics


# ──────────────────────────────────────────────
# Helper: score a candidate RANDAO mix
# ──────────────────────────────────────────────

def _score_mix(
    candidate_mix: bytes,
    target_epoch: int,
    validators: List[Validator],
    vdf_pipeline=None,
    vdf_epoch_for_mix: int = -1,
) -> int:
    """
    Compute how many adversarial proposer slots result from candidate_mix
    being used as the seed for target_epoch's proposer schedule.

    In VDF mode, the mix is run through the VDF backend (unchecked, ignoring
    temporal delay — the adversary is doing lookahead planning, not real scheduling).
    """
    adv_ids = {v.validator_id for v in validators if v.is_adversarial}
    if vdf_pipeline is not None:
        # Use VDF output as the schedule seed
        result = vdf_pipeline.backend.compute(candidate_mix)
        seed_mix = result.output
    else:
        seed_mix = candidate_mix
    sched = ProposerSchedule(target_epoch, seed_mix, validators)
    return sched.count_adversarial(adv_ids)


# ──────────────────────────────────────────────
# Strategy base
# ──────────────────────────────────────────────

class AdversaryStrategy:
    """Abstract base; subclasses implement decide_epoch_reveals."""

    def __init__(self, validators: List[Validator], adv_seed: int) -> None:
        self.validators = validators
        self.adv_ids = {v.validator_id for v in validators if v.is_adversarial}
        self.adv_seed = adv_seed
        self.name = "base"

    def decide_epoch_reveals(
        self,
        epoch: int,
        adv_slots: List[int],           # slot-in-epoch indices for adversary
        adv_reveals: List[bytes],        # pre-committed reveals for those slots
        current_mix_at_epoch_start: bytes,
        honest_reveals_this_epoch: Dict[int, bytes],  # slot_in_epoch -> reveal
        target_epoch: int,
        rng: np.random.Generator,
        vdf_pipeline=None,
    ) -> Dict[int, bool]:
        """
        Return {slot_in_epoch: propose (True/False)} for adversarial slots.
        Honest slots are not included in the dict (always propose).
        """
        raise NotImplementedError

    # ── utility: compute final mix from a slot decision ────────────────────

    def _simulate_mix(
        self,
        base_mix: bytes,
        adv_slots: List[int],
        adv_reveals: List[bytes],
        propose_mask: int,              # bitmask over adv_slots
        honest_reveals_this_epoch: Dict[int, bytes],
    ) -> bytes:
        """
        Simulate the final epoch RANDAO mix given a bitmask of which adversarial
        reveals to include, plus all honest reveals.
        """
        # Collect all (slot, reveal) pairs and sort by slot
        pairs: List[Tuple[int, bytes]] = []
        for bit, (s, r) in enumerate(zip(adv_slots, adv_reveals)):
            if (propose_mask >> bit) & 1:
                pairs.append((s, r))
        for s, r in honest_reveals_this_epoch.items():
            pairs.append((s, r))
        pairs.sort(key=lambda x: x[0])

        mix = base_mix
        for _, r in pairs:
            mix = _xor32(mix, randao_contribution(r))
        return mix


# ──────────────────────────────────────────────
# Strategy 1: Honest (no attack)
# ──────────────────────────────────────────────

class HonestStrategy(AdversaryStrategy):
    """Adversary validators always propose (no RANDAO manipulation)."""

    def __init__(self, validators: List[Validator], adv_seed: int) -> None:
        super().__init__(validators, adv_seed)
        self.name = "honest"

    def decide_epoch_reveals(
        self,
        epoch: int,
        adv_slots: List[int],
        adv_reveals: List[bytes],
        current_mix_at_epoch_start: bytes,
        honest_reveals_this_epoch: Dict[int, bytes],
        target_epoch: int,
        rng: np.random.Generator,
        vdf_pipeline=None,
    ) -> Dict[int, bool]:
        return {s: True for s in adv_slots}


# ──────────────────────────────────────────────
# Strategy 2: Greedy withhold (one-step Monte Carlo)
# ──────────────────────────────────────────────

class GreedyWithhold(AdversaryStrategy):
    """
    At each adversarial slot (in chronological order), make a greedy
    propose/skip decision based on expected adversarial count in target_epoch.

    Uncertainty over future honest reveals is handled via Monte Carlo sampling
    (n_mc_samples random futures).  The adversary does NOT know honest reveals
    for future slots within the epoch.

    This models a computationally-bounded adversary and is the representative
    World-B strategy (limited lookahead, high temporal uncertainty).
    """

    def __init__(
        self,
        validators: List[Validator],
        adv_seed: int,
        n_mc_samples: int = 20,
        mc_lookahead_epochs: int = 0,
    ) -> None:
        super().__init__(validators, adv_seed)
        self.n_mc = n_mc_samples
        self.mc_lookahead = mc_lookahead_epochs   # extra epochs of honest reveals to sample
        self.name = "greedy"

    def decide_epoch_reveals(
        self,
        epoch: int,
        adv_slots: List[int],
        adv_reveals: List[bytes],
        current_mix_at_epoch_start: bytes,
        honest_reveals_this_epoch: Dict[int, bytes],
        target_epoch: int,
        rng: np.random.Generator,
        vdf_pipeline=None,
    ) -> Dict[int, bool]:
        decisions: Dict[int, bool] = {}
        running_mix = current_mix_at_epoch_start

        # Apply ALL honest reveals that precede the first adv slot (already known)
        # Then greedily decide each adv slot as we encounter it.
        all_slots_sorted = sorted(
            list(range(SLOTS_PER_EPOCH)),
        )
        adv_set = set(adv_slots)
        adv_iter = {s: r for s, r in zip(adv_slots, adv_reveals)}

        # Process honest slots strictly before the first adv slot
        for s in all_slots_sorted:
            if s in adv_set:
                # Adversary slot: make greedy decision
                my_reveal = adv_iter[s]
                propose = self._greedy_decide(
                    slot_in_epoch=s,
                    my_reveal=my_reveal,
                    running_mix=running_mix,
                    honest_reveals_seen=honest_reveals_this_epoch,
                    adv_slots_remaining=[ss for ss in adv_slots if ss > s],
                    adv_reveals_remaining=[adv_iter[ss] for ss in adv_slots if ss > s],
                    target_epoch=target_epoch,
                    rng=rng,
                    vdf_pipeline=vdf_pipeline,
                )
                decisions[s] = propose
                if propose:
                    running_mix = _xor32(running_mix, randao_contribution(my_reveal))
            else:
                # Honest slot — apply reveal if known
                hr = honest_reveals_this_epoch.get(s)
                if hr is not None:
                    running_mix = _xor32(running_mix, randao_contribution(hr))

        return decisions

    def _greedy_decide(
        self,
        slot_in_epoch: int,
        my_reveal: bytes,
        running_mix: bytes,
        honest_reveals_seen: Dict[int, bytes],
        adv_slots_remaining: List[int],
        adv_reveals_remaining: List[bytes],
        target_epoch: int,
        rng: np.random.Generator,
        vdf_pipeline=None,
    ) -> bool:
        """Monte Carlo estimate: propose vs skip this slot."""
        score_propose = 0
        score_skip = 0

        for _ in range(self.n_mc):
            # Sample random honest reveals for slots we haven't seen yet
            sampled_honest = {}
            for s in range(SLOTS_PER_EPOCH):
                if s not in honest_reveals_seen and s not in set(adv_slots_remaining + [slot_in_epoch]):
                    # Unknown future honest slot: sample random reveal
                    rand_bytes = rng.bytes(32)
                    sampled_honest[s] = rand_bytes

            combined_honest = {**honest_reveals_seen, **sampled_honest}

            # For remaining adversarial slots: assume they all propose (optimistic)
            remaining_pairs = list(zip(adv_slots_remaining, adv_reveals_remaining))

            # Option A: propose (include my reveal)
            mix_propose = _xor32(running_mix, randao_contribution(my_reveal))
            for sr, rr in remaining_pairs:
                mix_propose = _xor32(mix_propose, randao_contribution(rr))
            for s, hr in combined_honest.items():
                if s > slot_in_epoch:
                    mix_propose = _xor32(mix_propose, randao_contribution(hr))

            # Option B: skip (exclude my reveal)
            mix_skip = running_mix
            for sr, rr in remaining_pairs:
                mix_skip = _xor32(mix_skip, randao_contribution(rr))
            for s, hr in combined_honest.items():
                if s > slot_in_epoch:
                    mix_skip = _xor32(mix_skip, randao_contribution(hr))

            score_propose += _score_mix(mix_propose, target_epoch, self.validators, vdf_pipeline)
            score_skip    += _score_mix(mix_skip,    target_epoch, self.validators, vdf_pipeline)

        return score_propose >= score_skip


# ──────────────────────────────────────────────
# Strategy 3: Optimal withhold (full enumeration)
# ──────────────────────────────────────────────

class OptimalWithhold(AdversaryStrategy):
    """
    Enumerate all 2^k subsets of adversarial reveals (k = adversarial slots in
    the epoch) and pick the subset maximising adversarial count in target_epoch.

    Full-information: the adversary knows all honest reveals for the epoch
    (epoch_seed is considered "leaked" or the adversary can predict them from
    the deterministic simulation seed).

    This is the worst-case adversary for World A and approximates the paper's
    "selfish mixing + lookahead" strategy.

    vdf_compute_budget: how many VDF evaluations are allowed when scoring mixes.
      budget = 2^k → try all options (same power as World A, unconstrained)
      budget = 1   → adversary is VDF-constrained (World B mode)
    """

    def __init__(
        self,
        validators: List[Validator],
        adv_seed: int,
        vdf_compute_budget: int = 2**30,
    ) -> None:
        super().__init__(validators, adv_seed)
        self.vdf_compute_budget = vdf_compute_budget
        self.name = "optimal"

    def decide_epoch_reveals(
        self,
        epoch: int,
        adv_slots: List[int],
        adv_reveals: List[bytes],
        current_mix_at_epoch_start: bytes,
        honest_reveals_this_epoch: Dict[int, bytes],
        target_epoch: int,
        rng: np.random.Generator,
        vdf_pipeline=None,
    ) -> Dict[int, bool]:
        k = len(adv_slots)
        if k == 0:
            return {}

        # Compute base mix: genesis mix XOR all honest reveals for this epoch
        base_mix = current_mix_at_epoch_start
        for s in sorted(honest_reveals_this_epoch):
            base_mix = _xor32(base_mix, randao_contribution(honest_reveals_this_epoch[s]))

        best_score = -1
        best_mask = (1 << k) - 1   # default: reveal all

        n_masks = 1 << k

        # Scoring method:
        #   budget=1 (VDF-constrained): score candidate mixes by RANDAO-direct hash,
        #     NOT by VDF output. This models the adversary who can't enumerate VDF
        #     evaluations — they pick the best RANDAO-direct mix, but the actual
        #     schedule is determined by VDF(mix) which doesn't align with their heuristic.
        #   budget≥n_masks (unconstrained): score by VDF output — fully optimal.
        vdf_constrained = (vdf_pipeline is not None and self.vdf_compute_budget < n_masks)
        score_vdf = None if vdf_constrained else vdf_pipeline

        for mask in range(n_masks):
            candidate_mix = base_mix
            for bit in range(k):
                if (mask >> bit) & 1:
                    candidate_mix = _xor32(
                        candidate_mix, randao_contribution(adv_reveals[bit])
                    )
            # Score with or without VDF depending on budget
            score = _score_mix(
                candidate_mix, target_epoch, self.validators, score_vdf
            )
            if score > best_score:
                best_score = score
                best_mask = mask

        return {s: bool((best_mask >> bit) & 1) for bit, s in enumerate(adv_slots)}


# ──────────────────────────────────────────────
# Strategy 4: Fork adversary (withhold + end-of-epoch fork)
# ──────────────────────────────────────────────

class ForkAdversary(OptimalWithhold):
    """
    Extends OptimalWithhold with an end-of-epoch fork attempt.

    Inspired by the paper's "ex-ante forking" option:
      1. During epoch e, the adversary withholds their last `fork_depth` blocks.
      2. They monitor the RANDAO mix as honest reveals accumulate.
      3. At epoch end, if the reveal of their withheld blocks would yield a
         better schedule than the canonical chain, they publish the alternate
         chain (fork attempt).

    The fork is modelled as a call to ChainState.attempt_fork().
    """

    def __init__(
        self,
        validators: List[Validator],
        adv_seed: int,
        fork_depth: int = 3,
        vdf_compute_budget: int = 2**30,
    ) -> None:
        super().__init__(validators, adv_seed, vdf_compute_budget)
        self.fork_depth = fork_depth
        self.name = "fork"

    # Fork metadata stored per epoch (accessed by simulator at EPOCH_END)
    def fork_plan(
        self,
        epoch: int,
        adv_slots: List[int],
        adv_reveals: List[bytes],
        decisions: Dict[int, bool],
    ) -> Optional[List[int]]:
        """
        Return the list of withheld slot indices that the adversary may
        broadcast as a fork at epoch end.  Returns None if no fork planned.
        """
        # Slots where adversary decided to skip (withheld, not just missed)
        withheld = [s for s in adv_slots if not decisions.get(s, True)]
        if not withheld:
            return None
        # Only attempt fork on the last `fork_depth` withheld slots
        return withheld[-self.fork_depth:]


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def make_adversary(
    strategy_name: str,
    validators: List[Validator],
    adv_seed: int,
    vdf_compute_budget: int = 2**30,
    n_mc_samples: int = 20,
) -> AdversaryStrategy:
    """Construct an adversary strategy by name."""
    if strategy_name == "honest":
        return HonestStrategy(validators, adv_seed)
    elif strategy_name == "greedy":
        return GreedyWithhold(validators, adv_seed, n_mc_samples=n_mc_samples)
    elif strategy_name == "optimal":
        return OptimalWithhold(validators, adv_seed, vdf_compute_budget=vdf_compute_budget)
    elif strategy_name == "fork":
        return ForkAdversary(validators, adv_seed, vdf_compute_budget=vdf_compute_budget)
    else:
        raise ValueError(f"Unknown adversary strategy: {strategy_name!r}")
