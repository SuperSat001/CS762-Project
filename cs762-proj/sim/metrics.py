"""
PART 8: Metrics collection.

Tracks:
* adversarial_blocks / total_blocks
* number of forks (slots where > 1 child added)
* number of reorgs (previous head no longer on canonical chain)
* missed slots
* proposer distribution vs stake weight
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional


class Metrics:
    def __init__(self) -> None:
        self.total_blocks: int = 0
        self.adversarial_blocks: int = 0
        self.honest_blocks: int = 0

        self.forks: int = 0            # times a slot produced > 1 child
        self.reorgs: int = 0           # head changes to non-child
        self.missed_slots: int = 0

        # proposer_id → blocks proposed (canonical)
        self.canonical_proposals: Dict[int, int] = defaultdict(int)
        # proposer_id → stake weight (set externally)
        self.stake_weights: Dict[int, int] = {}

        self._prev_head_id: Optional[int] = None
        self._slots_with_multiple_children: set = set()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_block(self, proposer_id: int, is_adversarial: bool) -> None:
        self.total_blocks += 1
        if is_adversarial:
            self.adversarial_blocks += 1
        else:
            self.honest_blocks += 1

    def record_missed_slot(self) -> None:
        self.missed_slots += 1

    def record_fork(self, parent_id: int) -> None:
        """Call when a second (or more) child is added to a block."""
        if parent_id not in self._slots_with_multiple_children:
            self._slots_with_multiple_children.add(parent_id)
            self.forks += 1

    def record_head_update(self, new_head_id: int, new_head_parent_id: Optional[int]) -> None:
        """Detect reorgs: new head is NOT a child of previous head."""
        if self._prev_head_id is not None:
            if new_head_parent_id != self._prev_head_id:
                # New head jumps to a non-child → reorg
                self.reorgs += 1
        self._prev_head_id = new_head_id

    def record_canonical_proposal(self, proposer_id: int) -> None:
        self.canonical_proposals[proposer_id] += 1

    def set_stake_weights(self, weights: Dict[int, int]) -> None:
        self.stake_weights = dict(weights)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def adversarial_ratio(self) -> float:
        if self.total_blocks == 0:
            return 0.0
        return self.adversarial_blocks / self.total_blocks

    def proposer_distribution(self) -> Dict[int, float]:
        """Fraction of canonical proposals per validator."""
        total = sum(self.canonical_proposals.values())
        if total == 0:
            return {}
        return {v: c / total for v, c in self.canonical_proposals.items()}

    def stake_fraction(self, validator_id: int) -> float:
        total_stake = sum(self.stake_weights.values())
        if total_stake == 0:
            return 0.0
        return self.stake_weights.get(validator_id, 0) / total_stake

    def summary(self) -> dict:
        return {
            "total_blocks":        self.total_blocks,
            "adversarial_blocks":  self.adversarial_blocks,
            "honest_blocks":       self.honest_blocks,
            "adversarial_ratio":   round(self.adversarial_ratio(), 4),
            "forks":               self.forks,
            "reorgs":              self.reorgs,
            "missed_slots":        self.missed_slots,
            "canonical_proposals": dict(self.canonical_proposals),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"Metrics(total={s['total_blocks']}, "
            f"adv={s['adversarial_blocks']} ({s['adversarial_ratio']:.1%}), "
            f"forks={s['forks']}, reorgs={s['reorgs']}, "
            f"missed={s['missed_slots']})"
        )
