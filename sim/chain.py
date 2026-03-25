"""
Blockchain chain state: validators, blocks, proposer scheduling, fork choice.

Design choices (simplified from full Ethereum spec):
  • Equal stake per validator (can be overridden via stake parameter).
  • Proposer selection: deterministic from RANDAO mix + slot index via
    stake-weighted shuffle (Ethereum compute_proposer_index analogue).
  • Canonical chain = linear sequence of blocks; missed slots recorded separately.
  • Fork choice: "longest chain wins" — adversary can attempt a short reorg
    by withholding blocks and then broadcasting a heavier alternate chain.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SLOTS_PER_EPOCH: int = 32
SLOT_DURATION: float = 1.0   # 1 sim-time unit per slot (≈12 s in mainnet)
PROP_DELAY: float = 0.05     # time inside a slot before block is published


# ──────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────

@dataclass
class Validator:
    validator_id: int
    is_adversarial: bool
    stake: float = 1.0


# ──────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────

@dataclass
class Block:
    """A single canonical block (proposed or missed)."""
    slot: int
    epoch: int
    proposer_id: int
    parent_hash: bytes
    randao_reveal: Optional[bytes]   # None if the slot was missed
    is_adversarial: bool
    is_missed: bool = False
    # Computed on creation:
    block_hash: bytes = field(init=False, default=b"")

    def __post_init__(self) -> None:
        if self.is_missed:
            self.block_hash = b"\x00" * 32
        else:
            data = (
                struct.pack(">QQQ", self.slot, self.epoch, self.proposer_id)
                + self.parent_hash
                + (self.randao_reveal or b"")
            )
            self.block_hash = hashlib.sha256(data).digest()


# ──────────────────────────────────────────────
# Proposer schedule
# ──────────────────────────────────────────────

class ProposerSchedule:
    """
    Maps slot-within-epoch → validator_id for one epoch.

    Algorithm (stake-weighted):
      For each slot s:
        h = SHA256(randao_mix ‖ s_bytes)
        target = h mod total_stake_units
        walk validators by cumulative stake → first validator whose cumulative
        stake exceeds target is the proposer.

    This is deterministic and depends only on the RANDAO mix for that epoch.
    """

    def __init__(
        self,
        epoch: int,
        randao_mix: bytes,
        validators: List[Validator],
    ) -> None:
        self.epoch = epoch
        self.randao_mix = randao_mix
        self._schedule: Dict[int, int] = {}
        self._build(validators)

    def _build(self, validators: List[Validator]) -> None:
        total_stake_units = sum(int(v.stake * 1000) for v in validators)
        for s in range(SLOTS_PER_EPOCH):
            data = self.randao_mix + struct.pack(">Q", s)
            h = int.from_bytes(hashlib.sha256(data).digest(), "big")
            target = h % total_stake_units
            cumul = 0
            chosen = validators[-1].validator_id
            for v in validators:
                cumul += int(v.stake * 1000)
                if cumul > target:
                    chosen = v.validator_id
                    break
            self._schedule[s] = chosen

    def proposer_for(self, slot: int) -> int:
        return self._schedule[slot % SLOTS_PER_EPOCH]

    def adversarial_slots(self, adv_ids: set[int]) -> List[int]:
        """Return slot-within-epoch indices assigned to adversarial validators."""
        return [s for s, vid in self._schedule.items() if vid in adv_ids]

    def count_adversarial(self, adv_ids: set[int]) -> int:
        return sum(1 for vid in self._schedule.values() if vid in adv_ids)

    def slot_map(self) -> Dict[int, int]:
        """Return full slot→validator_id mapping (copy)."""
        return dict(self._schedule)


# ──────────────────────────────────────────────
# Chain state
# ──────────────────────────────────────────────

class ChainState:
    """
    Maintains the canonical chain and proposer schedules for all epochs.

    Key operations:
      • set_schedule(epoch, mix)  – build proposer schedule from RANDAO mix
      • add_block(block)          – append a block to the canonical chain
      • mark_missed(slot)         – record a missed slot (no proposer)
      • attempt_fork(...)         – adversary tries to replace tail blocks
    """

    def __init__(
        self,
        validators: List[Validator],
        genesis_mix: bytes,
    ) -> None:
        self.validators = validators
        self.n_validators = len(validators)
        self._val_map: Dict[int, Validator] = {v.validator_id: v for v in validators}
        self._adv_ids: set[int] = {v.validator_id for v in validators if v.is_adversarial}

        # Genesis block (slot 0 placeholder)
        self._head_hash: bytes = b"\x00" * 32
        self._head_slot: int = 0

        # Slot → canonical block (missed slots stored with is_missed=True)
        self._blocks: Dict[int, Block] = {}

        # Proposer schedules indexed by epoch
        self._schedules: Dict[int, ProposerSchedule] = {}

        # Build epoch 0 schedule from genesis mix
        self._schedules[0] = ProposerSchedule(0, genesis_mix, validators)
        self._genesis_mix = genesis_mix

        # Statistics
        self.missed_slots: List[int] = []
        self.reorg_count: int = 0
        self.forked_honest_count: int = 0  # honest blocks removed by reorgs

    # ── schedule management ───────────────────

    def set_schedule(self, epoch: int, randao_mix: bytes) -> ProposerSchedule:
        sched = ProposerSchedule(epoch, randao_mix, self.validators)
        self._schedules[epoch] = sched
        return sched

    def get_schedule(self, epoch: int) -> Optional[ProposerSchedule]:
        return self._schedules.get(epoch)

    def proposer_for(self, slot: int) -> int:
        epoch = slot // SLOTS_PER_EPOCH
        sched = self._schedules.get(epoch)
        if sched is None:
            raise ValueError(f"No proposer schedule for epoch {epoch} (slot {slot})")
        return sched.proposer_for(slot)

    def is_adversarial_proposer(self, slot: int) -> bool:
        return self.proposer_for(slot) in self._adv_ids

    # ── block operations ──────────────────────

    def add_block(self, block: Block) -> None:
        """Add a block to the canonical chain."""
        self._blocks[block.slot] = block
        if block.slot > self._head_slot:
            self._head_slot = block.slot
            self._head_hash = block.block_hash

    def mark_missed(self, slot: int) -> None:
        """Record that slot was missed (no block proposed)."""
        self.missed_slots.append(slot)
        epoch = slot // SLOTS_PER_EPOCH
        # Proposer lookup may fail during VDF warmup when schedule not yet built
        try:
            proposer = self.proposer_for(slot)
            is_adv = proposer in self._adv_ids
        except ValueError:
            proposer = -1
            is_adv = False
        missed_block = Block(
            slot=slot,
            epoch=epoch,
            proposer_id=proposer,
            parent_hash=self._head_hash,
            randao_reveal=None,
            is_adversarial=is_adv,
            is_missed=True,
        )
        self._blocks[slot] = missed_block

    def get_block(self, slot: int) -> Optional[Block]:
        return self._blocks.get(slot)

    def head_slot(self) -> int:
        return self._head_slot

    def head_hash(self) -> bytes:
        return self._head_hash

    # ── fork / reorg ──────────────────────────

    def attempt_fork(
        self,
        fork_start_slot: int,
        alternate_blocks: List[Block],
        current_time: float,
    ) -> bool:
        """
        Adversary publishes alternate blocks starting at fork_start_slot.

        Fork succeeds (heaviest-chain rule) if the alternate sub-chain has
        at least as many non-missed blocks as the canonical sub-chain it
        replaces.

        Returns True if the fork was accepted.
        """
        fork_end_slot = fork_start_slot + len(alternate_blocks) - 1

        # Count non-missed blocks in canonical range
        canonical_weight = sum(
            1
            for s in range(fork_start_slot, fork_end_slot + 1)
            if self._blocks.get(s) and not self._blocks[s].is_missed
        )
        # Count non-missed blocks in alternate chain
        alternate_weight = sum(1 for b in alternate_blocks if not b.is_missed)

        if alternate_weight >= canonical_weight:
            # Record honest blocks displaced
            for s in range(fork_start_slot, fork_end_slot + 1):
                old = self._blocks.get(s)
                if old and not old.is_missed and not old.is_adversarial:
                    self.forked_honest_count += 1

            # Apply alternate chain
            for b in alternate_blocks:
                self._blocks[b.slot] = b

            self.reorg_count += 1
            return True

        return False

    # ── query helpers ─────────────────────────

    def chain_weight(self, up_to_slot: int) -> int:
        """Number of non-missed blocks from slot 1 to up_to_slot (inclusive)."""
        return sum(
            1
            for s in range(1, up_to_slot + 1)
            if self._blocks.get(s) and not self._blocks[s].is_missed
        )

    def adversarial_blocks_in_range(self, lo: int, hi: int) -> int:
        """Count adversarial (non-missed) blocks in [lo, hi]."""
        return sum(
            1
            for s in range(lo, hi + 1)
            if self._blocks.get(s)
            and not self._blocks[s].is_missed
            and self._blocks[s].is_adversarial
        )

    def total_blocks_in_range(self, lo: int, hi: int) -> int:
        return sum(
            1
            for s in range(lo, hi + 1)
            if self._blocks.get(s) and not self._blocks[s].is_missed
        )

    def adversarial_fraction_so_far(self, current_slot: int) -> float:
        total = self.chain_weight(current_slot)
        adv = self.adversarial_blocks_in_range(1, current_slot)
        return adv / total if total > 0 else 0.0

    def missed_slot_rate(self, lo: int, hi: int) -> float:
        n = hi - lo + 1
        missed = sum(
            1
            for s in range(lo, hi + 1)
            if self._blocks.get(s) and self._blocks[s].is_missed
        )
        return missed / n if n > 0 else 0.0
