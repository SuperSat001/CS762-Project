"""
PART 4: RANDAO accumulator (strict spec).

State: R_e  (256-bit integer)
Init:  R_0  = 0

Per epoch-end:
    R_e = R_{e-1} XOR (XOR of randao_reveals of canonical blocks in epoch)

Proposer selection for epoch e+2:
    seed   = SHA-256( e.to_bytes || R_e )
    shuffle validators deterministically with that seed
    assign 32 proposers (one per slot)

CRITICAL:
* Only canonical blocks contribute (reorged / missed slots = 0 reveal).
* VDF wraps the seed computation — see vdf.py.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Tuple

SLOTS_PER_EPOCH = 32
MASK_256 = (1 << 256) - 1


def _xor_reveals(reveals: List[int]) -> int:
    result = 0
    for r in reveals:
        result ^= r
    return result & MASK_256


def _epoch_seed(epoch: int, randao_value: int) -> bytes:
    """Deterministic seed bytes from epoch number and RANDAO value."""
    data = epoch.to_bytes(8, "big") + randao_value.to_bytes(32, "big")
    return hashlib.sha256(data).digest()


def compute_proposer_schedule(
    epoch: int,
    randao_value: int,
    validator_ids: List[int],
) -> List[int]:
    """
    Shuffle validators with the epoch seed and return SLOTS_PER_EPOCH
    proposer ids (one per slot in the epoch).
    """
    seed = _epoch_seed(epoch, randao_value)
    rng = random.Random(seed)
    shuffled = list(validator_ids)
    rng.shuffle(shuffled)
    # wrap around if fewer validators than slots
    schedule: List[int] = []
    for i in range(SLOTS_PER_EPOCH):
        schedule.append(shuffled[i % len(shuffled)])
    return schedule


class RandaoState:
    """Tracks per-epoch RANDAO accumulator and proposer schedules."""

    def __init__(self, validator_ids: List[int]) -> None:
        self._validator_ids = list(validator_ids)
        # epoch → R_e
        self._epoch_values: Dict[int, int] = {0: 0}
        # epoch → list[proposer_id per slot]  (length = SLOTS_PER_EPOCH)
        self._schedules: Dict[int, Optional[List[int]]] = {}
        # pending VDF inputs: epoch → raw R_e (before VDF)
        self._pending_vdf: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Epoch finalisation (called by simulator at epoch end)
    # ------------------------------------------------------------------

    def finalize_epoch(
        self,
        epoch: int,
        canonical_reveals: List[int],
        use_vdf: bool = False,
    ) -> int:
        """
        Compute R_epoch from R_{epoch-1} XOR all canonical reveals.

        If use_vdf is True the raw R_e is stored as pending; the proposer
        schedule is NOT yet available (returns -1 as sentinel).
        If use_vdf is False the schedule for epoch+2 is computed immediately.

        Returns the new R_e value.
        """
        prev = self._epoch_values.get(epoch - 1, 0)
        new_r = (prev ^ _xor_reveals(canonical_reveals)) & MASK_256
        self._epoch_values[epoch] = new_r

        if use_vdf:
            # Schedule will be set when VDF_COMPLETE fires
            self._pending_vdf[epoch] = new_r
            self._schedules[epoch + 2] = None          # not yet available
        else:
            self._schedules[epoch + 2] = compute_proposer_schedule(
                epoch + 2, new_r, self._validator_ids
            )

        return new_r

    def complete_vdf(self, epoch: int, vdf_output: int) -> None:
        """
        Called when VDF_COMPLETE fires for this epoch.
        Computes the proposer schedule for epoch+2 using the VDF output.
        """
        self._schedules[epoch + 2] = compute_proposer_schedule(
            epoch + 2, vdf_output, self._validator_ids
        )
        self._pending_vdf.pop(epoch, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_randao(self, epoch: int) -> int:
        return self._epoch_values.get(epoch, 0)

    def get_schedule(self, epoch: int) -> Optional[List[int]]:
        """
        Returns the proposer schedule for `epoch`, or None if not yet
        available (VDF pending) or epoch hasn't been computed.
        """
        return self._schedules.get(epoch)

    def proposer_for_slot(self, slot: int) -> Optional[int]:
        epoch = slot // SLOTS_PER_EPOCH
        slot_index = slot % SLOTS_PER_EPOCH
        schedule = self.get_schedule(epoch)
        if schedule is None:
            return None
        return schedule[slot_index]

    def is_schedule_ready(self, epoch: int) -> bool:
        sched = self._schedules.get(epoch)
        return sched is not None

    def pending_vdf_epochs(self) -> List[int]:
        return list(self._pending_vdf.keys())
