"""
RANDAO reveal and accumulator logic.

Protocol (Ethereum-simplified):
  • Each validator v has a "secret" keyed by (validator_id, epoch).
    For honest validators the secret is drawn fresh from the simulation RNG
    at epoch start (unknown to adversary in advance).
    For adversarial validators the secret is deterministic from adv_seed
    (known in advance by the adversary).

  • reveal(v, e) = H(secret_v_e)          — what the validator broadcasts
  • contribution = H(reveal(v, e))        — XOR'd into the accumulator
  • mix update:  mix ← mix XOR contribution

  • At end of epoch e:  mix[e] = fold over all contributing slots
    (missed slots contribute nothing — this is the manipulation vector).

  • The final mix[e] is used to seed proposer schedule for epoch e+1.
    In VDF mode, mix[e] → VDF → available D epochs later → used for epoch e+D+1.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict, Optional

import numpy as np

RANDAO_DOMAIN = b"pos_des_randao_v1"


# ──────────────────────────────────────────────
# Low-level hash helpers
# ──────────────────────────────────────────────

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _xor32(a: bytes, b: bytes) -> bytes:
    """XOR two 32-byte values."""
    return bytes(x ^ y for x, y in zip(a, b))


# ──────────────────────────────────────────────
# Reveal generation
# ──────────────────────────────────────────────

def adversary_reveal(adv_seed: int, validator_id: int, epoch: int) -> bytes:
    """Deterministic reveal for adversarial validators (pre-committed)."""
    packed = struct.pack(">QQQ", adv_seed, validator_id, epoch)
    return _sha256(packed + RANDAO_DOMAIN)


def honest_reveal(epoch_seed: int, validator_id: int, epoch: int) -> bytes:
    """
    Honest validator's reveal.  epoch_seed is fresh per epoch (drawn from the
    global simulation RNG at epoch start, unknown to the adversary in advance).
    """
    packed = struct.pack(">QQQ", epoch_seed, validator_id, epoch)
    return _sha256(packed + RANDAO_DOMAIN + b"honest")


def randao_contribution(reveal: bytes) -> bytes:
    """The 32-byte value XOR'd into the accumulator for a given reveal."""
    return _sha256(reveal + b"contrib")


# ──────────────────────────────────────────────
# Epoch-level RANDAO accumulator
# ──────────────────────────────────────────────

class RANDAOState:
    """
    Tracks RANDAO mixes across all epochs.

    mix[-1]  = genesis mix (from seed)
    mix[e]   = final mix at end of epoch e
    """

    def __init__(self, genesis_seed: int) -> None:
        genesis_mix = _sha256(struct.pack(">Q", genesis_seed) + b"genesis_randao")
        self._mixes: Dict[int, bytes] = {-1: genesis_mix}
        self._current_epoch: int = -1
        self._current_mix: bytes = genesis_mix

    # ── epoch lifecycle ───────────────────────

    def begin_epoch(self, epoch: int) -> None:
        """Reset accumulator to the previous epoch's finalized mix."""
        self._current_epoch = epoch
        self._current_mix = self._mixes.get(epoch - 1, self._mixes[-1])

    def apply_reveal(self, reveal: bytes) -> None:
        """Incorporate one validator's RANDAO reveal into the running mix."""
        contrib = randao_contribution(reveal)
        self._current_mix = _xor32(self._current_mix, contrib)

    def skip_slot(self) -> None:
        """A slot was missed; accumulator unchanged."""
        pass  # intentionally empty – documenting the no-op

    def finalize_epoch(self, epoch: int) -> bytes:
        """Lock in the epoch's final mix and return it."""
        self._mixes[epoch] = self._current_mix
        return self._current_mix

    # ── query ─────────────────────────────────

    def get_mix(self, epoch: int) -> Optional[bytes]:
        """Return the finalized mix for epoch (None if not yet finalized)."""
        return self._mixes.get(epoch)

    def current_mix(self) -> bytes:
        return self._current_mix

    # ── adversary helper ──────────────────────

    def simulate_final_mix(
        self,
        current_mix: bytes,
        adv_reveals_to_include: list[bytes],
        honest_reveals_remaining: list[bytes],
    ) -> bytes:
        """
        Compute the hypothetical final epoch mix given:
          - current_mix: accumulator state so far in the epoch
          - adv_reveals_to_include: subset of adversarial reveals to XOR in
          - honest_reveals_remaining: expected honest reveals for remaining slots

        Used by the adversary to score candidate strategies.
        """
        mix = current_mix
        for r in adv_reveals_to_include:
            mix = _xor32(mix, randao_contribution(r))
        for r in honest_reveals_remaining:
            mix = _xor32(mix, randao_contribution(r))
        return mix
