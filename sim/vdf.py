"""
Verifiable Delay Function (VDF) module.

Two backends:
  1. FastSimVDF  – mock VDF using iterated SHA-256 (sequential but fast).
     Deterministic, verifiable, captures temporal-causality semantics.
  2. ChiaVDFAdapter – thin wrapper for the real chiavdf library (optional).

VDFPipeline manages the lifecycle inside the DES:
  • submit(epoch, mix, available_at): pre-compute result; mark it unavailable
    until simulated time >= available_at.
  • get_result(epoch, now): returns result iff now >= available_at.
  • Strict temporal causality: proposer schedules MUST call get_result and
    handle None (not yet available) rather than reading results early.

Security model:
  In World A the adversary can enumerate 2^k RANDAO mixes and instantly
  evaluate which gives the best schedule.  In World B, evaluating each
  candidate mix requires running VDF(mix, T) which takes T sequential steps.
  The simulation models this with a "vdf_compute_budget" parameter: the number
  of VDF evaluations the adversary can afford per decision point.  With
  budget=1 the adversary is constrained to pick the FIRST (non-optimized)
  mix; with budget=2^k it is equivalent to World A.
"""
from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

VDF_DOMAIN = b"pos_des_vdf_v1"


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class VDFResult:
    input_data: bytes
    output: bytes
    proof: bytes      # short commitment (not a real proof in the mock)
    iterations: int   # number of sequential steps performed


# ──────────────────────────────────────────────
# Abstract backend
# ──────────────────────────────────────────────

class VDFBackend(ABC):
    @abstractmethod
    def compute(self, input_data: bytes, iterations: int) -> VDFResult:
        ...

    @abstractmethod
    def verify(self, result: VDFResult) -> bool:
        ...


# ──────────────────────────────────────────────
# Backend 1: Fast simulated VDF (mock)
# ──────────────────────────────────────────────

class FastSimVDF(VDFBackend):
    """
    Mock VDF: output = SHA256^T(input).

    • Deterministic and verifiable (re-run the chain).
    • NOT a real VDF – just a sequential hash chain with no delay in wall time.
    • The *simulated* delay is enforced by VDFPipeline.available_at, not by
      actual wall-clock computation.
    • iterations=256 by default (fast even for large simulations).
    """

    def compute(self, input_data: bytes, iterations: int = 256) -> VDFResult:
        x = input_data + VDF_DOMAIN
        for _ in range(iterations):
            x = hashlib.sha256(x).digest()
        proof = hashlib.sha256(b"proof:" + x + input_data).digest()
        return VDFResult(
            input_data=input_data,
            output=x,
            proof=proof,
            iterations=iterations,
        )

    def verify(self, result: VDFResult) -> bool:
        recomputed = self.compute(result.input_data, result.iterations)
        return recomputed.output == result.output and recomputed.proof == result.proof


# ──────────────────────────────────────────────
# Backend 2: chiavdf adapter (optional)
# ──────────────────────────────────────────────

class ChiaVDFAdapter(VDFBackend):
    """
    Adapter for the real Chia VDF (https://github.com/Chia-Network/chiavdf).

    Requires: `pip install chiavdf`

    Falls back gracefully to FastSimVDF if the library is not installed.
    """

    def __init__(self) -> None:
        try:
            import chiavdf  # type: ignore
            self._chiavdf = chiavdf
            self._available = True
        except ImportError:
            self._available = False
            self._fallback = FastSimVDF()

    def compute(self, input_data: bytes, iterations: int = 1_000_000) -> VDFResult:
        if not self._available:
            return self._fallback.compute(input_data, min(iterations, 256))
        # chiavdf expects a discriminant and challenge; we derive them from input.
        discriminant = int.from_bytes(
            hashlib.sha256(b"disc:" + input_data).digest()[:16], "big"
        ) | 1
        challenge = input_data[:32].ljust(32, b"\x00")
        output, proof = self._chiavdf.prove(
            challenge, discriminant.to_bytes(16, "big"), iterations
        )
        return VDFResult(
            input_data=input_data,
            output=output,
            proof=proof,
            iterations=iterations,
        )

    def verify(self, result: VDFResult) -> bool:
        if not self._available:
            return self._fallback.verify(result)
        discriminant = int.from_bytes(
            hashlib.sha256(b"disc:" + result.input_data).digest()[:16], "big"
        ) | 1
        challenge = result.input_data[:32].ljust(32, b"\x00")
        return bool(
            self._chiavdf.verify(
                challenge,
                discriminant.to_bytes(16, "big"),
                result.output,
                result.proof,
                result.iterations,
            )
        )


# ──────────────────────────────────────────────
# VDF pipeline (used by the DES simulator)
# ──────────────────────────────────────────────

class VDFPipeline:
    """
    Manages VDF computations inside the DES.

    Temporal causality enforcement:
      get_result(epoch, current_time) returns None if current_time < available_at[epoch].
      The simulator must use this check before using VDF output for scheduling.

    Adversary compute budget:
      vdf_compute_budget controls how many VDF evaluations the adversary can
      perform per epoch when choosing which RANDAO mix to target.
        budget=1  → adversary cannot grind (real-VDF–like constraint)
        budget=∞  → adversary can try all 2^k options (same as World A)
    """

    def __init__(
        self,
        backend: VDFBackend,
        delay_slots: int,
        vdf_compute_budget: int = 1,
    ) -> None:
        self.backend = backend
        self.delay_slots = delay_slots
        self.vdf_compute_budget = vdf_compute_budget
        self._results: Dict[int, VDFResult] = {}
        self._available_at: Dict[int, float] = {}

    def submit(
        self, epoch: int, randao_mix: bytes, epoch_end_time: float
    ) -> VDFResult:
        """
        Pre-compute VDF result for epoch's RANDAO mix.
        Result becomes available at epoch_end_time + delay_slots (sim time).
        """
        result = self.backend.compute(randao_mix)
        self._results[epoch] = result
        self._available_at[epoch] = epoch_end_time + self.delay_slots
        return result

    def get_result(self, epoch: int, current_time: float) -> Optional[VDFResult]:
        """
        Return VDF result for epoch iff current_time >= available_at[epoch].
        Returns None if not yet available — caller must handle this.
        """
        if epoch not in self._results:
            return None
        if current_time < self._available_at[epoch]:
            return None  # strict temporal causality
        return self._results[epoch]

    def get_result_unchecked(self, epoch: int) -> Optional[VDFResult]:
        """Return VDF result regardless of availability (for adversary lookahead)."""
        return self._results.get(epoch)

    def is_available(self, epoch: int, current_time: float) -> bool:
        return self.get_result(epoch, current_time) is not None

    def available_at(self, epoch: int) -> Optional[float]:
        return self._available_at.get(epoch)

    def adversary_can_evaluate(self, budget_used: int) -> bool:
        """Check if adversary has remaining VDF compute budget."""
        return budget_used < self.vdf_compute_budget


def make_vdf_pipeline(
    use_real_vdf: bool = False,
    delay_slots: int = 64,
    vdf_compute_budget: int = 1,
) -> VDFPipeline:
    """Factory: construct the right backend based on flags."""
    if use_real_vdf:
        backend: VDFBackend = ChiaVDFAdapter()
    else:
        backend = FastSimVDF()
    return VDFPipeline(backend, delay_slots, vdf_compute_budget)
