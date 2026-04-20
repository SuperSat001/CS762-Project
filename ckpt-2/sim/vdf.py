"""
PART 7: VDF (Verifiable Delay Function) integration.

Simplified model:
* eval(input, current_time) → (output, ready_time)
  where ready_time = current_time + delay_slots

* output = SHA-256^t(input)  (iterated hash as VDF stand-in)
  This preserves the "delay" semantics: output is deterministic but
  requires sequential computation proportional to t.

* verify(input, output, t) checks the iterated hash.

The real VDF (e.g. RSA repeated-squaring) is replaced by iterated
SHA-256 to keep the simulator self-contained.
"""

from __future__ import annotations

import hashlib


def _iterated_hash(value: int, iterations: int) -> int:
    """SHA-256 applied `iterations` times to the 32-byte big-endian
    representation of `value`.  Returns a 256-bit integer."""
    current = value.to_bytes(32, "big")
    for _ in range(iterations):
        current = hashlib.sha256(current).digest()
    return int.from_bytes(current, "big")


class VDF:
    """
    Simplified VDF.

    Parameters
    ----------
    security_param : int
        λ in the paper — not used in this simplified model beyond logging.
    delay_slots : int
        t — number of slots the VDF takes to evaluate.
        In the real protocol this corresponds to ~1 epoch worth of time so
        adversaries cannot learn future proposer assignments before the
        current epoch ends.
    iterations : int
        Number of hash iterations used to simulate sequential work.
        Default 1 keeps tests fast; a real deployment would use a large t.
    """

    def __init__(
        self,
        security_param: int = 128,
        delay_slots: int = 32,
        iterations: int = 1,
    ) -> None:
        self.security_param = security_param
        self.delay_slots = delay_slots
        self.iterations = iterations

    def setup(self, security_param: int, delay_slots: int) -> None:
        self.security_param = security_param
        self.delay_slots = delay_slots

    def eval(self, input_value: int, current_slot: int) -> tuple[int, int]:
        """
        Evaluate the VDF.

        Returns
        -------
        (output, ready_slot)
            output     — VDF output (256-bit int)
            ready_slot — the simulation slot at which the output is
                         considered available.
        """
        output = _iterated_hash(input_value, self.iterations)
        ready_slot = current_slot + self.delay_slots
        return output, ready_slot

    def verify(self, input_value: int, output: int, *, iterations: int = None) -> bool:
        """
        Verify that `output` is the correct VDF evaluation of `input_value`.
        """
        iters = iterations if iterations is not None else self.iterations
        expected = _iterated_hash(input_value, iters)
        return expected == output
