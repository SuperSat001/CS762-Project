"""
PART 6: Validator strategies.

Each strategy implements:
    propose_action(state_view, slot) → ProposalAction
    publish_private(state_view, slot) → bool

ProposalAction:
    PROPOSE  — build and broadcast a block on the current head
    SKIP     — do not propose (missed slot)
    PRIVATE  — build a block but keep it private

StateView is a lightweight snapshot passed to strategies so they
CANNOT see future randomness or private chain state of others.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .simulator import StateView


class ProposalAction(Enum):
    PROPOSE = auto()
    SKIP    = auto()
    PRIVATE = auto()


class Strategy:
    """Abstract base class."""

    def propose_action(self, state_view: "StateView", slot: int) -> ProposalAction:
        raise NotImplementedError

    def publish_private(self, state_view: "StateView", slot: int) -> bool:
        """
        Called each slot for adversarial validators with a private chain.
        Return True to release the private chain to the network.
        """
        return False

    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# 1. Honest strategy
# ---------------------------------------------------------------------------

class HonestStrategy(Strategy):
    """Always proposes; always builds on the current head."""

    def propose_action(self, state_view: "StateView", slot: int) -> ProposalAction:
        return ProposalAction.PROPOSE

    def publish_private(self, state_view: "StateView", slot: int) -> bool:
        return False  # honest validator never withholds


# ---------------------------------------------------------------------------
# 2. Selfish-mixing strategy  (RANDAO manipulation via skipping)
# ---------------------------------------------------------------------------

class SelfishMixingStrategy(Strategy):
    """
    Skips proposal in 'tail' slots of an epoch (slots ≥ epoch_end - tail)
    when the adversary is the proposer, to prevent their reveal from
    entering R_e and bias the next seed.

    Parameters
    ----------
    tail : int
        Number of tail slots to consider (default 1 = last slot of epoch).
    skip_probability : float
        Probability of skipping when in a tail slot (default 1.0 = always).
    """

    def __init__(self, tail: int = 1, skip_probability: float = 1.0) -> None:
        self.tail = tail
        self.skip_probability = skip_probability
        import random
        self._rng = random.Random()

    def _is_tail_slot(self, slot: int) -> bool:
        from .randao import SLOTS_PER_EPOCH
        slot_in_epoch = slot % SLOTS_PER_EPOCH
        return slot_in_epoch >= (SLOTS_PER_EPOCH - self.tail)

    def propose_action(self, state_view: "StateView", slot: int) -> ProposalAction:
        if self._is_tail_slot(slot):
            if self._rng.random() < self.skip_probability:
                return ProposalAction.SKIP
        return ProposalAction.PROPOSE

    def publish_private(self, state_view: "StateView", slot: int) -> bool:
        return False


# ---------------------------------------------------------------------------
# 3. Forking (private-chain) strategy
# ---------------------------------------------------------------------------

class ForkingStrategy(Strategy):
    """
    Builds blocks privately and releases them strategically.

    Release condition (simple policy):
    * Release when private chain length advantage ≥ release_threshold,
      OR at the start of the epoch boundary.
    * Randomised: release with some probability each slot.

    Parameters
    ----------
    release_threshold : int
        Release when private lead ≥ this many blocks.
    release_probability : float
        Per-slot probability of releasing even if below threshold.
    """

    def __init__(
        self,
        release_threshold: int = 2,
        release_probability: float = 0.0,
    ) -> None:
        self.release_threshold = release_threshold
        self.release_probability = release_probability
        self._private_lead: int = 0      # tracked externally by simulator
        import random
        self._rng = random.Random()

    def propose_action(self, state_view: "StateView", slot: int) -> ProposalAction:
        # Always build privately
        return ProposalAction.PRIVATE

    def publish_private(self, state_view: "StateView", slot: int) -> bool:
        if self._private_lead >= self.release_threshold:
            self._private_lead = 0
            return True
        if self.release_probability > 0 and self._rng.random() < self.release_probability:
            self._private_lead = 0
            return True
        return False

    def increment_lead(self) -> None:
        self._private_lead += 1

    def reset_lead(self) -> None:
        self._private_lead = 0
