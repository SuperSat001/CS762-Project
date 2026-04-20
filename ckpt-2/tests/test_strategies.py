"""
TEST GROUP 5: Strategies
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.strategies import (
    HonestStrategy,
    SelfishMixingStrategy,
    ForkingStrategy,
    ProposalAction,
)
from sim.randao import SLOTS_PER_EPOCH
from sim.simulator import StateView


def _view(slot=0, head_id=0, private_lead=0):
    return StateView(
        current_slot=slot,
        head_id=head_id,
        epoch=slot // SLOTS_PER_EPOCH,
        private_lead=private_lead,
    )


# ---------------------------------------------------------------------------
# test_honest_always_proposes
# ---------------------------------------------------------------------------

class TestHonestStrategy:
    def test_always_proposes(self):
        s = HonestStrategy()
        for slot in range(64):
            assert s.propose_action(_view(slot), slot) == ProposalAction.PROPOSE

    def test_never_publishes_private(self):
        s = HonestStrategy()
        for slot in range(32):
            assert s.publish_private(_view(slot, private_lead=5), slot) is False

    def test_name(self):
        assert HonestStrategy().name() == "HonestStrategy"


# ---------------------------------------------------------------------------
# test_selfish_skips_tail
# ---------------------------------------------------------------------------

class TestSelfishMixingStrategy:
    def _tail_slot(self, tail=1) -> int:
        """Return the last slot of epoch 0."""
        return SLOTS_PER_EPOCH - tail

    def test_skips_in_tail_slot(self):
        s = SelfishMixingStrategy(tail=1, skip_probability=1.0)
        tail_slot = SLOTS_PER_EPOCH - 1
        action = s.propose_action(_view(tail_slot), tail_slot)
        assert action == ProposalAction.SKIP

    def test_proposes_in_non_tail_slot(self):
        s = SelfishMixingStrategy(tail=1, skip_probability=1.0)
        for slot in range(SLOTS_PER_EPOCH - 1):
            action = s.propose_action(_view(slot), slot)
            assert action == ProposalAction.PROPOSE

    def test_skips_multiple_tail_slots(self):
        tail = 3
        s = SelfishMixingStrategy(tail=tail, skip_probability=1.0)
        for i in range(tail):
            slot = SLOTS_PER_EPOCH - 1 - i
            action = s.propose_action(_view(slot), slot)
            assert action == ProposalAction.SKIP, f"Expected SKIP at slot {slot}"

    def test_zero_probability_never_skips(self):
        s = SelfishMixingStrategy(tail=1, skip_probability=0.0)
        for slot in range(SLOTS_PER_EPOCH):
            action = s.propose_action(_view(slot), slot)
            assert action == ProposalAction.PROPOSE

    def test_never_publishes_private(self):
        s = SelfishMixingStrategy()
        assert s.publish_private(_view(), 0) is False


# ---------------------------------------------------------------------------
# test_forking_private_chain
# ---------------------------------------------------------------------------

class TestForkingStrategy:
    def test_always_builds_privately(self):
        s = ForkingStrategy()
        for slot in range(32):
            action = s.propose_action(_view(slot), slot)
            assert action == ProposalAction.PRIVATE

    def test_no_publish_below_threshold(self):
        s = ForkingStrategy(release_threshold=3, release_probability=0.0)
        s._private_lead = 2
        view = _view(private_lead=2)
        assert s.publish_private(view, slot=5) is False

    def test_publishes_at_threshold(self):
        s = ForkingStrategy(release_threshold=3, release_probability=0.0)
        s._private_lead = 3
        view = _view(private_lead=3)
        result = s.publish_private(view, slot=5)
        assert result is True

    def test_lead_resets_after_publish(self):
        s = ForkingStrategy(release_threshold=2, release_probability=0.0)
        s.increment_lead()
        s.increment_lead()
        s.publish_private(_view(private_lead=2), slot=1)
        assert s._private_lead == 0

    def test_increment_lead(self):
        s = ForkingStrategy()
        for i in range(5):
            s.increment_lead()
        assert s._private_lead == 5

    def test_reset_lead(self):
        s = ForkingStrategy()
        s.increment_lead()
        s.increment_lead()
        s.reset_lead()
        assert s._private_lead == 0

    def test_private_blocks_not_visible_before_publish(self):
        """
        The ForkingStrategy returns PRIVATE, meaning blocks added by the
        simulator are held in _private_blocks until publish_private returns True.
        This test verifies the strategy's state (lead tracking) matches.
        """
        s = ForkingStrategy(release_threshold=10, release_probability=0.0)
        # Simulate 5 private proposals
        for _ in range(5):
            action = s.propose_action(_view(), slot=1)
            assert action == ProposalAction.PRIVATE
            s.increment_lead()

        # Should NOT release yet (lead=5 < threshold=10)
        assert s.publish_private(_view(private_lead=5), slot=5) is False
        assert s._private_lead == 5
