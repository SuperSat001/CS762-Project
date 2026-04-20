"""
TEST GROUP 4: Event Engine
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.events import EventQueue, EventType


# ---------------------------------------------------------------------------
# test_event_ordering
# ---------------------------------------------------------------------------

class TestEventOrdering:
    def test_events_ordered_by_slot(self):
        q = EventQueue()
        q.push(EventType.SLOT_START, slot=5)
        q.push(EventType.SLOT_START, slot=2)
        q.push(EventType.SLOT_START, slot=8)
        slots = [q.pop().slot for _ in range(3)]
        assert slots == [2, 5, 8]

    def test_same_slot_ordered_by_priority(self):
        q = EventQueue()
        q.push(EventType.FORK_CHOICE_UPDATE, slot=0)
        q.push(EventType.ATTEST,             slot=0)
        q.push(EventType.BLOCK_PROPOSE,      slot=0)
        q.push(EventType.SLOT_START,         slot=0)
        types = [q.pop().type for _ in range(4)]
        assert types == [
            EventType.SLOT_START,
            EventType.BLOCK_PROPOSE,
            EventType.ATTEST,
            EventType.FORK_CHOICE_UPDATE,
        ]

    def test_epoch_end_after_fork_choice(self):
        q = EventQueue()
        q.push(EventType.EPOCH_END,          slot=31)
        q.push(EventType.FORK_CHOICE_UPDATE, slot=31)
        types = [q.pop().type for _ in range(2)]
        assert types == [EventType.FORK_CHOICE_UPDATE, EventType.EPOCH_END]

    def test_vdf_complete_fires_at_correct_slot(self):
        q = EventQueue()
        q.push(EventType.SLOT_START,   slot=10)
        q.push(EventType.VDF_COMPLETE, slot=42)
        q.push(EventType.SLOT_START,   slot=40)

        first = q.pop()
        assert first.slot == 10
        second = q.pop()
        assert second.slot == 40
        third = q.pop()
        assert third.type == EventType.VDF_COMPLETE
        assert third.slot == 42

    def test_stable_ordering_same_slot_same_type(self):
        """Same slot + same type → insertion order preserved (seq)."""
        q = EventQueue()
        e1 = q.push(EventType.ATTEST, slot=5, payload="first")
        e2 = q.push(EventType.ATTEST, slot=5, payload="second")
        out1 = q.pop()
        out2 = q.pop()
        assert out1.payload == "first"
        assert out2.payload == "second"

    def test_is_empty(self):
        q = EventQueue()
        assert q.is_empty()
        q.push(EventType.SLOT_START, slot=0)
        assert not q.is_empty()
        q.pop()
        assert q.is_empty()

    def test_len(self):
        q = EventQueue()
        for i in range(7):
            q.push(EventType.SLOT_START, slot=i)
        assert len(q) == 7


# ---------------------------------------------------------------------------
# test_slot_sequence
# ---------------------------------------------------------------------------

class TestSlotSequence:
    def test_standard_slot_order(self):
        """Push the 4 standard per-slot events and verify pop order."""
        q = EventQueue()
        slot = 3
        q.push(EventType.SLOT_START,         slot)
        q.push(EventType.BLOCK_PROPOSE,      slot)
        q.push(EventType.ATTEST,             slot)
        q.push(EventType.FORK_CHOICE_UPDATE, slot)

        expected = [
            EventType.SLOT_START,
            EventType.BLOCK_PROPOSE,
            EventType.ATTEST,
            EventType.FORK_CHOICE_UPDATE,
        ]
        got = [q.pop().type for _ in range(4)]
        assert got == expected

    def test_multi_slot_interleaving(self):
        """Events for slot N+1 come after slot N events."""
        q = EventQueue()
        for slot in range(3):
            q.push(EventType.SLOT_START,         slot)
            q.push(EventType.BLOCK_PROPOSE,      slot)
            q.push(EventType.ATTEST,             slot)
            q.push(EventType.FORK_CHOICE_UPDATE, slot)

        prev_slot = -1
        prev_prio = -1
        while not q.is_empty():
            ev = q.pop()
            if ev.slot == prev_slot:
                assert ev.priority >= prev_prio
            else:
                assert ev.slot > prev_slot
            prev_slot = ev.slot
            prev_prio = ev.priority
