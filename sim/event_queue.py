"""
Discrete-Event Simulation (DES) global event queue.

Each event is a tuple: (sim_time, event_id, event_type, payload).
- sim_time:   float, simulation clock when event fires
- event_id:   int,   monotone counter for stable ordering of same-time events
- event_type: str,   one of the EVENT_* constants below
- payload:    dict,  event-specific data

The queue is a min-heap on (sim_time, event_id).  Popping always returns the
earliest event, advancing the simulation clock to that point in O(log n).
"""
from __future__ import annotations

import heapq
from typing import Any

# ──────────────────────────────────────────────
# Event-type constants
# ──────────────────────────────────────────────
EVENT_SLOT_START       = "SLOT_START"
EVENT_BLOCK_PROPOSE    = "BLOCK_PROPOSE"
EVENT_EPOCH_START      = "EPOCH_START"
EVENT_EPOCH_END        = "EPOCH_END"
EVENT_VDF_COMPLETE     = "VDF_COMPLETE"
EVENT_ADVERSARY_DECIDE = "ADVERSARY_DECIDE"
EVENT_FORK_ATTEMPT     = "FORK_ATTEMPT"


class EventQueue:
    """Priority-queue–based DES engine.

    Usage::

        eq = EventQueue()
        eq.schedule(0.0, EVENT_SLOT_START, {"slot": 0})
        while not eq.empty():
            t, eid, etype, payload = eq.pop()
            ...
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str, Any]] = []
        self._counter: int = 0

    # ── public API ────────────────────────────

    def schedule(self, time: float, event_type: str, payload: Any = None) -> int:
        """Insert an event and return its unique event_id."""
        eid = self._counter
        self._counter += 1
        heapq.heappush(self._heap, (time, eid, event_type, payload))
        return eid

    def pop(self) -> tuple[float, int, str, Any]:
        """Remove and return the next (earliest) event.

        Returns:
            (sim_time, event_id, event_type, payload)

        Raises:
            IndexError if the queue is empty.
        """
        return heapq.heappop(self._heap)

    def peek_time(self) -> float | None:
        """Return the time of the next event without removing it."""
        return self._heap[0][0] if self._heap else None

    def empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)

    # ── invariant checking (used by tests) ────

    def assert_causal_order(self) -> None:
        """Assert that no event in the heap is scheduled before the current min-time.
        This is always true for a valid min-heap; this is a sanity check for tests.
        """
        if not self._heap:
            return
        min_t = self._heap[0][0]
        for t, eid, etype, _ in self._heap:
            assert t >= min_t, (
                f"Causality violation: event {eid} ({etype}) at t={t} < min_t={min_t}"
            )
