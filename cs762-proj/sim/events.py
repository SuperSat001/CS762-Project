"""
PART 5: Discrete-event system using a priority queue.

Event types (ordered by execution priority within a slot):
    SLOT_START        priority 0
    BLOCK_PROPOSE     priority 1
    ATTEST            priority 2
    FORK_CHOICE_UPDATE priority 3
    EPOCH_END         priority 4  (fires at end of last slot of epoch)
    VDF_COMPLETE      priority 5  (fires at arbitrary future slot)

The queue is keyed on (slot, priority, sequence_number).
sequence_number breaks ties deterministically for same-slot same-priority events.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, List, Optional


class EventType(IntEnum):
    SLOT_START         = 0
    BLOCK_PROPOSE      = 1
    ATTEST             = 2
    FORK_CHOICE_UPDATE = 3
    EPOCH_END          = 4
    VDF_COMPLETE       = 5


@dataclass(order=True)
class Event:
    slot: int
    priority: int           # EventType value
    seq: int                # insertion counter for stable ordering
    type: EventType = field(compare=False)
    payload: Any       = field(compare=False, default=None)

    def __repr__(self) -> str:
        return f"Event({self.type.name}, slot={self.slot}, seq={self.seq})"


class EventQueue:
    """Min-heap priority queue for simulation events."""

    def __init__(self) -> None:
        self._heap: List[Event] = []
        self._counter = 0

    def push(
        self,
        event_type: EventType,
        slot: int,
        payload: Any = None,
    ) -> Event:
        seq = self._counter
        self._counter += 1
        ev = Event(
            slot=slot,
            priority=int(event_type),
            seq=seq,
            type=event_type,
            payload=payload,
        )
        heapq.heappush(self._heap, ev)
        return ev

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Event]:
        return self._heap[0] if self._heap else None

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)


# ---------------------------------------------------------------------------
# Payloads (typed dicts kept simple)
# ---------------------------------------------------------------------------

def slot_start_payload(slot: int) -> dict:
    return {"slot": slot}

def propose_payload(slot: int, proposer_id: int) -> dict:
    return {"slot": slot, "proposer_id": proposer_id}

def attest_payload(slot: int) -> dict:
    return {"slot": slot}

def fork_choice_payload(slot: int) -> dict:
    return {"slot": slot}

def epoch_end_payload(epoch: int) -> dict:
    return {"epoch": epoch}

def vdf_complete_payload(epoch: int, vdf_output: int) -> dict:
    return {"epoch": epoch, "vdf_output": vdf_output}
