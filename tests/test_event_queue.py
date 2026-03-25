"""
Tests for sim/event_queue.py
Gate 1 coverage: queue ordering, causality, determinism.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sim.event_queue import EventQueue, EVENT_SLOT_START, EVENT_EPOCH_END, EVENT_VDF_COMPLETE


# ──────────────────────────────────────────────
# Basic ordering
# ──────────────────────────────────────────────

def test_empty_queue():
    eq = EventQueue()
    assert eq.empty()
    assert len(eq) == 0
    assert eq.peek_time() is None


def test_single_event():
    eq = EventQueue()
    eq.schedule(5.0, EVENT_SLOT_START, {"slot": 1})
    assert not eq.empty()
    assert len(eq) == 1
    t, eid, etype, payload = eq.pop()
    assert t == 5.0
    assert etype == EVENT_SLOT_START
    assert payload["slot"] == 1
    assert eq.empty()


def test_ordering_multiple_events():
    """Events must come out in non-decreasing time order."""
    eq = EventQueue()
    times = [10.0, 3.0, 7.5, 1.0, 7.5, 0.0]
    for i, t in enumerate(times):
        eq.schedule(t, EVENT_SLOT_START, {"i": i})

    prev_t = -1.0
    while not eq.empty():
        t, eid, etype, payload = eq.pop()
        assert t >= prev_t, f"Ordering violation: {t} < {prev_t}"
        prev_t = t


def test_stable_ordering_same_time():
    """Same-time events must come out in insertion order (FIFO by event_id)."""
    eq = EventQueue()
    for i in range(5):
        eq.schedule(1.0, EVENT_SLOT_START, {"seq": i})

    seqs = []
    while not eq.empty():
        t, eid, etype, payload = eq.pop()
        seqs.append(payload["seq"])
    assert seqs == [0, 1, 2, 3, 4], f"Expected FIFO order, got {seqs}"


def test_event_ids_monotone():
    """event_id must strictly increase with each scheduled event."""
    eq = EventQueue()
    ids = []
    for t in [3.0, 1.0, 2.0]:
        eid = eq.schedule(t, EVENT_SLOT_START, {})
        ids.append(eid)
    assert ids == sorted(ids) and ids == list(range(3))


def test_causality_assert():
    """assert_causal_order() should not raise on a valid heap."""
    eq = EventQueue()
    eq.schedule(0.0, EVENT_SLOT_START, {})
    eq.schedule(5.0, EVENT_EPOCH_END, {})
    eq.schedule(10.0, EVENT_VDF_COMPLETE, {})
    eq.assert_causal_order()   # should not raise


def test_peek_time():
    eq = EventQueue()
    eq.schedule(7.0, EVENT_EPOCH_END, {})
    eq.schedule(2.0, EVENT_SLOT_START, {})
    assert eq.peek_time() == 2.0


def test_large_volume():
    """Insert and drain 10 000 events; verify strict ordering."""
    import random
    rng = random.Random(42)
    eq = EventQueue()
    N = 10_000
    for _ in range(N):
        t = rng.uniform(0, 1000)
        eq.schedule(t, EVENT_SLOT_START, {})
    assert len(eq) == N

    prev_t = -1.0
    count = 0
    while not eq.empty():
        t, eid, etype, payload = eq.pop()
        assert t >= prev_t
        prev_t = t
        count += 1
    assert count == N


def test_deterministic_across_runs():
    """Same schedule calls → same pop order (no randomness in EventQueue)."""
    def build_and_drain(seed_offset: int):
        eq = EventQueue()
        times = [float(i % 7) for i in range(20)]
        for t in times:
            eq.schedule(t, EVENT_SLOT_START, {"v": int(t)})
        return [eq.pop()[0] for _ in range(20)]

    out1 = build_and_drain(0)
    out2 = build_and_drain(0)
    assert out1 == out2
