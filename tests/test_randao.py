"""
Tests for sim/randao.py
Gate 1 coverage: RANDAO update/use timing, XOR correctness, reproducibility.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sim.randao import (
    RANDAOState,
    adversary_reveal,
    honest_reveal,
    randao_contribution,
    _xor32,
    _sha256,
)


# ──────────────────────────────────────────────
# Hash / XOR primitives
# ──────────────────────────────────────────────

def test_xor32_identity():
    z = b"\x00" * 32
    a = _sha256(b"hello")
    assert _xor32(a, z) == a
    assert _xor32(z, a) == a


def test_xor32_self_cancels():
    a = _sha256(b"world")
    assert _xor32(a, a) == b"\x00" * 32


def test_xor32_commutativity():
    a = _sha256(b"abc")
    b = _sha256(b"xyz")
    assert _xor32(a, b) == _xor32(b, a)


def test_sha256_length():
    h = _sha256(b"test")
    assert len(h) == 32


# ──────────────────────────────────────────────
# Reveal generation
# ──────────────────────────────────────────────

def test_adversary_reveal_deterministic():
    r1 = adversary_reveal(1, 7, 3)
    r2 = adversary_reveal(1, 7, 3)
    assert r1 == r2


def test_adversary_reveal_different_epochs():
    r1 = adversary_reveal(1, 7, 0)
    r2 = adversary_reveal(1, 7, 1)
    assert r1 != r2


def test_honest_reveal_deterministic():
    r1 = honest_reveal(epoch_seed=99, validator_id=2, epoch=0)
    r2 = honest_reveal(epoch_seed=99, validator_id=2, epoch=0)
    assert r1 == r2


def test_honest_vs_adversary_different():
    r_h = honest_reveal(epoch_seed=1, validator_id=5, epoch=0)
    r_a = adversary_reveal(1, 5, 0)
    assert r_h != r_a


def test_randao_contribution_length():
    r = adversary_reveal(1, 2, 3)
    c = randao_contribution(r)
    assert len(c) == 32


# ──────────────────────────────────────────────
# RANDAOState lifecycle
# ──────────────────────────────────────────────

def test_genesis_mix_nonzero():
    state = RANDAOState(genesis_seed=42)
    gm = state.get_mix(-1)
    assert gm is not None
    assert len(gm) == 32
    assert gm != b"\x00" * 32


def test_different_seeds_different_genesis():
    s1 = RANDAOState(genesis_seed=1)
    s2 = RANDAOState(genesis_seed=2)
    assert s1.get_mix(-1) != s2.get_mix(-1)


def test_begin_epoch_resets_to_prev_mix():
    state = RANDAOState(genesis_seed=0)
    state.begin_epoch(0)
    rev = adversary_reveal(0, 0, 0)
    state.apply_reveal(rev)
    mix_after_reveal = state.current_mix()
    mix0 = state.finalize_epoch(0)
    assert mix0 == mix_after_reveal

    # Start epoch 1: accumulator resets to mix[0]
    state.begin_epoch(1)
    assert state.current_mix() == mix0


def test_apply_reveal_changes_mix():
    state = RANDAOState(genesis_seed=42)
    state.begin_epoch(0)
    before = state.current_mix()
    state.apply_reveal(adversary_reveal(0, 1, 0))
    after = state.current_mix()
    assert before != after


def test_skip_slot_no_change():
    state = RANDAOState(genesis_seed=42)
    state.begin_epoch(0)
    before = state.current_mix()
    state.skip_slot()
    after = state.current_mix()
    assert before == after


def test_finalize_epoch_idempotent_retrieval():
    state = RANDAOState(genesis_seed=7)
    state.begin_epoch(0)
    state.apply_reveal(adversary_reveal(7, 3, 0))
    mix = state.finalize_epoch(0)
    assert state.get_mix(0) == mix


def test_epoch_mix_ordering():
    """mix[1] should differ from mix[0] when different reveals are applied."""
    state = RANDAOState(genesis_seed=100)
    state.begin_epoch(0)
    state.apply_reveal(adversary_reveal(100, 0, 0))
    state.finalize_epoch(0)

    state.begin_epoch(1)
    state.apply_reveal(adversary_reveal(100, 1, 1))
    state.finalize_epoch(1)

    assert state.get_mix(0) != state.get_mix(1)


def test_not_finalized_returns_none():
    state = RANDAOState(genesis_seed=0)
    assert state.get_mix(5) is None


# ──────────────────────────────────────────────
# simulate_final_mix helper
# ──────────────────────────────────────────────

def test_simulate_final_mix_empty():
    state = RANDAOState(genesis_seed=5)
    base = state.get_mix(-1)
    result = state.simulate_final_mix(base, [], [])
    assert result == base


def test_simulate_final_mix_adv_vs_skip():
    """Including a reveal should differ from not including it."""
    state = RANDAOState(genesis_seed=5)
    base = state.get_mix(-1)
    rev = adversary_reveal(5, 2, 0)
    mix_with = state.simulate_final_mix(base, [rev], [])
    mix_without = state.simulate_final_mix(base, [], [])
    assert mix_with != mix_without


def test_simulate_final_mix_xor_property():
    """XOR twice with same reveal cancels out."""
    state = RANDAOState(genesis_seed=5)
    base = state.get_mix(-1)
    rev = adversary_reveal(5, 2, 0)
    mix_once = state.simulate_final_mix(base, [rev], [])
    # The contribution of rev XOR'd with itself should recover base (via honest list)
    from sim.randao import randao_contribution
    contrib = randao_contribution(rev)
    mix_twice = state.simulate_final_mix(mix_once, [rev], [])
    assert mix_twice == base  # XOR property: (base XOR c) XOR c = base


def test_reproducibility_across_instances():
    """Two state instances with same seed produce identical mixes."""
    def run(seed):
        state = RANDAOState(genesis_seed=seed)
        state.begin_epoch(0)
        state.apply_reveal(adversary_reveal(seed, 1, 0))
        state.apply_reveal(honest_reveal(seed ^ 0xFF, 2, 0))
        return state.finalize_epoch(0)

    assert run(99) == run(99)
    assert run(99) != run(100)
