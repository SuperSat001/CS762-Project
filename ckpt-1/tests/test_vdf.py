"""
Tests for sim/vdf.py
Gate 3 coverage: VDF temporal causality, determinism, pipeline availability.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sim.vdf import FastSimVDF, VDFPipeline, VDFResult, make_vdf_pipeline


# ──────────────────────────────────────────────
# FastSimVDF
# ──────────────────────────────────────────────

def test_fast_sim_vdf_deterministic():
    vdf = FastSimVDF()
    x = b"hello world" + b"\x00" * 21
    r1 = vdf.compute(x, iterations=16)
    r2 = vdf.compute(x, iterations=16)
    assert r1.output == r2.output
    assert r1.proof  == r2.proof


def test_fast_sim_vdf_output_length():
    vdf = FastSimVDF()
    r = vdf.compute(b"\xAB" * 32, iterations=8)
    assert len(r.output) == 32


def test_fast_sim_vdf_verify_valid():
    vdf = FastSimVDF()
    r = vdf.compute(b"\x01" * 32, iterations=4)
    assert vdf.verify(r)


def test_fast_sim_vdf_verify_tampered_output():
    vdf = FastSimVDF()
    r = vdf.compute(b"\x02" * 32, iterations=4)
    # Tamper output
    bad = VDFResult(
        input_data=r.input_data,
        output=bytes([b ^ 0xFF for b in r.output]),
        proof=r.proof,
        iterations=r.iterations,
    )
    assert not vdf.verify(bad)


def test_fast_sim_vdf_different_inputs_different_outputs():
    vdf = FastSimVDF()
    r1 = vdf.compute(b"\xAA" * 32, iterations=8)
    r2 = vdf.compute(b"\xBB" * 32, iterations=8)
    assert r1.output != r2.output


def test_fast_sim_vdf_more_iterations_different():
    vdf = FastSimVDF()
    x = b"\xCC" * 32
    r1 = vdf.compute(x, iterations=4)
    r2 = vdf.compute(x, iterations=8)
    assert r1.output != r2.output


# ──────────────────────────────────────────────
# VDFPipeline – temporal causality
# ──────────────────────────────────────────────

def test_pipeline_not_available_before_delay():
    pipeline = make_vdf_pipeline(use_real_vdf=False, delay_slots=64)
    mix = b"\xDE" * 32
    pipeline.submit(epoch=0, randao_mix=mix, epoch_end_time=32.0)

    # Available at 32.0 + 64 = 96.0
    assert pipeline.get_result(0, current_time=50.0) is None
    assert pipeline.get_result(0, current_time=95.9) is None


def test_pipeline_available_exactly_at_delay():
    pipeline = make_vdf_pipeline(use_real_vdf=False, delay_slots=64)
    mix = b"\xDE" * 32
    pipeline.submit(epoch=0, randao_mix=mix, epoch_end_time=32.0)
    assert pipeline.get_result(0, current_time=96.0) is not None


def test_pipeline_available_after_delay():
    pipeline = make_vdf_pipeline(use_real_vdf=False, delay_slots=64)
    mix = b"\xDE" * 32
    pipeline.submit(epoch=0, randao_mix=mix, epoch_end_time=32.0)
    r = pipeline.get_result(0, current_time=200.0)
    assert r is not None
    assert r.input_data == mix


def test_pipeline_unknown_epoch_returns_none():
    pipeline = make_vdf_pipeline(use_real_vdf=False, delay_slots=64)
    assert pipeline.get_result(42, current_time=9999.0) is None


def test_pipeline_is_available_bool():
    pipeline = make_vdf_pipeline(use_real_vdf=False, delay_slots=32)
    mix = b"\xAB" * 32
    pipeline.submit(epoch=5, randao_mix=mix, epoch_end_time=160.0)
    assert not pipeline.is_available(5, 160.0)  # just submitted, delay=32
    assert not pipeline.is_available(5, 191.9)
    assert pipeline.is_available(5, 192.0)


def test_pipeline_multiple_epochs_independent():
    pipeline = make_vdf_pipeline(use_real_vdf=False, delay_slots=32)
    for epoch in range(5):
        mix = bytes([epoch] * 32)
        pipeline.submit(epoch=epoch, randao_mix=mix, epoch_end_time=float(epoch * 32))

    # Epoch 0 ends at t=0, available at t=32
    # Epoch 4 ends at t=128, available at t=160
    t_now = 100.0
    assert pipeline.is_available(0, t_now)
    assert pipeline.is_available(1, t_now)
    assert pipeline.is_available(2, t_now)
    assert not pipeline.is_available(3, t_now)   # t=96+32=128 > 100
    assert not pipeline.is_available(4, t_now)


def test_pipeline_output_deterministic():
    """Same RANDAO mix → same VDF output regardless of when submitted."""
    mix = b"\xFF" * 32
    p1 = make_vdf_pipeline(delay_slots=10)
    p2 = make_vdf_pipeline(delay_slots=10)
    p1.submit(0, mix, 0.0)
    p2.submit(0, mix, 50.0)
    r1 = p1.get_result(0, 10.0)
    r2 = p2.get_result(0, 60.0)
    assert r1 is not None and r2 is not None
    assert r1.output == r2.output


def test_pipeline_unchecked_before_available():
    """get_result_unchecked ignores temporal constraint (adversary lookahead)."""
    pipeline = make_vdf_pipeline(delay_slots=1000)
    mix = b"\x11" * 32
    pipeline.submit(0, mix, epoch_end_time=0.0)
    # Not available yet at t=0
    assert pipeline.get_result(0, 0.0) is None
    # But unchecked works
    assert pipeline.get_result_unchecked(0) is not None


def test_compute_budget_flag():
    pipeline = make_vdf_pipeline(delay_slots=32, vdf_compute_budget=1)
    assert pipeline.adversary_can_evaluate(0)
    assert not pipeline.adversary_can_evaluate(1)
    assert not pipeline.adversary_can_evaluate(100)


def test_make_vdf_pipeline_factory():
    p = make_vdf_pipeline(use_real_vdf=False, delay_slots=48, vdf_compute_budget=3)
    assert p.delay_slots == 48
    assert p.vdf_compute_budget == 3
