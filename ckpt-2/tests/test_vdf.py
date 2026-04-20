"""
TEST GROUP 6: VDF
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.vdf import VDF, _iterated_hash
from sim.randao import RandaoState, SLOTS_PER_EPOCH


# ---------------------------------------------------------------------------
# test_vdf_delay
# ---------------------------------------------------------------------------

class TestVdfDelay:
    def test_output_not_available_before_ready_slot(self):
        vdf = VDF(delay_slots=32)
        current_slot = 10
        _, ready_slot = vdf.eval(input_value=0xABCD, current_slot=current_slot)
        assert ready_slot > current_slot

    def test_ready_slot_equals_current_plus_delay(self):
        vdf = VDF(delay_slots=32)
        current_slot = 64
        _, ready_slot = vdf.eval(input_value=12345, current_slot=current_slot)
        assert ready_slot == current_slot + 32

    def test_zero_delay_available_immediately(self):
        vdf = VDF(delay_slots=0)
        current_slot = 5
        _, ready_slot = vdf.eval(input_value=0, current_slot=current_slot)
        assert ready_slot == current_slot

    def test_output_is_integer(self):
        vdf = VDF()
        output, _ = vdf.eval(input_value=42, current_slot=0)
        assert isinstance(output, int)
        assert output >= 0


# ---------------------------------------------------------------------------
# test_vdf_determinism
# ---------------------------------------------------------------------------

class TestVdfDeterminism:
    def test_same_input_same_output(self):
        vdf = VDF(iterations=3)
        out1, _ = vdf.eval(0xDEADBEEF, current_slot=0)
        out2, _ = vdf.eval(0xDEADBEEF, current_slot=0)
        assert out1 == out2

    def test_different_input_different_output(self):
        vdf = VDF(iterations=3)
        out1, _ = vdf.eval(0x0001, current_slot=0)
        out2, _ = vdf.eval(0x0002, current_slot=0)
        assert out1 != out2

    def test_current_slot_does_not_affect_output(self):
        """Output must be the same regardless of when eval is called."""
        vdf = VDF(iterations=2)
        out1, _ = vdf.eval(999, current_slot=10)
        out2, _ = vdf.eval(999, current_slot=200)
        assert out1 == out2

    def test_iterated_hash_deterministic(self):
        h1 = _iterated_hash(12345, iterations=5)
        h2 = _iterated_hash(12345, iterations=5)
        assert h1 == h2

    def test_iterated_hash_zero_iterations(self):
        """0 iterations should return the input unchanged (as integer)."""
        # 0 iterations: current = input bytes, no hashing applied
        # By our definition, the loop doesn't execute, so we return int(input bytes)
        val = 0xABCDEF
        result = _iterated_hash(val, iterations=0)
        # Result is just the 32-byte big-endian encoding interpreted as int
        expected = int.from_bytes(val.to_bytes(32, "big"), "big")
        assert result == expected == val


# ---------------------------------------------------------------------------
# test_vdf_blocks_schedule
# ---------------------------------------------------------------------------

class TestVdfBlocksSchedule:
    def test_schedule_unavailable_until_vdf_complete(self):
        """
        After epoch_end with use_vdf=True, the proposer schedule for
        epoch+2 must remain None until complete_vdf is called.
        """
        validators = list(range(10))
        state = RandaoState(validators)

        state.finalize_epoch(0, [0x1234], use_vdf=True)

        # Schedule for epoch 2 not yet available
        assert state.get_schedule(2) is None
        assert not state.is_schedule_ready(2)

    def test_schedule_available_after_vdf_complete(self):
        validators = list(range(10))
        state = RandaoState(validators)

        state.finalize_epoch(0, [0x1234], use_vdf=True)
        state.complete_vdf(0, vdf_output=0xABCD)

        assert state.is_schedule_ready(2)
        sched = state.get_schedule(2)
        assert sched is not None
        assert len(sched) == SLOTS_PER_EPOCH

    def test_vdf_output_used_for_schedule(self):
        """VDF output (not raw R_e) seeds the proposer schedule."""
        validators = list(range(8))
        state1 = RandaoState(validators)
        state2 = RandaoState(validators)

        state1.finalize_epoch(0, [0x9999], use_vdf=True)
        state2.finalize_epoch(0, [0x9999], use_vdf=True)

        # Give different VDF outputs
        state1.complete_vdf(0, vdf_output=0xAAAA)
        state2.complete_vdf(0, vdf_output=0xBBBB)

        sched1 = state1.get_schedule(2)
        sched2 = state2.get_schedule(2)
        assert sched1 != sched2

    def test_vdf_verify(self):
        vdf = VDF(iterations=2)
        output, _ = vdf.eval(0x5678, current_slot=0)
        assert vdf.verify(0x5678, output)
        assert not vdf.verify(0x5678, output + 1)

    def test_setup_changes_delay(self):
        vdf = VDF(delay_slots=10)
        vdf.setup(security_param=256, delay_slots=64)
        _, ready = vdf.eval(0, current_slot=0)
        assert ready == 64
