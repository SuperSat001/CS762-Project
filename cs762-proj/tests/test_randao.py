"""
TEST GROUP 3: RANDAO
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.randao import (
    RandaoState,
    compute_proposer_schedule,
    SLOTS_PER_EPOCH,
    _xor_reveals,
    _epoch_seed,
)
from sim.block import BlockTree, compute_randao_reveal
from sim.fork_choice import ForkChoice


VALIDATORS = list(range(10))  # 10 validators, ids 0-9


# ---------------------------------------------------------------------------
# test_randao_update_basic
# ---------------------------------------------------------------------------

class TestRandaoUpdateBasic:
    def test_initial_value_is_zero(self):
        state = RandaoState(VALIDATORS)
        assert state.get_randao(0) == 0

    def test_single_reveal_xor(self):
        state = RandaoState(VALIDATORS)
        reveal = 0xABCD1234
        state.finalize_epoch(1, [reveal])
        # R_1 = R_0 XOR reveal = 0 XOR reveal = reveal
        assert state.get_randao(1) == reveal

    def test_multiple_reveals_xor(self):
        state = RandaoState(VALIDATORS)
        reveals = [0x1111, 0x2222, 0x4444]
        expected = 0x1111 ^ 0x2222 ^ 0x4444
        state.finalize_epoch(1, reveals)
        assert state.get_randao(1) == expected

    def test_sequential_epochs_chain(self):
        state = RandaoState(VALIDATORS)
        r1_reveals = [0xABCD]
        r2_reveals = [0x1234]
        state.finalize_epoch(1, r1_reveals)
        state.finalize_epoch(2, r2_reveals)
        r1 = state.get_randao(1)
        r2 = state.get_randao(2)
        assert r1 == 0xABCD
        assert r2 == r1 ^ 0x1234

    def test_empty_epoch_unchanged(self):
        state = RandaoState(VALIDATORS)
        state.finalize_epoch(1, [0xDEAD])
        state.finalize_epoch(2, [])        # missed epoch
        r1 = state.get_randao(1)
        r2 = state.get_randao(2)
        assert r2 == r1   # XOR with 0 = same


# ---------------------------------------------------------------------------
# test_randao_excludes_reorg
# ---------------------------------------------------------------------------

class TestRandaoExcludesReorg:
    def test_reorged_reveal_not_counted(self):
        """
        Build two chains. The reorged chain's reveal must NOT be in R_e.
        """
        tree = BlockTree()
        fc = ForkChoice(tree)

        # Canonical branch: slot 1
        b_canonical = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b_canonical.randao_reveal = 0xAAAA

        # Reorged branch: slot 1 (sibling)
        b_reorg = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)
        b_reorg.randao_reveal = 0xBBBB

        # Attestations push canonical branch as head
        for v in range(7):
            fc.attest(v, b_canonical.id)
        fc.attest(7, b_reorg.id)

        # Collect canonical reveals for epoch 0
        canonical_chain = fc.canonical_chain()
        reveals = [
            blk.randao_reveal
            for blk in canonical_chain
            if blk.id != BlockTree.GENESIS_ID
        ]

        # Only canonical block should be in reveals
        assert 0xAAAA in reveals
        assert 0xBBBB not in reveals


# ---------------------------------------------------------------------------
# test_randao_missed_block
# ---------------------------------------------------------------------------

class TestRandaoMissedBlock:
    def test_missed_slot_contributes_nothing(self):
        state = RandaoState(VALIDATORS)
        # Epoch 1: only one block proposed (slot 0 canonical, rest missed)
        state.finalize_epoch(1, [0x1234])
        # Epoch 2: all slots missed
        state.finalize_epoch(2, [])

        r1 = state.get_randao(1)
        r2 = state.get_randao(2)
        assert r1 == 0x1234
        assert r2 == r1   # no reveals → XOR identity


# ---------------------------------------------------------------------------
# test_proposer_determinism
# ---------------------------------------------------------------------------

class TestProposerDeterminism:
    def test_same_seed_same_schedule(self):
        s1 = compute_proposer_schedule(1, 0xCAFE, VALIDATORS)
        s2 = compute_proposer_schedule(1, 0xCAFE, VALIDATORS)
        assert s1 == s2

    def test_different_seed_different_schedule(self):
        s1 = compute_proposer_schedule(1, 0xCAFE, VALIDATORS)
        s2 = compute_proposer_schedule(1, 0xBEEF, VALIDATORS)
        assert s1 != s2

    def test_schedule_length(self):
        s = compute_proposer_schedule(1, 0, VALIDATORS)
        assert len(s) == SLOTS_PER_EPOCH

    def test_schedule_contains_only_validators(self):
        s = compute_proposer_schedule(1, 0, VALIDATORS)
        for v in s:
            assert v in VALIDATORS

    def test_epoch_seed_deterministic(self):
        seed1 = _epoch_seed(5, 0xABCD)
        seed2 = _epoch_seed(5, 0xABCD)
        assert seed1 == seed2

    def test_epoch_seed_differs_by_epoch(self):
        seed1 = _epoch_seed(5, 0xABCD)
        seed2 = _epoch_seed(6, 0xABCD)
        assert seed1 != seed2

    def test_schedule_ready_after_finalize(self):
        state = RandaoState(VALIDATORS)
        state.finalize_epoch(0, [])
        assert state.is_schedule_ready(2)   # epoch 0 makes schedule for epoch 2

    def test_schedule_not_ready_with_vdf_pending(self):
        state = RandaoState(VALIDATORS)
        state.finalize_epoch(0, [], use_vdf=True)
        assert not state.is_schedule_ready(2)

    def test_schedule_ready_after_vdf_complete(self):
        state = RandaoState(VALIDATORS)
        state.finalize_epoch(0, [], use_vdf=True)
        # simulate VDF completing with some output
        state.complete_vdf(0, vdf_output=0x9999)
        assert state.is_schedule_ready(2)
