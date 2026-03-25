"""
TEST GROUP 2: Fork Choice (LMD-GHOST)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.block import BlockTree
from sim.fork_choice import ForkChoice


def _linear_chain(n: int):
    """Build a linear chain of n blocks beyond genesis. Returns (tree, fc, block_ids)."""
    tree = BlockTree()
    fc = ForkChoice(tree)
    ids = []
    parent = BlockTree.GENESIS_ID
    for i in range(n):
        b = tree.add_block(parent, i + 1, i, 0)
        ids.append(b.id)
        parent = b.id
    return tree, fc, ids


# ---------------------------------------------------------------------------
# test_single_chain_head
# ---------------------------------------------------------------------------

class TestSingleChainHead:
    def test_head_is_genesis_with_no_blocks(self):
        tree = BlockTree()
        fc = ForkChoice(tree)
        assert fc.head_id() == BlockTree.GENESIS_ID

    def test_head_is_last_block_linear(self):
        tree, fc, ids = _linear_chain(5)
        # With no attestations, tie-breaks by lowest block id → deepest wins
        # because LMD-GHOST picks max weight; with zero attestations each child
        # weight is 0, so we rely on tie-break (lower id).
        # Add attestation to push the head to the tip.
        fc.attest(validator_id=0, block_id=ids[-1])
        assert fc.head_id() == ids[-1]

    def test_head_follows_attestation(self):
        tree, fc, ids = _linear_chain(3)
        fc.attest(0, ids[2])
        assert fc.head_id() == ids[2]


# ---------------------------------------------------------------------------
# test_fork_choice_weight
# ---------------------------------------------------------------------------

class TestForkChoiceWeight:
    def test_heavier_branch_wins(self):
        tree = BlockTree()
        fc = ForkChoice(tree)

        # Two branches from genesis
        b_a = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)  # branch A
        b_b = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)  # branch B

        # 3 validators vote for A, 1 for B
        fc.attest(0, b_a.id)
        fc.attest(1, b_a.id)
        fc.attest(2, b_a.id)
        fc.attest(3, b_b.id)

        assert fc.head_id() == b_a.id

    def test_equal_weight_tie_break_lower_id(self):
        tree = BlockTree()
        fc = ForkChoice(tree)
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)

        fc.attest(0, b1.id)
        fc.attest(1, b2.id)

        # Tie: pick lower id (b1 < b2 because b1 was added first)
        assert fc.head_id() == min(b1.id, b2.id)

    def test_deeper_chain_wins_with_support(self):
        tree = BlockTree()
        fc = ForkChoice(tree)

        # A: genesis → b1 → b2
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(b1.id, 2, 0, 0)

        # B: genesis → b3  (1 level)
        b3 = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)

        fc.attest(0, b2.id)   # votes for deeper chain
        fc.attest(1, b3.id)

        assert fc.head_id() == b2.id

    def test_compute_weight_counts_subtree(self):
        tree = BlockTree()
        fc = ForkChoice(tree)
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(b1.id, 2, 0, 0)

        fc.attest(0, b2.id)  # supports both b1 and b2 subtree
        fc.attest(1, b1.id)  # supports only b1 subtree

        assert fc.compute_weight(BlockTree.GENESIS_ID) == 2
        assert fc.compute_weight(b1.id) == 2
        assert fc.compute_weight(b2.id) == 1


# ---------------------------------------------------------------------------
# test_latest_attestation_only
# ---------------------------------------------------------------------------

class TestLatestAttestationOnly:
    def test_validator_changing_vote_updates_weight(self):
        tree = BlockTree()
        fc = ForkChoice(tree)

        b_a = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b_b = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)

        fc.attest(0, b_a.id)
        assert fc.compute_weight(b_a.id) == 1
        assert fc.compute_weight(b_b.id) == 0

        # Validator 0 switches to B
        fc.attest(0, b_b.id)
        assert fc.compute_weight(b_a.id) == 0
        assert fc.compute_weight(b_b.id) == 1

    def test_only_latest_vote_counts(self):
        tree = BlockTree()
        fc = ForkChoice(tree)

        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(b1.id, 2, 0, 0)
        b3 = tree.add_block(b2.id, 3, 0, 0)

        fc.attest(0, b1.id)
        fc.attest(0, b2.id)
        fc.attest(0, b3.id)

        # Only b3 should count — weight of b3 = 1, b1/b2 subtree also = 1
        assert fc.compute_weight(b3.id) == 1
        assert fc.latest_attestation(0) == b3.id

    def test_canonical_chain_correct(self):
        tree, fc, ids = _linear_chain(4)
        fc.attest(0, ids[3])
        chain = fc.canonical_chain()
        chain_ids = [b.id for b in chain]
        assert chain_ids[-1] == ids[3]
        assert chain_ids[0] == BlockTree.GENESIS_ID
