"""
TEST GROUP 1: Block + Chain
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.block import Block, BlockTree, compute_randao_reveal


# ---------------------------------------------------------------------------
# test_block_creation
# ---------------------------------------------------------------------------

class TestBlockCreation:
    def test_fields_set_correctly(self):
        tree = BlockTree()
        blk = tree.add_block(
            parent_id=BlockTree.GENESIS_ID,
            slot=1,
            proposer_id=42,
            randao_reveal=0xDEADBEEF,
        )
        assert blk.slot == 1
        assert blk.proposer_id == 42
        assert blk.randao_reveal == 0xDEADBEEF
        assert blk.parent_id == BlockTree.GENESIS_ID
        assert blk.weight == 0
        assert blk.children == []

    def test_block_id_is_unique(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(BlockTree.GENESIS_ID, 2, 1, 0)
        assert b1.id != b2.id

    def test_genesis_exists(self):
        tree = BlockTree()
        genesis = tree.get(BlockTree.GENESIS_ID)
        assert genesis.slot == BlockTree.GENESIS_SLOT
        assert genesis.parent_id is None

    def test_randao_reveal_deterministic(self):
        r1 = compute_randao_reveal(secret=999, epoch=3)
        r2 = compute_randao_reveal(secret=999, epoch=3)
        assert r1 == r2

    def test_randao_reveal_differs_by_epoch(self):
        r1 = compute_randao_reveal(secret=999, epoch=3)
        r2 = compute_randao_reveal(secret=999, epoch=4)
        assert r1 != r2

    def test_block_not_in_empty_tree(self):
        tree = BlockTree()
        assert 999 not in tree

    def test_add_block_unknown_parent_raises(self):
        tree = BlockTree()
        with pytest.raises(KeyError):
            tree.add_block(parent_id=999, slot=1, proposer_id=0, randao_reveal=0)


# ---------------------------------------------------------------------------
# test_chain_linking
# ---------------------------------------------------------------------------

class TestChainLinking:
    def test_parent_child_wired(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(b1.id, 2, 1, 0)
        assert b1.id in tree.get(BlockTree.GENESIS_ID).children
        assert b2.id in tree.get(b1.id).children

    def test_ancestors_returns_correct_chain(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(b1.id, 2, 1, 0)
        b3 = tree.add_block(b2.id, 3, 2, 0)
        chain = tree.ancestors(b3.id)
        ids = [b.id for b in chain]
        assert ids == [BlockTree.GENESIS_ID, b1.id, b2.id, b3.id]

    def test_is_ancestor_true(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(b1.id, 2, 1, 0)
        assert tree.is_ancestor(BlockTree.GENESIS_ID, b2.id)
        assert tree.is_ancestor(b1.id, b2.id)

    def test_is_ancestor_false(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(BlockTree.GENESIS_ID, 2, 1, 0)  # sibling
        assert not tree.is_ancestor(b1.id, b2.id)
        assert not tree.is_ancestor(b2.id, b1.id)


# ---------------------------------------------------------------------------
# test_fork_tree
# ---------------------------------------------------------------------------

class TestForkTree:
    def test_multiple_children_from_genesis(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)
        genesis = tree.get(BlockTree.GENESIS_ID)
        assert b1.id in genesis.children
        assert b2.id in genesis.children
        assert len(genesis.children) == 2

    def test_fork_siblings_have_same_parent(self):
        tree = BlockTree()
        b1 = tree.add_block(BlockTree.GENESIS_ID, 1, 0, 0)
        b2 = tree.add_block(BlockTree.GENESIS_ID, 1, 1, 0)
        assert tree.get(b1.id).parent_id == BlockTree.GENESIS_ID
        assert tree.get(b2.id).parent_id == BlockTree.GENESIS_ID

    def test_total_block_count(self):
        tree = BlockTree()
        for i in range(5):
            tree.add_block(BlockTree.GENESIS_ID, i + 1, i, 0)
        # genesis + 5 blocks
        assert len(tree) == 6
