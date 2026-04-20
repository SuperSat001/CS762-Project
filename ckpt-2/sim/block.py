"""
PART 1 & 2: Block model and chain tree.

A Block carries only consensus-relevant fields:
  id, parent_id, slot, proposer_id, randao_reveal, weight, children

No transactions, no state root, no gas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

@dataclass
class Block:
    id: int
    parent_id: Optional[int]
    slot: int
    proposer_id: int
    randao_reveal: int          # 256-bit integer
    weight: int = 0             # attestation weight accumulated
    children: List[int] = field(default_factory=list)   # child block ids

    def __repr__(self) -> str:
        return (
            f"Block(id={self.id}, slot={self.slot}, "
            f"proposer={self.proposer_id}, parent={self.parent_id})"
        )


def compute_randao_reveal(secret: int, epoch: int) -> int:
    """BLS-reveal simplified as SHA-256(secret || epoch) → int."""
    data = secret.to_bytes(32, "big") + epoch.to_bytes(8, "big")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# BlockTree  (immutable structure; fork choice lives in fork_choice.py)
# ---------------------------------------------------------------------------

class BlockTree:
    """
    Stores all known blocks (honest + private).
    The *canonical* chain is determined by fork choice, not here.
    """

    GENESIS_SLOT = 0
    GENESIS_ID   = 0

    def __init__(self) -> None:
        genesis = Block(
            id=self.GENESIS_ID,
            parent_id=None,
            slot=self.GENESIS_SLOT,
            proposer_id=-1,
            randao_reveal=0,
        )
        self._blocks: Dict[int, Block] = {self.GENESIS_ID: genesis}
        self._next_id = 1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_block(
        self,
        parent_id: int,
        slot: int,
        proposer_id: int,
        randao_reveal: int,
    ) -> Block:
        """Create a new block, wire parent→child, return it."""
        if parent_id not in self._blocks:
            raise KeyError(f"Parent block {parent_id} not in tree")

        blk = Block(
            id=self._next_id,
            parent_id=parent_id,
            slot=slot,
            proposer_id=proposer_id,
            randao_reveal=randao_reveal,
        )
        self._next_id += 1
        self._blocks[blk.id] = blk
        self._blocks[parent_id].children.append(blk.id)
        return blk

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, block_id: int) -> Block:
        return self._blocks[block_id]

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._blocks

    def all_blocks(self) -> List[Block]:
        return list(self._blocks.values())

    def ancestors(self, block_id: int) -> List[Block]:
        """Return chain from genesis to block_id (inclusive), oldest first."""
        chain: List[Block] = []
        current = self._blocks.get(block_id)
        while current is not None:
            chain.append(current)
            if current.parent_id is None:
                break
            current = self._blocks.get(current.parent_id)
        chain.reverse()
        return chain

    def is_ancestor(self, ancestor_id: int, descendant_id: int) -> bool:
        """Return True if ancestor_id is on the path to descendant_id."""
        current_id: Optional[int] = descendant_id
        while current_id is not None:
            if current_id == ancestor_id:
                return True
            blk = self._blocks.get(current_id)
            if blk is None:
                return False
            current_id = blk.parent_id
        return False

    def __len__(self) -> int:
        return len(self._blocks)
