"""
PART 3: Simplified LMD-GHOST fork choice — incremental weight variant.

Rules:
* Each validator has exactly ONE latest attestation (latest-message rule).
* Weight of a node = number of validators whose latest attestation
  is in the subtree rooted at that node.
* Head = start at genesis, repeatedly move to child with highest weight.
  Tie-break: lower block id wins (deterministic).

Performance:
* Incremental score maintenance: when a validator switches attestation
  from A to B we walk A's ancestors and decrement, then B's ancestors
  and increment.  This makes attest() O(chain_depth) and head()
  O(chain_depth) — vs. the naïve O(validators × chain_depth²) per head.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .block import Block, BlockTree


class ForkChoice:
    def __init__(self, tree: BlockTree) -> None:
        self._tree = tree
        # validator_id → block_id of their latest attested block
        self._latest_attestation: Dict[int, int] = {}
        # block_id → subtree attestation score
        self._scores: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Attestations
    # ------------------------------------------------------------------

    def attest(self, validator_id: int, block_id: int) -> None:
        """Record (or update) the latest attestation for a validator."""
        if block_id not in self._tree:
            raise KeyError(f"Block {block_id} unknown to fork choice")

        old = self._latest_attestation.get(validator_id)
        if old == block_id:
            return  # no change

        # Subtract from old chain
        if old is not None:
            self._adjust(old, -1)

        # Add to new chain
        self._adjust(block_id, +1)
        self._latest_attestation[validator_id] = block_id

    def _adjust(self, block_id: int, delta: int) -> None:
        """Walk from block_id to genesis, adjusting scores by delta."""
        current: Optional[int] = block_id
        while current is not None:
            self._scores[current] = self._scores.get(current, 0) + delta
            blk = self._tree._blocks.get(current)
            if blk is None:
                break
            current = blk.parent_id

    # ------------------------------------------------------------------
    # Weight query
    # ------------------------------------------------------------------

    def compute_weight(self, block_id: int) -> int:
        """Return cached subtree attestation score."""
        return self._scores.get(block_id, 0)

    # ------------------------------------------------------------------
    # Head selection  O(chain_depth)
    # ------------------------------------------------------------------

    def head(self) -> Block:
        """Return the current head block via LMD-GHOST traversal."""
        current_id = BlockTree.GENESIS_ID
        while True:
            current_block = self._tree.get(current_id)
            if not current_block.children:
                return current_block
            best_child_id = max(
                current_block.children,
                key=lambda cid: (self.compute_weight(cid), -cid),
            )
            current_id = best_child_id

    def head_id(self) -> int:
        return self.head().id

    # ------------------------------------------------------------------
    # Reorg detection helper
    # ------------------------------------------------------------------

    def is_on_canonical(self, block_id: int) -> bool:
        """True if block_id is an ancestor-or-equal of the current head."""
        return self._tree.is_ancestor(block_id, self.head_id())

    # ------------------------------------------------------------------
    # Canonical chain
    # ------------------------------------------------------------------

    def canonical_chain(self) -> List[Block]:
        """Return the chain of blocks from genesis to current head."""
        return self._tree.ancestors(self.head_id())

    def latest_attestation(self, validator_id: int) -> Optional[int]:
        return self._latest_attestation.get(validator_id)
