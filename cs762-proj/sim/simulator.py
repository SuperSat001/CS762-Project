"""
PART 5 (cont.) + integration: Main discrete-event simulator.

Execution order per slot:
    SLOT_START → BLOCK_PROPOSE → ATTEST → FORK_CHOICE_UPDATE

VDF_COMPLETE fires at an arbitrary future slot when VDF evaluation finishes.
EPOCH_END fires at the end of the last slot of each epoch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .block import Block, BlockTree, compute_randao_reveal
from .events import (
    Event, EventQueue, EventType,
    attest_payload, epoch_end_payload, fork_choice_payload,
    propose_payload, slot_start_payload, vdf_complete_payload,
)
from .fork_choice import ForkChoice
from .metrics import Metrics
from .randao import RandaoState, SLOTS_PER_EPOCH
from .strategies import ForkingStrategy, ProposalAction, Strategy
from .vdf import VDF


# ---------------------------------------------------------------------------
# Validator record
# ---------------------------------------------------------------------------

@dataclass
class Validator:
    id: int
    stake: int
    strategy: Strategy
    secret: int   # used to compute randao_reveal


# ---------------------------------------------------------------------------
# StateView — what strategies can observe (no future info)
# ---------------------------------------------------------------------------

@dataclass
class StateView:
    current_slot: int
    head_id: int
    epoch: int
    # private lead of this validator (only meaningful for ForkingStrategy)
    private_lead: int = 0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Discrete-event Ethereum PoS consensus + RANDAO simulator.

    Parameters
    ----------
    validators       : list of Validator objects
    num_epochs       : how many epochs to run
    use_vdf          : if True, proposer schedule requires VDF to complete
    vdf_delay_slots  : how many slots the VDF takes (default = SLOTS_PER_EPOCH)
    """

    def __init__(
        self,
        validators: List[Validator],
        num_epochs: int = 4,
        use_vdf: bool = False,
        vdf_delay_slots: int = SLOTS_PER_EPOCH,
    ) -> None:
        self.validators: Dict[int, Validator] = {v.id: v for v in validators}
        self.adversarial_ids: Set[int] = {
            v.id for v in validators
            if not isinstance(v.strategy, __import__('sim.strategies', fromlist=['HonestStrategy']).HonestStrategy)
        }
        self.num_epochs = num_epochs
        self.use_vdf = use_vdf
        self.total_slots = num_epochs * SLOTS_PER_EPOCH

        # Core data structures
        self.tree = BlockTree()
        self.fork_choice = ForkChoice(self.tree)
        self.randao = RandaoState([v.id for v in validators])
        self.vdf = VDF(delay_slots=vdf_delay_slots)
        self.metrics = Metrics()
        self.metrics.set_stake_weights({v.id: v.stake for v in validators})

        # Private chain: validator_id → list of private block ids
        self._private_blocks: Dict[int, List[int]] = {}

        # Schedule initial proposers for epochs 0 and 1 using R_0 = 0
        # (before any VDF, the genesis schedule is known)
        all_ids = [v.id for v in validators]
        from .randao import compute_proposer_schedule
        self.randao._schedules[0] = compute_proposer_schedule(0, 0, all_ids)
        self.randao._schedules[1] = compute_proposer_schedule(1, 0, all_ids)

        # Event queue
        self.queue = EventQueue()

        # Current simulation slot
        self.current_slot: int = 0

        # Pre-schedule ALL slots so the queue is fully populated from the start.
        # This allows external code (e.g. tests) to iterate the queue directly
        # without having to call run() first.
        for s in range(self.total_slots):
            self._schedule_slot(s)

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def _schedule_slot(self, slot: int) -> None:
        """Push all four per-slot events."""
        self.queue.push(EventType.SLOT_START,         slot, slot_start_payload(slot))
        self.queue.push(EventType.BLOCK_PROPOSE,      slot, propose_payload(slot, -1))
        self.queue.push(EventType.ATTEST,             slot, attest_payload(slot))
        self.queue.push(EventType.FORK_CHOICE_UPDATE, slot, fork_choice_payload(slot))
        # EPOCH_END at last slot of epoch
        if (slot + 1) % SLOTS_PER_EPOCH == 0:
            epoch = slot // SLOTS_PER_EPOCH
            self.queue.push(EventType.EPOCH_END, slot, epoch_end_payload(epoch))

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> Metrics:
        while not self.queue.is_empty():
            event = self.queue.pop()
            self.current_slot = event.slot
            self._dispatch(event)

        # Record canonical proposals in metrics
        canonical = self.fork_choice.canonical_chain()
        for blk in canonical:
            if blk.id != BlockTree.GENESIS_ID:
                self.metrics.record_canonical_proposal(blk.proposer_id)

        return self.metrics

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        if event.type == EventType.SLOT_START:
            self._handle_slot_start(event)
        elif event.type == EventType.BLOCK_PROPOSE:
            self._handle_propose(event)
        elif event.type == EventType.ATTEST:
            self._handle_attest(event)
        elif event.type == EventType.FORK_CHOICE_UPDATE:
            self._handle_fork_choice(event)
        elif event.type == EventType.EPOCH_END:
            self._handle_epoch_end(event)
        elif event.type == EventType.VDF_COMPLETE:
            self._handle_vdf_complete(event)

    # ------------------------------------------------------------------
    # Handler: SLOT_START
    # ------------------------------------------------------------------

    def _handle_slot_start(self, event: Event) -> None:
        slot = event.payload["slot"]
        # Check if any ForkingStrategy validator wants to publish private chain
        for vid, private_ids in list(self._private_blocks.items()):
            if not private_ids:
                continue
            validator = self.validators[vid]
            sv = self._state_view(vid)
            if validator.strategy.publish_private(sv, slot):
                # Reveal all private blocks → they become public
                # (already in the tree; just remove from private set)
                self._private_blocks[vid] = []
                if isinstance(validator.strategy, ForkingStrategy):
                    validator.strategy.reset_lead()

    # ------------------------------------------------------------------
    # Handler: BLOCK_PROPOSE
    # ------------------------------------------------------------------

    def _handle_propose(self, event: Event) -> None:
        slot = event.payload["slot"]
        epoch = slot // SLOTS_PER_EPOCH

        proposer_id = self.randao.proposer_for_slot(slot)
        if proposer_id is None:
            # Schedule not yet available (VDF pending) — treat as missed
            self.metrics.record_missed_slot()
            return

        validator = self.validators.get(proposer_id)
        if validator is None:
            self.metrics.record_missed_slot()
            return

        sv = self._state_view(proposer_id)
        action = validator.strategy.propose_action(sv, slot)

        if action == ProposalAction.SKIP:
            self.metrics.record_missed_slot()
            return

        # Determine parent: adversarial ForkingStrategy builds on private tip
        if action == ProposalAction.PRIVATE:
            parent_id = self._private_tip(proposer_id)
        else:
            parent_id = self.fork_choice.head_id()

        # Check if this creates a fork (parent already has a child)
        parent_block = self.tree.get(parent_id)
        if parent_block.children:
            self.metrics.record_fork(parent_id)

        reveal = compute_randao_reveal(secret=validator.secret, epoch=epoch)
        blk = self.tree.add_block(
            parent_id=parent_id,
            slot=slot,
            proposer_id=proposer_id,
            randao_reveal=reveal,
        )

        is_adv = proposer_id in self.adversarial_ids
        self.metrics.record_block(proposer_id, is_adversarial=is_adv)

        if action == ProposalAction.PRIVATE:
            # Keep block private
            self._private_blocks.setdefault(proposer_id, []).append(blk.id)
            if isinstance(validator.strategy, ForkingStrategy):
                validator.strategy.increment_lead()

    # ------------------------------------------------------------------
    # Handler: ATTEST
    # ------------------------------------------------------------------

    def _handle_attest(self, event: Event) -> None:
        slot = event.payload["slot"]
        head_id = self.fork_choice.head_id()

        for validator in self.validators.values():
            # Validators attest to the current head (honest rule)
            # Adversarial validators using ForkingStrategy attest to
            # their private tip if they have one, otherwise to head.
            if isinstance(validator.strategy, ForkingStrategy):
                tip = self._private_tip(validator.id)
                # Only attest to private tip if it exists and is ahead
                private_blk = self.tree.get(tip)
                head_blk = self.tree.get(head_id)
                if private_blk.slot >= head_blk.slot:
                    attest_target = tip
                else:
                    attest_target = head_id
            else:
                attest_target = head_id
            self.fork_choice.attest(validator.id, attest_target)

    # ------------------------------------------------------------------
    # Handler: FORK_CHOICE_UPDATE
    # ------------------------------------------------------------------

    def _handle_fork_choice(self, event: Event) -> None:
        new_head = self.fork_choice.head()
        self.metrics.record_head_update(new_head.id, new_head.parent_id)

    # ------------------------------------------------------------------
    # Handler: EPOCH_END
    # ------------------------------------------------------------------

    def _handle_epoch_end(self, event: Event) -> None:
        epoch = event.payload["epoch"]

        # Collect canonical reveals for this epoch
        canonical = self.fork_choice.canonical_chain()
        epoch_start = epoch * SLOTS_PER_EPOCH
        epoch_end   = (epoch + 1) * SLOTS_PER_EPOCH

        reveals = [
            blk.randao_reveal
            for blk in canonical
            if epoch_start <= blk.slot < epoch_end
            and blk.id != BlockTree.GENESIS_ID
        ]

        if self.use_vdf:
            # Compute R_e but block schedule until VDF completes
            raw_r = self.randao.finalize_epoch(epoch, reveals, use_vdf=True)
            vdf_output, ready_slot = self.vdf.eval(raw_r, self.current_slot)
            # Schedule VDF_COMPLETE event
            self.queue.push(
                EventType.VDF_COMPLETE,
                ready_slot,
                vdf_complete_payload(epoch, vdf_output),
            )
        else:
            self.randao.finalize_epoch(epoch, reveals, use_vdf=False)

    # ------------------------------------------------------------------
    # Handler: VDF_COMPLETE
    # ------------------------------------------------------------------

    def _handle_vdf_complete(self, event: Event) -> None:
        epoch = event.payload["epoch"]
        vdf_output = event.payload["vdf_output"]
        self.randao.complete_vdf(epoch, vdf_output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _private_tip(self, validator_id: int) -> int:
        """Return the tip of a validator's private chain, or head."""
        private = self._private_blocks.get(validator_id, [])
        if private:
            return private[-1]
        return self.fork_choice.head_id()

    def _state_view(self, validator_id: int) -> StateView:
        private = self._private_blocks.get(validator_id, [])
        return StateView(
            current_slot=self.current_slot,
            head_id=self.fork_choice.head_id(),
            epoch=self.current_slot // SLOTS_PER_EPOCH,
            private_lead=len(private),
        )

    def _is_adversarial(self, validator_id: int) -> bool:
        return validator_id in self.adversarial_ids
