from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from sim.block import BlockTree, compute_randao_reveal
from sim.randao import SLOTS_PER_EPOCH, compute_proposer_schedule
from sim.simulator import Simulator, StateView, Validator
from sim.strategies import ForkingStrategy, HonestStrategy, ProposalAction, Strategy


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"
GRAPHS_DIR = RESULTS_DIR / "graphs"
DEBUG_DIR = RESULTS_DIR / "debug"
MPLCONFIGDIR = RESULTS_DIR / ".mplconfig"
XDG_CACHE_HOME = RESULTS_DIR / ".cache"

NUM_VALIDATORS = 100
MAX_EPOCHS = 200
EXPERIMENT_EPOCHS = 15
VISUALIZATION_EPOCHS = 3
SEEDS = (1, 2, 3)
ALPHAS = (0.1, 0.2, 0.3)
VDF_DELAYS = (0, 32)
STRATEGIES = ("honest_baseline", "adaptive_mixing", "forking")

RESULTS_CSV = DATA_DIR / "simulation_results.csv"


def ensure_directories() -> None:
    for path in (RESULTS_DIR, DATA_DIR, PLOTS_DIR, GRAPHS_DIR, DEBUG_DIR, MPLCONFIGDIR, XDG_CACHE_HOME):
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib_env() -> None:
    ensure_directories()
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))


@dataclass(frozen=True)
class ExperimentSpec:
    strategy_name: str
    alpha: float
    seed: int
    vdf_delay: int
    num_epochs: int = EXPERIMENT_EPOCHS

    @property
    def use_vdf(self) -> bool:
        return self.vdf_delay > 0


class CoalitionMixPlanner:
    """
    Chooses which tail-slot adversarial reveals to include in R_e.

    The planner exploits only currently knowable information:
    * proposer schedule for the current epoch
    * validators' own secrets
    * deterministic honest behavior

    When VDF is enabled the attack cannot evaluate epoch+2 schedules in time,
    so the analysis strategy simply proposes honestly.
    """

    def __init__(self, adversarial_ids: Sequence[int], tail_slots: int = 12) -> None:
        self.adversarial_ids = set(adversarial_ids)
        self.tail_slots = tail_slots
        self._sim: Optional[Simulator] = None
        self._cached_epoch_actions: Dict[int, Dict[Tuple[int, int], bool]] = {}

    def bind(self, simulator: Simulator) -> None:
        self._sim = simulator

    def should_skip(self, epoch: int, slot: int, validator_id: int) -> bool:
        plan = self._plan_epoch(epoch)
        return plan.get((slot, validator_id), False)

    def _plan_epoch(self, epoch: int) -> Dict[Tuple[int, int], bool]:
        if epoch in self._cached_epoch_actions:
            return self._cached_epoch_actions[epoch]
        if self._sim is None:
            raise RuntimeError("Planner is not bound to a simulator")

        schedule = self._sim.randao.get_schedule(epoch)
        if schedule is None:
            self._cached_epoch_actions[epoch] = {}
            return {}

        prev_randao = self._sim.randao.get_randao(epoch - 1) if epoch > 0 else 0
        base_randao = prev_randao
        candidate_reveals: List[Tuple[int, int, int]] = []
        epoch_start = epoch * SLOTS_PER_EPOCH
        tail_start_index = SLOTS_PER_EPOCH - self.tail_slots

        for offset, proposer_id in enumerate(schedule):
            slot = epoch_start + offset
            validator = self._sim.validators[proposer_id]
            reveal = compute_randao_reveal(validator.secret, epoch)
            if proposer_id in self.adversarial_ids and offset >= tail_start_index:
                candidate_reveals.append((slot, proposer_id, reveal))
            else:
                base_randao ^= reveal

        all_validator_ids = list(self._sim.validators.keys())
        best_mask = 0
        best_share = -1.0

        for mask in range(1 << len(candidate_reveals)):
            candidate_randao = base_randao
            for index, (_, _, reveal) in enumerate(candidate_reveals):
                if mask & (1 << index):
                    candidate_randao ^= reveal
            target_schedule = compute_proposer_schedule(epoch + 2, candidate_randao, all_validator_ids)
            share = sum(1 for proposer_id in target_schedule if proposer_id in self.adversarial_ids) / len(target_schedule)
            if share > best_share:
                best_share = share
                best_mask = mask

        plan: Dict[Tuple[int, int], bool] = {}
        for index, (slot, proposer_id, _) in enumerate(candidate_reveals):
            include_reveal = bool(best_mask & (1 << index))
            plan[(slot, proposer_id)] = not include_reveal

        self._cached_epoch_actions[epoch] = plan
        return plan


class AdaptiveMixingStrategy(Strategy):
    """Analysis-only strategy that skips tail-slot proposals when it improves epoch+2 share."""

    def __init__(self, planner: CoalitionMixPlanner, tail_slots: int = 12) -> None:
        self.planner = planner
        self.tail_slots = tail_slots
        self._sim: Optional[Simulator] = None
        self._validator_id: Optional[int] = None

    def bind(self, simulator: Simulator, validator_id: int) -> None:
        self._sim = simulator
        self._validator_id = validator_id

    def propose_action(self, state_view: StateView, slot: int) -> ProposalAction:
        if self._sim is None or self._validator_id is None:
            raise RuntimeError("AdaptiveMixingStrategy must be bound before use")
        if self._sim.use_vdf:
            return ProposalAction.PROPOSE
        epoch_slot = slot % SLOTS_PER_EPOCH
        if epoch_slot < SLOTS_PER_EPOCH - self.tail_slots:
            return ProposalAction.PROPOSE
        epoch = slot // SLOTS_PER_EPOCH
        if self.planner.should_skip(epoch, slot, self._validator_id):
            return ProposalAction.SKIP
        return ProposalAction.PROPOSE

    def publish_private(self, state_view: StateView, slot: int) -> bool:
        return False


class InstrumentedSimulator(Simulator):
    """Simulator wrapper that records per-slot actions and block metadata."""

    def __init__(self, validators: List[Validator], adversarial_ids: Iterable[int], **kwargs) -> None:
        super().__init__(validators, **kwargs)
        self.analysis_adversarial_ids = set(adversarial_ids)
        self.slot_logs: List[dict] = []
        self.block_annotations: Dict[int, dict] = {
            BlockTree.GENESIS_ID: {"action": "genesis", "is_adversarial": False, "slot": 0}
        }
        self.slot_actions: Dict[int, dict] = {}

    def _handle_propose(self, event) -> None:
        slot = event.payload["slot"]
        epoch = slot // SLOTS_PER_EPOCH

        proposer_id = self.randao.proposer_for_slot(slot)
        record = {
            "slot": slot,
            "epoch": epoch,
            "proposer_id": proposer_id,
            "proposer_type": "A" if proposer_id in self.analysis_adversarial_ids else "H",
            "action": "skip",
            "head_before": self.fork_choice.head_id(),
        }

        if proposer_id is None:
            self.metrics.record_missed_slot()
            record["action"] = "skip"
            self.slot_actions[slot] = record
            return

        validator = self.validators.get(proposer_id)
        if validator is None:
            self.metrics.record_missed_slot()
            self.slot_actions[slot] = record
            return

        state_view = self._state_view(proposer_id)
        action = validator.strategy.propose_action(state_view, slot)
        record["action"] = action.name.lower()

        if action == ProposalAction.SKIP:
            self.metrics.record_missed_slot()
            self.slot_actions[slot] = record
            return

        if action == ProposalAction.PRIVATE:
            parent_id = self._private_tip(proposer_id)
        else:
            parent_id = self.fork_choice.head_id()

        parent_block = self.tree.get(parent_id)
        if parent_block.children:
            self.metrics.record_fork(parent_id)

        reveal = compute_randao_reveal(secret=validator.secret, epoch=epoch)
        block = self.tree.add_block(
            parent_id=parent_id,
            slot=slot,
            proposer_id=proposer_id,
            randao_reveal=reveal,
        )

        is_adversarial = proposer_id in self.analysis_adversarial_ids
        self.metrics.record_block(proposer_id, is_adversarial=is_adversarial)
        self.block_annotations[block.id] = {
            "action": action.name.lower(),
            "is_adversarial": is_adversarial,
            "slot": slot,
        }

        if action == ProposalAction.PRIVATE:
            self._private_blocks.setdefault(proposer_id, []).append(block.id)
            if isinstance(validator.strategy, ForkingStrategy):
                validator.strategy.increment_lead()

        record["block_id"] = block.id
        record["parent_id"] = parent_id
        self.slot_actions[slot] = record

    def _handle_fork_choice(self, event) -> None:
        new_head = self.fork_choice.head()
        self.metrics.record_head_update(new_head.id, new_head.parent_id)

        slot = event.payload["slot"]
        slot_record = dict(self.slot_actions.get(slot, {"slot": slot, "action": "skip"}))
        slot_record["head_after"] = new_head.id
        slot_record["fork_depth"] = current_fork_depth(self.tree, self.fork_choice)
        slot_record["partial_randao"] = f"0x{partial_randao_for_slot(self, slot):064x}"
        self.slot_logs.append(slot_record)


def build_validators(strategy_name: str, alpha: float, seed: int) -> Tuple[List[Validator], List[int], Optional[CoalitionMixPlanner]]:
    adversarial_count = int(NUM_VALIDATORS * alpha)
    honest_count = NUM_VALIDATORS - adversarial_count
    adversarial_ids = list(range(honest_count, NUM_VALIDATORS))
    validators: List[Validator] = []

    for validator_id in range(honest_count):
        validators.append(
            Validator(
                id=validator_id,
                stake=1,
                strategy=HonestStrategy(),
                secret=(seed * 10_000) + validator_id + 1,
            )
        )

    planner: Optional[CoalitionMixPlanner] = None
    if strategy_name == "adaptive_mixing":
        planner = CoalitionMixPlanner(adversarial_ids, tail_slots=12)

    for offset, validator_id in enumerate(adversarial_ids):
        secret = (seed * 10_000) + honest_count + offset + 1
        if strategy_name == "honest_baseline":
            strategy: Strategy = HonestStrategy()
        elif strategy_name == "adaptive_mixing":
            strategy = AdaptiveMixingStrategy(planner, tail_slots=12)
        elif strategy_name == "forking":
            strategy = ForkingStrategy(release_threshold=3, release_probability=0.0)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        validators.append(Validator(id=validator_id, stake=1, strategy=strategy, secret=secret))

    return validators, adversarial_ids, planner


def bind_analysis_context(simulator: Simulator, adversarial_ids: Sequence[int], planner: Optional[CoalitionMixPlanner]) -> None:
    if planner is not None:
        planner.bind(simulator)
    for validator_id in adversarial_ids:
        strategy = simulator.validators[validator_id].strategy
        if hasattr(strategy, "bind"):
            strategy.bind(simulator, validator_id)


def run_instrumented_simulation(spec: ExperimentSpec) -> Tuple[InstrumentedSimulator, dict]:
    if spec.num_epochs > MAX_EPOCHS:
        raise ValueError(f"Requested {spec.num_epochs} epochs, max is {MAX_EPOCHS}")

    validators, adversarial_ids, planner = build_validators(spec.strategy_name, spec.alpha, spec.seed)
    simulator = InstrumentedSimulator(
        validators,
        adversarial_ids=adversarial_ids,
        num_epochs=spec.num_epochs,
        use_vdf=spec.use_vdf,
        vdf_delay_slots=spec.vdf_delay,
    )
    bind_analysis_context(simulator, adversarial_ids, planner)

    start = time.perf_counter()
    metrics = simulator.run()
    runtime = time.perf_counter() - start

    reward = proposer_share(simulator, adversarial_ids)
    row = {
        "strategy": spec.strategy_name,
        "alpha": spec.alpha,
        "seed": spec.seed,
        "vdf_delay": spec.vdf_delay,
        "use_vdf": spec.use_vdf,
        "num_epochs": spec.num_epochs,
        "total_slots": simulator.total_slots,
        "runtime_seconds": runtime,
        "adversarial_reward": reward,
        "slot_gain": reward - spec.alpha,
        "fork_rate": metrics.forks / simulator.total_slots,
        "reorg_rate": metrics.reorgs / simulator.total_slots,
        "missed_slots": metrics.missed_slots,
        "forks": metrics.forks,
        "reorgs": metrics.reorgs,
        "total_blocks": metrics.total_blocks,
        "canonical_adversarial_fraction": canonical_adversarial_fraction(simulator, adversarial_ids),
    }
    return simulator, row


def proposer_share(simulator: Simulator, adversarial_ids: Sequence[int]) -> float:
    adversarial = set(adversarial_ids)
    adversarial_slots = 0
    total_slots = 0
    for epoch in range(2, simulator.num_epochs):
        schedule = simulator.randao.get_schedule(epoch)
        if schedule is None:
            continue
        total_slots += len(schedule)
        adversarial_slots += sum(1 for proposer_id in schedule if proposer_id in adversarial)
    return adversarial_slots / total_slots if total_slots else 0.0


def canonical_adversarial_fraction(simulator: Simulator, adversarial_ids: Sequence[int]) -> float:
    adversarial = set(adversarial_ids)
    canonical = [block for block in simulator.fork_choice.canonical_chain() if block.id != BlockTree.GENESIS_ID]
    if not canonical:
        return 0.0
    return sum(1 for block in canonical if block.proposer_id in adversarial) / len(canonical)


def partial_randao_for_slot(simulator: Simulator, slot: int) -> int:
    epoch = slot // SLOTS_PER_EPOCH
    prev_randao = simulator.randao.get_randao(epoch - 1) if epoch > 0 else 0
    canonical = simulator.fork_choice.canonical_chain()
    epoch_start = epoch * SLOTS_PER_EPOCH
    reveals = [
        block.randao_reveal
        for block in canonical
        if block.id != BlockTree.GENESIS_ID and epoch_start <= block.slot <= slot
    ]
    value = prev_randao
    for reveal in reveals:
        value ^= reveal
    return value


def current_fork_depth(tree: BlockTree, fork_choice) -> int:
    canonical_ids = {block.id for block in fork_choice.canonical_chain()}
    max_depth = 0
    for block in tree.all_blocks():
        if block.id in canonical_ids:
            continue
        depth = 0
        current = block
        while current.parent_id is not None and current.parent_id not in canonical_ids:
            depth += 1
            current = tree.get(current.parent_id)
        if current.parent_id is not None:
            depth += 1
        max_depth = max(max_depth, depth)
    return max_depth


def block_time_series(simulator: InstrumentedSimulator, adversarial_ids: Sequence[int]) -> pd.DataFrame:
    adversarial = set(adversarial_ids)
    blocks = sorted(
        (block for block in simulator.tree.all_blocks() if block.id != BlockTree.GENESIS_ID),
        key=lambda block: (block.slot, block.id),
    )
    cumulative_honest = 0
    cumulative_adversarial = 0
    rows = []
    for block in blocks:
        if block.proposer_id in adversarial:
            cumulative_adversarial += 1
        else:
            cumulative_honest += 1
        rows.append(
            {
                "slot": block.slot,
                "block_id": block.id,
                "cumulative_honest_blocks": cumulative_honest,
                "cumulative_adversarial_blocks": cumulative_adversarial,
            }
        )
    return pd.DataFrame(rows)


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV)


def save_results(df: pd.DataFrame) -> None:
    ensure_directories()
    df.to_csv(RESULTS_CSV, index=False)


def aggregated_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["strategy", "alpha", "vdf_delay"], as_index=False)
        .agg(
            adversarial_reward=("adversarial_reward", "mean"),
            slot_gain=("slot_gain", "mean"),
            fork_rate=("fork_rate", "mean"),
            reorg_rate=("reorg_rate", "mean"),
            missed_slots=("missed_slots", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
    )


def pick_tree_visual_seed(df: pd.DataFrame, alpha: float = 0.3) -> int:
    subset = df[(df["strategy"] == "forking") & (df["alpha"] == alpha)]
    if subset.empty:
        return SEEDS[0]
    pivot = subset.pivot_table(index="seed", columns="vdf_delay", values="fork_rate", aggfunc="mean")
    if 0 in pivot.columns and 32 in pivot.columns:
        pivot["delta"] = pivot[0] - pivot[32]
        return int(pivot["delta"].sort_values(ascending=False).index[0])
    return int(subset.iloc[0]["seed"])
