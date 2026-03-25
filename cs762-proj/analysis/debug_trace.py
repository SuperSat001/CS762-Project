from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import DEBUG_DIR, ExperimentSpec, ensure_directories, run_instrumented_simulation


def _ascii_tree(simulator) -> str:
    canonical_ids = {block.id for block in simulator.fork_choice.canonical_chain()}
    lines = []
    blocks = sorted((block for block in simulator.tree.all_blocks() if block.id != 0), key=lambda block: (block.slot, block.id))
    for block in blocks:
        marker = "*" if block.id in canonical_ids else "-"
        lines.append(
            f"{marker} block={block.id:>3} slot={block.slot:>2} parent={block.parent_id:>3} proposer={block.proposer_id:>3}"
        )
    return "\n".join(lines)


def generate_debug_trace() -> None:
    ensure_directories()
    spec = ExperimentSpec(strategy_name="adaptive_mixing", alpha=0.3, seed=2, vdf_delay=0, num_epochs=3)
    simulator, _ = run_instrumented_simulation(spec)

    lines = [
        "Debug trace for adaptive mixing without VDF",
        f"config: strategy={spec.strategy_name} alpha={spec.alpha} seed={spec.seed} epochs={spec.num_epochs}",
        "",
    ]

    for record in simulator.slot_logs:
        lines.append(
            "slot={slot:>3} proposer={proposer_id!s:>3} type={proposer_type} "
            "action={action:<7} head={head_after:>3} fork_depth={fork_depth:<2} "
            "partial_randao={partial_randao}".format(**record)
        )

    lines.extend(["", "ASCII tree snapshot:", _ascii_tree(simulator)])
    (DEBUG_DIR / "trace.txt").write_text("\n".join(lines))


def main() -> None:
    generate_debug_trace()


if __name__ == "__main__":
    main()
