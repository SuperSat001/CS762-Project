from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (
    ExperimentSpec,
    GRAPHS_DIR,
    configure_matplotlib_env,
    ensure_directories,
    load_results,
    pick_tree_visual_seed,
    run_instrumented_simulation,
)

configure_matplotlib_env()

import matplotlib.pyplot as plt


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({"figure.dpi": 180, "savefig.dpi": 220, "font.size": 10})


def _compute_branch_positions(simulator) -> dict[int, float]:
    canonical = simulator.fork_choice.canonical_chain()
    canonical_ids = {block.id for block in canonical}
    positions = {block.id: (block.slot, 0.0) for block in canonical}
    next_offset = 1
    sibling_offsets: dict[int, int] = defaultdict(int)

    blocks = sorted((block for block in simulator.tree.all_blocks() if block.id != 0), key=lambda block: (block.slot, block.id))
    for block in blocks:
        if block.id in canonical_ids:
            continue
        parent = simulator.tree.get(block.parent_id)
        parent_y = positions.get(parent.id, (parent.slot, 0.0))[1]
        depth = sibling_offsets[parent.id] + 1
        sibling_offsets[parent.id] += 1
        if parent.id in canonical_ids:
            y = -float(next_offset)
            next_offset += 1
        else:
            y = parent_y - 0.28 * depth
        positions[block.id] = (block.slot, y)
    positions[0] = (0, 0.0)
    return positions


def _node_color(simulator, block_id: int, canonical_ids: set[int]) -> str:
    if block_id not in canonical_ids:
        return "#d9d9d9"
    block = simulator.tree.get(block_id)
    return "#e45756" if block.proposer_id in simulator.analysis_adversarial_ids else "#4c78a8"


def _draw_tree(ax, simulator, title: str) -> None:
    canonical = simulator.fork_choice.canonical_chain()
    canonical_ids = {block.id for block in canonical}
    canonical_edges = {(block.parent_id, block.id) for block in canonical if block.parent_id is not None}
    positions = _compute_branch_positions(simulator)

    for block in sorted(simulator.tree.all_blocks(), key=lambda blk: (blk.slot, blk.id)):
        if block.parent_id is None:
            continue
        x0, y0 = positions[block.parent_id]
        x1, y1 = positions[block.id]
        is_canonical_edge = (block.parent_id, block.id) in canonical_edges
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="black" if is_canonical_edge else "#c7c7c7",
            linewidth=2.5 if is_canonical_edge else 1.0,
            alpha=1.0 if is_canonical_edge else 0.9,
            zorder=1,
        )

    xs = []
    ys = []
    colors = []
    sizes = []
    for block in sorted(simulator.tree.all_blocks(), key=lambda blk: (blk.slot, blk.id)):
        if block.id == 0:
            continue
        x, y = positions[block.id]
        xs.append(x)
        ys.append(y)
        colors.append(_node_color(simulator, block.id, canonical_ids))
        sizes.append(35 if block.id in canonical_ids else 24)

    ax.scatter(xs, ys, c=colors, s=sizes, edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Fork depth")
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_ylim(min(ys + [0]) - 0.7, 0.8)


def generate_blockchain_figure() -> None:
    _apply_style()
    ensure_directories()
    results = load_results()
    seed = pick_tree_visual_seed(results, alpha=0.3)

    spec_no_vdf = ExperimentSpec(strategy_name="forking", alpha=0.3, seed=seed, vdf_delay=0, num_epochs=3)
    spec_vdf = ExperimentSpec(strategy_name="forking", alpha=0.3, seed=seed, vdf_delay=32, num_epochs=3)
    sim_no_vdf, row_no_vdf = run_instrumented_simulation(spec_no_vdf)
    sim_vdf, row_vdf = run_instrumented_simulation(spec_vdf)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _draw_tree(
        axes[0],
        sim_no_vdf,
        f"Forking without VDF\nforks={row_no_vdf['forks']} reorgs={row_no_vdf['reorgs']}",
    )
    _draw_tree(
        axes[1],
        sim_vdf,
        f"Forking with VDF\nforks={row_vdf['forks']} reorgs={row_vdf['reorgs']}",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Canonical honest", markerfacecolor="#4c78a8", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="Canonical adversarial", markerfacecolor="#e45756", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="Orphaned / reorged", markerfacecolor="#d9d9d9", markersize=7),
        plt.Line2D([0], [0], color="black", linewidth=2.5, label="Canonical path"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(GRAPHS_DIR / "blockchain_tree.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    generate_blockchain_figure()


if __name__ == "__main__":
    main()
