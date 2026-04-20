#!/usr/bin/env python3
"""
From-scratch implementation of a RANDao manipulation MDP-inspired simulator.

This script implements:
1) State/action/reward dynamics for tail/head attack-string states.
2) Selfish-mixing and ex-ante forking option generation.
3) Finite-horizon MDP rollouts (expectimax-style Monte Carlo) for policy/value.
4) Simulation/plots approximating the paper's theoretical figures:
   - Figure 7 (expected slots vs stake)
   - Figure 8 (strategic action proportions)
   - Figure 11 (target-slot utility)
   - Figure 12 (throughput degradation)
   - Table 2 (effective stake, PoS columns)

Notes:
- This is a self-contained reimplementation from the paper text (not paper code).
- The paper's empirical chain-history figures (e.g., Figures 9/10/13) require external
  beacon-chain datasets and are not regenerated here.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SLOTS_PER_EPOCH = 32
P_BOOST = 0.4


@dataclasses.dataclass(frozen=True)
class Config:
    tail_len: int = 6
    head_len: int = 2
    lookahead_depth: int = 2
    lookahead_rollouts: int = 8
    epochs_per_u: int = 1200
    burn_in: int = 200
    target_slot: int = 8


@dataclasses.dataclass(frozen=True)
class State:
    """Extended attack string state: (tail.curr + '.' + head.next) + '*' + tail.next."""

    attack_string: str
    next_tail: str


@dataclasses.dataclass(frozen=True)
class Option:
    kind: str  # honest, selfish, fork, regret
    status: str  # C/N string over current tail slots
    sacrifice: int  # missed adversarial tail blocks only


def sample_epoch_string(u: float, rng: np.random.Generator) -> str:
    return "".join("A" if x else "H" for x in (rng.random(SLOTS_PER_EPOCH) < u))


def build_state(curr_epoch: str, next_epoch: str, cfg: Config) -> State:
    tail = curr_epoch[-cfg.tail_len :]
    head = next_epoch[: cfg.head_len]
    next_tail = next_epoch[-cfg.tail_len :]
    return State(f"{tail}.{head}", next_tail)


def transition_state(state: State, sampled_future_epoch: str, cfg: Config) -> State:
    next_attack = f"{state.next_tail}.{sampled_future_epoch[: cfg.head_len]}"
    next_tail = sampled_future_epoch[-cfg.tail_len :]
    return State(next_attack, next_tail)


def split_attack_string(attack_string: str) -> Tuple[str, str]:
    tail, head = attack_string.split(".")
    return tail, head


def is_all_a(s: str) -> bool:
    return len(s) > 0 and set(s) == {"A"}


def min_fork(u: float, eta: int, p_boost: float = P_BOOST) -> int:
    if u <= 0:
        return 10**9
    need = math.ceil(((eta * (1 - 2 * u)) - p_boost) / u)
    return max(need, 1)


def ex_ante(u: float, a1: int, h: int, a2: int, remain: int) -> List[Tuple[int, int]]:
    """
    Approximation of Appendix Algorithm 1.
    Returns (x, n): sacrifice x from the A-run after H-cluster and n private A-slots needed.
    """
    candidates: List[int] = []
    if a1 <= remain:
        candidates.append(0)
        x = 1
        while (a1 + h + x) <= remain and x <= a2:
            candidates.append(x)
            x += 1

    out: List[Tuple[int, int]] = []
    for x in candidates:
        n = min_fork(u, h + x)
        if n <= a1:
            out.append((x, n))
    return out


def fork_action(plan: str, a1: int, n: int) -> Sequence[str] | None:
    """
    Approximation of Appendix Algorithm 3.
    plan: C/N desired statuses over first A-cluster length a1.
    Returns action words over a1 slots (Propose/Miss/Hide), or None if infeasible.
    """
    cand_idx = [i for i in range(a1 - n + 1) if plan[i] == "C"]
    if not cand_idx:
        return None
    pivot = max(cand_idx)
    act: List[str] = []
    for i in range(a1):
        if i < pivot:
            act.append("Propose" if plan[i] == "C" else "Miss")
        else:
            act.append("Hide" if plan[i] == "C" else "Miss")
    return act


def sacrifice_from_status(status: str, tail: str) -> int:
    return sum(1 for s, t in zip(status, tail) if t == "A" and s == "N")


def selfish_options(tail: str) -> List[Option]:
    """All propose/miss combos on adversarial tail slots."""
    a_pos = [i for i, ch in enumerate(tail) if ch == "A"]
    base = ["C"] * len(tail)
    opts: List[Option] = []

    # Honest behavior (no misses)
    status0 = "".join(base)
    opts.append(Option("honest", status0, 0))

    # Selfish-mixing variants (at least one miss)
    for mask in range(1, 1 << len(a_pos)):
        status = base[:]
        for bit, pos in enumerate(a_pos):
            if (mask >> bit) & 1:
                status[pos] = "N"
        s = "".join(status)
        opts.append(Option("selfish", s, sacrifice_from_status(s, tail)))
    return opts


def find_a_h_a_patterns(tail: str, head: str) -> List[Tuple[int, int, int, int]]:
    """Find patterns start_idx, a1, h, a2 in tail+head where pattern starts with A^a1 H^h A^a2."""
    x = tail + head
    n_tail = len(tail)
    out: List[Tuple[int, int, int, int]] = []
    i = 0
    while i < n_tail:
        if x[i] != "A":
            i += 1
            continue
        j = i
        while j < len(x) and x[j] == "A":
            j += 1
        a1 = j - i
        k = j
        while k < len(x) and x[k] == "H":
            k += 1
        h = k - j
        if h == 0:
            i = j
            continue
        l = k
        while l < len(x) and x[l] == "A":
            l += 1
        a2 = l - k
        if a2 > 0:
            out.append((i, a1, h, a2))
        i = j
    return out


def forking_options(tail: str, head: str, u: float) -> List[Option]:
    """
    From-scratch approximation of recursive forking option generation.
    Produces extra fork/regret realizations over tail statuses.
    """
    n_tail = len(tail)
    out: List[Option] = []

    for start, a1, h, a2 in find_a_h_a_patterns(tail, head):
        remain = n_tail - start
        ex = ex_ante(u, a1, h, a2, remain)
        if not ex:
            continue

        for x, n in ex:
            # The plan is desired C/N statuses on first A-cluster.
            for bits in itertools.product("CN", repeat=a1):
                plan = "".join(bits)
                act = fork_action(plan, a1, n)
                if act is None:
                    continue

                # Fork-complete status: plan + N^(h+x) + C on finishing A
                sf = ["C"] * n_tail
                for i in range(a1):
                    idx = start + i
                    if idx < n_tail:
                        sf[idx] = plan[i]
                for i in range(h + x):
                    idx = start + a1 + i
                    if idx < n_tail:
                        sf[idx] = "N"
                idx_finish = start + a1 + h + x
                if idx_finish < n_tail:
                    sf[idx_finish] = "C"
                s_fork = "".join(sf)
                out.append(Option("fork", s_fork, sacrifice_from_status(s_fork, tail)))

                # Regret status: map Propose->C, Hide/Miss->N over first A-cluster; honest H^h kept canonical
                sr = ["C"] * n_tail
                for i, word in enumerate(act):
                    idx = start + i
                    if idx < n_tail:
                        sr[idx] = "C" if word == "Propose" else "N"
                for i in range(h):
                    idx = start + a1 + i
                    if idx < n_tail:
                        sr[idx] = "C"
                s_regret = "".join(sr)
                out.append(Option("regret", s_regret, sacrifice_from_status(s_regret, tail)))

    # Deduplicate exact duplicates.
    uniq: Dict[Tuple[str, str], Option] = {}
    for opt in out:
        key = (opt.kind, opt.status)
        prev = uniq.get(key)
        if prev is None or opt.sacrifice < prev.sacrifice:
            uniq[key] = opt
    return list(uniq.values())


def enumerate_options(state: State, u: float, enable_forking: bool = True) -> List[Option]:
    tail, head = split_attack_string(state.attack_string)
    opts = selfish_options(tail)
    if enable_forking:
        opts.extend(forking_options(tail, head, u))

    # Deduplicate by (kind,status), keep lowest sacrifice.
    dedup: Dict[Tuple[str, str], Option] = {}
    for o in opts:
        key = (o.kind, o.status)
        old = dedup.get(key)
        if old is None or o.sacrifice < old.sacrifice:
            dedup[key] = o
    return list(dedup.values())


def reward_from_sampled_epoch(epoch: str, option: Option) -> float:
    return float(epoch.count("A") - option.sacrifice)


class LookaheadValue:
    """Finite-horizon Monte Carlo lookahead value estimator V_d(state)."""

    def __init__(self, cfg: Config, u: float, enable_forking: bool, rng_seed: int):
        self.cfg = cfg
        self.u = u
        self.enable_forking = enable_forking
        self.rng_seed = rng_seed
        self._cache: Dict[Tuple[State, int], float] = {}

    def value(self, state: State, depth: int) -> float:
        if depth <= 0:
            return SLOTS_PER_EPOCH * self.u
        key = (state, depth)
        if key in self._cache:
            return self._cache[key]

        # Deterministic local RNG from (state, depth, base_seed) for reproducible memoized estimates.
        h = hash((state.attack_string, state.next_tail, depth, self.rng_seed)) & 0xFFFFFFFF
        rng = np.random.default_rng(h)

        opts = enumerate_options(state, self.u, self.enable_forking)
        vals = []
        for _ in range(self.cfg.lookahead_rollouts):
            best = -1e18
            for opt in opts:
                e2 = sample_epoch_string(self.u, rng)
                nxt = transition_state(state, e2, self.cfg)
                v = reward_from_sampled_epoch(e2, opt)
                if depth > 1:
                    v += self.value(nxt, depth - 1)
                if v > best:
                    best = v
            vals.append(best)

        est = float(np.mean(vals))
        self._cache[key] = est
        return est


def choose_option_and_next(
    state: State,
    u: float,
    cfg: Config,
    rng: np.random.Generator,
    val_fn: LookaheadValue,
    enable_forking: bool,
) -> Tuple[Option, State, float, str]:
    """Sample one candidate epoch per option, then pick option maximizing immediate+lookahead."""
    opts = enumerate_options(state, u, enable_forking)
    best: Tuple[float, Option, State, float] | None = None

    # for opt in opts:
    #     e2 = sample_epoch_string(u, rng)
    #     nxt = transition_state(state, e2, cfg)
    #     immediate = reward_from_sampled_epoch(e2, opt)
    #     total = immediate + val_fn.value(nxt, cfg.lookahead_depth - 1)
    #     if best is None or total > best[0]:
    #         best = (total, opt, nxt, immediate)

    # assert best is not None
    # _, opt, nxt, immediate = best
    # return opt, nxt, immediate, opt.kind
    best_expected_total = -1e18
    best_opt = opts[0]
    
    for opt in opts:
        # Estimate statistical average for this strategy
        avg_val = 0
        for _ in range(cfg.lookahead_rollouts):
            e_sim = sample_epoch_string(u, rng)
            nxt_sim = transition_state(state, e_sim, cfg)
            avg_val += reward_from_sampled_epoch(e_sim, opt) + val_fn.value(nxt_sim, cfg.lookahead_depth - 1)
        avg_val /= cfg.lookahead_rollouts
        
        if avg_val > best_expected_total:
            best_expected_total = avg_val
            best_opt = opt
    
    # Apply the committed strategy to the ACTUAL future outcome
    actual_e2 = sample_epoch_string(u, rng)
    nxt = transition_state(state, actual_e2, cfg)
    return best_opt, nxt, reward_from_sampled_epoch(actual_e2, best_opt), best_opt.kind


def simulate_policy(
    u: float,
    cfg: Config,
    enable_forking: bool,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    val_fn = LookaheadValue(cfg, u, enable_forking, rng_seed=seed + 17)

    # Initialize with two random epochs.
    e_curr = sample_epoch_string(u, rng)
    e_next = sample_epoch_string(u, rng)
    state = build_state(e_curr, e_next, cfg)

    totals = defaultdict(float)
    action_counts = defaultdict(int)
    n_eval = 0

    for t in range(cfg.epochs_per_u):
        opt, nxt, immediate, kind = choose_option_and_next(
            state=state,
            u=u,
            cfg=cfg,
            rng=rng,
            val_fn=val_fn,
            enable_forking=enable_forking,
        )

        # Throughput accounting: adversarial misses + forked honest blocks from status in tail.
        tail, _ = split_attack_string(state.attack_string)
        missed_adv = sum(1 for s, x in zip(opt.status, tail) if x == "A" and s == "N")
        forked_honest = sum(1 for s, x in zip(opt.status, tail) if x == "H" and s == "N")

        state = nxt

        if t >= cfg.burn_in:
            n_eval += 1
            totals["slots"] += immediate
            totals["missed_adv"] += missed_adv
            totals["forked_honest"] += forked_honest
            action_counts[kind] += 1

    out = {
        "expected_slots": totals["slots"] / max(1, n_eval),
        "missed_adv": totals["missed_adv"] / max(1, n_eval),
        "forked_honest": totals["forked_honest"] / max(1, n_eval),
        "throughput": 1.0 - (totals["missed_adv"] + totals["forked_honest"]) / (max(1, n_eval) * SLOTS_PER_EPOCH),
    }
    total_actions = max(1, sum(action_counts.values()))
    for k in ["honest", "selfish", "fork", "regret"]:
        out[f"p_{k}"] = action_counts[k] / total_actions
    return out


def target_slot_probability(
    u: float,
    cfg: Config,
    enable_forking: bool,
    n_trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)

    # Random state from two random epochs per trial.
    hits = 0
    for _ in range(n_trials):
        e_curr = sample_epoch_string(u, rng)
        e_next = sample_epoch_string(u, rng)
        state = build_state(e_curr, e_next, cfg)
        opts = enumerate_options(state, u, enable_forking)


        chosen_opt = opts[0] # Honest behavior
            
        # The actual randomness is computed AFTER the choice is made.
        actual_e2 = sample_epoch_string(u, rng)
        if actual_e2[cfg.target_slot] == "A":
            hits += 1

        # One-shot objective: maximize target-slot indicator in epoch+2.
        # best_hit = 0
        # for _opt in opts:
        #     e2 = sample_epoch_string(u, rng)
        #     h = 1 if e2[cfg.target_slot] == "A" else 0
        #     if h > best_hit:
        #         best_hit = h
        #         if best_hit == 1:
        #             break
        # hits += best_hit
    return hits / max(1, n_trials)


def run_all(cfg: Config, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Stake grid used for figures.
    u_grid = np.round(np.arange(0.05, 0.451, 0.05), 3)

    stats_sm = []
    stats_fork = []
    for idx, u in enumerate(u_grid):
        stats_sm.append(simulate_policy(u, cfg, enable_forking=False, seed=1000 + idx))
        stats_fork.append(simulate_policy(u, cfg, enable_forking=True, seed=2000 + idx))

    # -------- Figure 7 --------
    expected_sm = np.array([d["expected_slots"] for d in stats_sm])
    expected_fk = np.array([d["expected_slots"] for d in stats_fork])
    forked_h = np.array([d["forked_honest"] for d in stats_fork])
    honest = SLOTS_PER_EPOCH * u_grid
    all_adv = expected_fk + forked_h

    plt.figure(figsize=(8, 5))
    plt.plot(u_grid, expected_fk, label="Selfish mixing + forking (this implementation)")
    plt.plot(u_grid, all_adv, label="All adversarial slots")
    plt.plot(u_grid, forked_h, label="Forked honest blocks")
    plt.plot(u_grid, honest, label="Honest")
    plt.plot(u_grid, expected_sm, label="Selfish mixing only")
    plt.xlabel("Stakes (U)")
    plt.ylabel("Expected number of slots")
    plt.title("Figure 7-style: RANDao Manipulation Efficacy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure7_style.png"), dpi=200)
    plt.close()

    # -------- Figure 8 --------
    p_fork = np.array([d["p_fork"] for d in stats_fork]) * 100
    p_regret = np.array([d["p_regret"] for d in stats_fork]) * 100
    p_selfish = np.array([d["p_selfish"] for d in stats_fork]) * 100
    p_honest = np.array([d["p_honest"] for d in stats_fork]) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(u_grid, p_fork, label="Forking")
    plt.plot(u_grid, p_regret, label="Regret")
    plt.plot(u_grid, p_selfish, label="Selfish mixing")
    plt.plot(u_grid, p_honest, label="Honest")
    plt.xlabel("Stakes (U)")
    plt.ylabel("Probability (%)")
    plt.title("Figure 8-style: Strategic Behavior in Stationary Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure8_style.png"), dpi=200)
    plt.close()

    # -------- Figure 11 --------
    p_honest_target = u_grid.copy()
    p_manip_target = np.array(
        [target_slot_probability(u, cfg, True, n_trials=1800, seed=3000 + i) for i, u in enumerate(u_grid)]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(u_grid, p_manip_target, label="Manipulated")
    plt.plot(u_grid, p_honest_target, label="Honest")
    plt.xlabel("Stakes (U)")
    plt.ylabel("Probability of getting target slot")
    plt.title("Figure 11-style: Target Slot Utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure11_style.png"), dpi=200)
    plt.close()

    # -------- Figure 12 --------
    throughput = np.array([d["throughput"] for d in stats_fork])
    m_adv = np.array([d["missed_adv"] for d in stats_fork]) / SLOTS_PER_EPOCH
    f_h = np.array([d["forked_honest"] for d in stats_fork]) / SLOTS_PER_EPOCH

    plt.figure(figsize=(8, 5))
    plt.plot(u_grid, throughput, label="Chain throughput")
    plt.plot(u_grid, f_h, label="Forked honest blocks")
    plt.plot(u_grid, m_adv, label="Missed adversarial slots")
    plt.xlabel("Stakes (U)")
    plt.ylabel("Fraction")
    plt.title("Figure 12-style: Throughput Degradation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure12_style.png"), dpi=200)
    plt.close()

    # -------- Table 2-style (PoS columns from this implementation) --------
    table_stakes = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    rows = ["stake,selfish_mixing_pct,selfish_mixing_plus_forking_pct"]
    for i, u in enumerate(table_stakes):
        sm = simulate_policy(u, cfg, enable_forking=False, seed=4000 + i)["expected_slots"] / SLOTS_PER_EPOCH * 100
        fk = simulate_policy(u, cfg, enable_forking=True, seed=5000 + i)["expected_slots"] / SLOTS_PER_EPOCH * 100
        rows.append(f"{u:.2f},{sm:.4f},{fk:.4f}")

    with open(os.path.join(out_dir, "table2_style_pos.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    # Save summary json-like text.
    summary_lines = [
        "Generated outputs:",
        "- figure7_style.png",
        "- figure8_style.png",
        "- figure11_style.png",
        "- figure12_style.png",
        "- table2_style_pos.csv",
        "",
        f"Config: {cfg}",
    ]
    with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="From-scratch RANDao MDP attack simulator")
    p.add_argument("--out-dir", default="outputs", help="Directory for generated plots/tables")
    p.add_argument("--tail-len", type=int, default=6)
    p.add_argument("--head-len", type=int, default=2)
    p.add_argument("--lookahead-depth", type=int, default=2)
    p.add_argument("--lookahead-rollouts", type=int, default=8)
    p.add_argument("--epochs-per-u", type=int, default=1200)
    p.add_argument("--burn-in", type=int, default=200)
    p.add_argument("--target-slot", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        tail_len=args.tail_len,
        head_len=args.head_len,
        lookahead_depth=args.lookahead_depth,
        lookahead_rollouts=args.lookahead_rollouts,
        epochs_per_u=args.epochs_per_u,
        burn_in=args.burn_in,
        target_slot=args.target_slot,
    )
    run_all(cfg, args.out_dir)
