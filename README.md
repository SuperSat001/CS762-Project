# Robust On-Chain Randomness: Mitigating RANDAO Forking and Selfish Mixing Attacks via Verifiable Delay Functions

**CS762: Advanced Blockchain Technology — Course Project**

Arihant Vashista (22B0958) · Satyankar Chandra (22B0967) · Antriksh Punia (22B1031)

---

## Abstract

Ethereum's Proof-of-Stake consensus relies on RANDAO — an XOR-accumulation of BLS signature reveals — to assign block-proposal slots. An adversary controlling a fraction α of stake can strategically withhold or delay their reveals at the tail of an epoch, generating up to 2^t candidate RANDAO outputs and selecting the one most favorable to future slot allocation. This project empirically proves that selfish mixing and forking attacks yield significant above-proportional returns, and evaluates Verifiable Delay Functions (VDFs) as a cryptographic countermeasure that removes the adversary's ability to enumerate outcomes in parallel.

Two independent simulation frameworks are built and compared:
1. An **MDP-based Monte Carlo simulator** (Checkpoint 1) that directly models the adversary's decision problem as a finite-horizon Markov Decision Process with Expectimax rollouts.
2. A **Discrete Event Simulator** (Checkpoint 2) with a modular, event-driven architecture that models realistic slot/epoch timing, LMD-GHOST fork choice, and adversarial proposer strategies.
3. A **SageMath benchmarking suite** (Checkpoint 3) that characterizes the concrete performance of a class-group (Wesolowski) VDF for parameter selection.

---

## Background

### RANDAO

Ethereum PoS operates in discrete **slots** (12 s each); 32 consecutive slots form an **epoch**. At each slot, one validator proposes a block. The RANDAO accumulator for epoch e is updated as:

```
R_e = R_{e-1} ⊕ ⊕_{i=0}^{31} h(r_i^e)
```

where r_i^e is the proposer's BLS signature reveal in slot i, and a missed block contributes h(r_i^e) := 0. R_e determines the proposer schedule for epoch e+2.

### Attacks

**Selfish Mixing:** An adversary controlling t consecutive tail slots generates 2^t candidate values for R_e by choosing to publish or withhold each of its reveals. It selects the R_e* that maximizes its expected future slot share, at a cost of s missed blocks.

**Forking Attack (AHA·):** The adversary privately builds a competing block at slot i, waits for an honest block at slot i+1, then extends its private chain in the next epoch. If the fork succeeds via LMD-GHOST (aided by a proposer-boost), the honest block is reorged out, removing its RANDAO contribution and shifting the outcome space further in the adversary's favor.

**Combined AHA·:** Forking and selfish mixing are applied jointly at epoch boundaries, increasing the number of reachable RANDAO outcomes from 3 to 5 per epoch boundary.

### VDF Mitigation

A VDF V = (Setup, Eval, Verify) transforms the RANDAO mix R_e into a final seed:

```
y_e = VDF(f(R_e))    seed = h(y_e)
```

Because Eval requires T *sequential* steps (no parallelism), the adversary cannot compute multiple candidate outputs before the deadline. The adversary must commit to publish/withhold decisions without knowing future randomness, collapsing the selfish mixing advantage.

The class-group (Wesolowski) VDF is chosen for its **trustless setup**: the discriminant Δ is derived from a public seed with no ceremony, and the group order is unknown by construction, so no party can shortcut the sequential computation.

---

## Repository Structure

```
CS762-Project/
├── ckpt-1/                         # Checkpoint 1: MDP simulation
│   ├── randao_mdp_from_scratch.py  # World A — attack without VDF
│   ├── vdf_simulation.py           # World B — attack with VDF mitigation
│   ├── 2025-037.pdf                # Reference: Nagy & Seres (2025) "Forking the RANDAO"
│   ├── 2025-037 2.pdf              # Reference: alternate copy / annotation
│   └── 2024-198.pdf                # Reference: related RANDAO/VDF work
│
├── ckpt-2/                         # Checkpoint 2: Discrete Event Simulator
│   ├── sim/                        # Core DES modules
│   │   ├── simulator.py
│   │   ├── block.py
│   │   ├── events.py
│   │   ├── fork_choice.py
│   │   ├── metrics.py
│   │   ├── randao.py
│   │   ├── vdf.py
│   │   └── strategies.py
│   ├── analysis/                   # Experiment orchestration & visualization
│   │   ├── common.py
│   │   ├── run_experiments.py
│   │   ├── plot_results.py
│   │   ├── visualize_blockchain.py
│   │   └── debug_trace.py
│   ├── tests/                      # pytest test suite (8 modules)
│   ├── experiments.py              # 8 hand-crafted experiment scenarios
│   ├── conftest.py
│   ├── pytest.ini
│   └── results*/                   # Archived CSV data and PNG plots
│
├── ckpt-3/                         # Checkpoint 3: VDF benchmarking
│   ├── vdf_benchmark.sage          # SageMath class-group VDF benchmark
│   └── README_vdf_benchmark.md     # Usage instructions
│
└── presentation.pdf                # Final project presentation (44 slides)
```

---

## Checkpoint 1: MDP Simulation

**Goal:** Model the adversary's decision problem as a Markov Decision Process and solve it approximately using finite-horizon Monte Carlo Expectimax, without and with VDF.

### Design

**State:** An extended attack string `tail.head` — the last `tail_len` (default 6) characters of the current epoch's proposer assignments (A = adversarial, H = honest) concatenated with the first `head_len` (default 2) characters of the next epoch, plus the next tail (used for state transitions).

**Actions (Options):** At each epoch, the adversary chooses an `Option` — a C/N status string over its tail slots and a `kind`:
- `honest` — publish all blocks, no sacrifice
- `selfish` — miss ≥1 adversarial tail slot to influence R_e
- `fork` — privately build a block, reorg an honest block out of RANDAO
- `regret` — fork attempted but abandoned (honest behavior realized)

**Reward:** `A-count(epoch+2) - sacrifice` — future slots won minus blocks missed.

**Value function:** `LookaheadValue` — memoized Monte Carlo Expectimax over `lookahead_depth` horizons, averaging `lookahead_rollouts` sampled futures per state.

### Files

| File | Description |
|------|-------------|
| [ckpt-1/randao_mdp_from_scratch.py](ckpt-1/randao_mdp_from_scratch.py) | **World A (no VDF).** Adversary enumerates candidate RANDAO outcomes and picks the best. `choose_option_and_next` samples one future per option and selects the maximizing option — modeling an adversary with outcome visibility. Generates Figures 7, 8, 11, 12 and Table 2 from the paper. |
| [ckpt-1/vdf_simulation.py](ckpt-1/vdf_simulation.py) | **World B (with VDF).** Same MDP structure, but `choose_option_and_next` uses statistical averaging across multiple rollouts to select a strategy *before* seeing the realized randomness — modeling the commitment constraint imposed by VDF. `target_slot_probability` reverts to honest behavior. Shows the adversarial advantage collapses to near stake-proportional. |
| [ckpt-1/2025-037.pdf](ckpt-1/2025-037.pdf) | Primary reference: Nagy & Seres (2025) "Forking the RANDAO" |
| [ckpt-1/2024-198.pdf](ckpt-1/2024-198.pdf) | Supporting reference paper |

### Running

```bash
cd ckpt-1

# World A: RANDAO attack without VDF
python randao_mdp_from_scratch.py --out-dir outputs_no_vdf

# World B: RANDAO attack with VDF (adversary cannot enumerate outcomes)
python vdf_simulation.py --out-dir outputs_vdf
```

**CLI options (both scripts):**

| Flag | Default | Description |
|------|---------|-------------|
| `--out-dir` | `outputs` | Output directory for plots/tables |
| `--tail-len` | `6` | Number of tail slots in state window |
| `--head-len` | `2` | Number of head slots from next epoch |
| `--lookahead-depth` | `2` | Horizon depth for value estimation |
| `--lookahead-rollouts` | `8` | Monte Carlo samples per lookahead step |
| `--epochs-per-u` | `1200` | Simulation length per stake value |
| `--burn-in` | `200` | Epochs discarded for stationarity |
| `--target-slot` | `8` | Slot index used for target-slot utility |

**Outputs (per run):**
- `figure7_style.png` — Expected adversarial slots vs. stake (selfish mixing only vs. selfish + forking vs. honest baseline)
- `figure8_style.png` — Probability of each strategic action (honest / selfish / fork / regret) as stake grows
- `figure11_style.png` — Target-slot capture probability (manipulated vs. honest)
- `figure12_style.png` — Blockchain throughput degradation (chain throughput / forked honest / missed adversarial)
- `table2_style_pos.csv` — Effective slot share table across stake values (PoS columns)
- `run_summary.txt` — Config echo and output manifest

**Dependencies:** Python 3.9+, `numpy`, `matplotlib`

---

## Checkpoint 2: Discrete Event Simulator (DES)

**Goal:** Independently validate the MDP results using a realistic, event-driven simulation with slot-level timing, LMD-GHOST fork choice, and pluggable adversarial strategies. Sweep parameters systematically and generate publication-quality plots.

### Architecture

Events are processed in order `SLOT_START → BLOCK_PROPOSE → ATTEST → FORK_CHOICE_UPDATE`, with `EPOCH_END` and `VDF_COMPLETE` inserted when needed. 100 validators, 32 slots/epoch, 15 epochs/run.

### sim/ — Core Modules

| File | Description |
|------|-------------|
| [ckpt-2/sim/simulator.py](ckpt-2/sim/simulator.py) | Top-level event-driven loop. `Simulator` class orchestrates the DES. `Validator` record holds id, stake, strategy, and secret. `StateView` exposes only information legally visible to a strategy at decision time (no future randomness). |
| [ckpt-2/sim/block.py](ckpt-2/sim/block.py) | `Block` dataclass (id, parent, slot, proposer, randao_reveal, weight, children) and `BlockTree` (immutable storage with genesis). `compute_randao_reveal(secret, epoch)` → 256-bit integer. |
| [ckpt-2/sim/events.py](ckpt-2/sim/events.py) | `EventType` enum, `Event` dataclass (slot, priority, seq, type, payload), `EventQueue` min-heap with deterministic tie-breaking by sequence number. |
| [ckpt-2/sim/fork_choice.py](ckpt-2/sim/fork_choice.py) | Incremental LMD-GHOST. `attest()` adjusts subtree weights in O(chain_depth); `head()` walks from genesis to the heaviest leaf. Reorg detection included. |
| [ckpt-2/sim/metrics.py](ckpt-2/sim/metrics.py) | Tracks adversarial_blocks, total_blocks, forks, reorgs, missed_slots, and proposer distribution vs. stake weight per run. |
| [ckpt-2/sim/randao.py](ckpt-2/sim/randao.py) | `RandaoState` tracks XOR-accumulated mixes across epochs. `compute_proposer_schedule(mix, slots, validators)` produces the deterministic slot assignment for epoch e+2. |
| [ckpt-2/sim/vdf.py](ckpt-2/sim/vdf.py) | `VDF.eval(input_value, current_slot)` returns `(output, ready_slot)`. Enforces temporal causality: the proposer schedule for epoch e+2 cannot be computed until `ready_slot` is reached. |
| [ckpt-2/sim/strategies.py](ckpt-2/sim/strategies.py) | `HonestStrategy` — always proposes on the head. `SelfishMixingStrategy` — skips tail slots to manipulate the RANDAO mix. `ForkingStrategy` — withholds blocks and releases a competing chain for reorg. |

### analysis/ — Experiment Framework

| File | Description |
|------|-------------|
| [ckpt-2/analysis/common.py](ckpt-2/analysis/common.py) | `ExperimentSpec` dataclass. `run_instrumented_simulation(spec)` executes one run and returns a metrics dict. `aggregated_results(df)` groups by strategy/alpha/vdf_delay. Constants: `NUM_VALIDATORS=100`, `EXPERIMENT_EPOCHS=15`, `ALPHAS=(0.1, 0.2, 0.3)`, `SEEDS=(10, 20, 30)`, `VDF_DELAYS=(0, 32)`. |
| [ckpt-2/analysis/run_experiments.py](ckpt-2/analysis/run_experiments.py) | Runs the full Cartesian product: 3 strategies × 3 α × 3 seeds × 2 VDF delays = **54 simulations**. Saves `results/data/simulation_results.csv`. |
| [ckpt-2/analysis/plot_results.py](ckpt-2/analysis/plot_results.py) | Generates: adversarial reward vs. α, fork rate vs. α, VDF effect on reward, time-series evolution. Saves PNGs to `results/plots/`. |
| [ckpt-2/analysis/visualize_blockchain.py](ckpt-2/analysis/visualize_blockchain.py) | Renders the block tree for a single run — canonical chain in black, private/forked branches in grey. Saves to `results/graphs/`. |
| [ckpt-2/analysis/debug_trace.py](ckpt-2/analysis/debug_trace.py) | Low-level event-by-event trace for debugging individual runs. |
| [ckpt-2/experiments.py](ckpt-2/experiments.py) | 8 manually constructed experiment scenarios (spot-checks for edge cases). |

### tests/ — Test Suite (8 modules, ~920 lines)

`test_block.py`, `test_events.py`, `test_fork_choice.py`, `test_integration.py`, `test_randao.py`, `test_strategies.py`, `test_vdf.py`

### Running

```bash
cd ckpt-2

# Run all unit and integration tests
python -m pytest tests/ -v

# Run 54-simulation sweep (writes results/data/simulation_results.csv)
python -m analysis.run_experiments

# Generate plots from saved CSV
python -m analysis.plot_results

# Visualize a block tree
python -m analysis.visualize_blockchain

# Run hand-crafted scenario checks
python experiments.py
```

**Dependencies:** Python 3.9+, `numpy`, `matplotlib`, `seaborn`, `pytest`

---

## Checkpoint 3: VDF Benchmarking (SageMath)

**Goal:** Characterize the concrete performance of a Wesolowski class-group VDF to select deployment parameters (discriminant size |Δ|, iteration count T) that enforce a ≥6–7 minute computation window — safely exceeding one slot (12 s) by 4× hardware margin.

### Why Class Groups?

| Construction | Setup | Notes |
|---|---|---|
| RSA-based (Wesolowski/Pietrzak) | Trusted MPC ceremony | Broken silently if any party retains p, q |
| **Class-group (Wesolowski)** | **Trustless** | Δ is a public integer; group order unknown by design |
| Isogeny-based (De Feo et al.) | Trustless | Research stage; large proof sizes |

Class-group VDFs are deployed in production (Chia Network), have compact Wesolowski proofs (~5 KB, ~1 ms verification), and rely on standard hardness assumptions.

### Design Parameters

- **Discriminant |Δ| = 2048 bits.** −Δ prime, −Δ ≡ 3 (mod 4). Derived from a public seed via rejection sampling — auditable, no ceremony.
- **Input encoding:** Hash-to-class-group via the prime-form method (Tonelli–Shanks + reduction). Squaring lands in the squares subgroup to avoid 2-torsion.
- **Evaluation:** T sequential NUDUPL calls (specialized squaring, ~2× faster than general NUCOMP composition).
- **Proof:** Wesolowski (1 group element, ~1.8 s generation, ~1 ms verification).
- **Selected T = 2^29** → ~7 min on reference CPU. With a 4× hardware safety margin, an attacker still requires >100 s per candidate — far exceeding any single-slot decision window.

### File

| File | Description |
|------|-------------|
| [ckpt-3/vdf_benchmark.sage](ckpt-3/vdf_benchmark.sage) | Full benchmark suite. Measures: (1) discriminant generation, (2) hash-to-class-group, (3) NUDUPL squaring throughput, (4) Wesolowski proof generation, (5) verification. Runs at 256-bit and 1024-bit discriminant sizes; extrapolates to 2048-bit. Saves results to `vdf_benchmark_results.json`. |
| [ckpt-3/README_vdf_benchmark.md](ckpt-3/README_vdf_benchmark.md) | Usage instructions and expected timing table. |

### Running

```bash
cd ckpt-3

# Correctness check only (< 1 min)
sage vdf_benchmark.sage --test

# Quick benchmark at 256-bit (~ 2 min)
sage vdf_benchmark.sage

# Full suite with 1024-bit and extrapolation to 2048-bit (10–30 min)
sage vdf_benchmark.sage --full
```

**Expected timings (reference CPU):**

| Setting | Squarings/sec | T = 2^29 estimate |
|---------|--------------|-------------------|
| 256-bit Δ | ~200k–500k | ~1–3 min |
| 1024-bit Δ | ~20k–50k | ~3–7 min |
| 2048-bit Δ (extrapolated) | ~1.2 × 10^6 (NUDUPL) | ~7 min |

Wesolowski proof generation: ~1.8 s · Verification: ~1.1 ms

**Dependencies:** [SageMath](https://www.sagemath.org/) 9.x+

---

## Key Results

| Claim | Evidence |
|-------|----------|
| RANDAO is biasable via selfish mixing | At α=0.45, adversary captures ~54% of slots (vs. expected 45%) — selfish mixing only |
| Combined selfish mixing + forking amplifies bias | At α=0.45, slot share rises to ~57% — a 12 percentage-point gain |
| VDF eliminates randomness manipulation advantage | Adversarial slot share drops to near stake-proportional in both MDP (World B) and DES (VDF delay=32) |
| VDF does not suppress structural forking | Fork rate vs. stake is identical with and without VDF — forking attacks chain topology, not future randomness |
| No throughput side-effect | Chain throughput is unaffected by VDF in both simulation frameworks |
| Class-group VDF is feasible at T=2^29 | ~7 min compute, ~1.8 s proof, ~1 ms verification; attacker needs >100 s per candidate with 4× hardware margin |

---

## References

- Nagy, Á., Seres, I. A. (2025). *Forking the RANDAO.* (`ckpt-1/2025-037.pdf`)
- Boneh, D., Bonneau, J., Bünz, B., Fisch, B. (2018). *Verifiable Delay Functions.* CRYPTO 2018.
- Wesolowski, B. (2019). *Efficient Verifiable Delay Functions.* EUROCRYPT 2019.
- Cohen, H. (1993). *A Course in Computational Algebraic Number Theory.* Springer GTM 138.
