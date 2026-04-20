# Experiment Results and Interpretation

## Experimental Setup

We evaluated the simulator on a fixed experiment grid designed to stay within the runtime budget while still showing clear attack behavior. Each run used 100 validators, 3 random seeds, adversarial stake fractions `alpha in {0.1, 0.2, 0.3}`, and VDF delays `{0, 32}` slots. The main batch used 15 epochs per run, which kept the total 54-run experiment suite under 5 minutes and the slowest individual run under 2 seconds.

The analysis compares three behaviors:

1. `honest_baseline`: the adversarially labeled validators behave honestly, so this is the no-attack control.
2. `adaptive_mixing`: an analysis-only RANDAO manipulation attack that decides whether to include tail-slot reveals based on the currently knowable effect on the epoch `e+2` proposer schedule.
3. `forking`: a withholding/private-chain strategy that creates forks and reorg pressure.

The goal of the experiments is to test two claims:

1. RANDAO manipulation can increase adversarial control over future proposer assignments.
2. A VDF delays access to the future proposer schedule and therefore reduces that advantage.

## Metric Definitions

The experiment pipeline records the following metrics in [simulation_results.csv](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/data/simulation_results.csv).

### `adversarial_reward`

This is the fraction of proposer slots assigned to adversarial validators in the realized future proposer schedules. It is the main attack-success metric in this project because the adaptive mixing attack targets future proposer control rather than immediate block production.

Interpretation:

- If `adversarial_reward ~= alpha`, the adversary is getting roughly the proposer share expected from stake alone.
- If `adversarial_reward > alpha`, the adversary has gained extra influence over future block proposals.
- If VDF reduces this number, the mitigation is working.

### `slot_gain`

This is defined as:

```text
slot_gain = adversarial_reward - alpha
```

It measures the adversary's excess proposer share above its stake share.

Interpretation:

- `slot_gain > 0`: the attack produced more future proposer slots than stake alone would justify.
- `slot_gain ~= 0`: no meaningful advantage.
- `slot_gain < 0`: the adversary did worse than its raw stake share.

### `fork_rate`

This is the number of fork events divided by the total number of simulated slots.

Interpretation:

- Higher `fork_rate` means the strategy creates more competing branches.
- In this simulator, it is mainly a liveness/structure stress metric rather than a randomness metric.
- It is most relevant for the forking strategy and for the blockchain tree visualizations.

### `reorg_rate`

This is the number of reorgs divided by the total number of simulated slots.

Interpretation:

- Higher `reorg_rate` means the canonical head changes discontinuously more often.
- This captures instability in the fork-choice outcome.
- It is useful for understanding how costly or disruptive private-chain behavior is.

### `missed_slots`

This counts how many slots ended without a public proposal being included.

Interpretation:

- In the adaptive mixing attack, missed slots correspond to deliberate reveal suppression.
- A nonzero value is expected for skip-based RANDAO manipulation.
- With VDF enabled, this drops to zero in our experiments because the strategy can no longer profitably condition on the future proposer schedule.

### `runtime_seconds`

Wall-clock runtime for a single simulation run.

Interpretation:

- This validates that the experiment pipeline meets the engineering constraint.
- In the final batch, the total runtime was about 74 seconds and the maximum single-run runtime was below 1.71 seconds.

## Main Findings

### 1. Adaptive RANDAO manipulation works without VDF

The clearest effect appears in the `adaptive_mixing` runs without VDF:

- At `alpha = 0.1`, adversarial reward rises to `0.143`, giving `slot_gain = +0.043`.
- At `alpha = 0.2`, adversarial reward rises to `0.304`, giving `slot_gain = +0.104`.
- At `alpha = 0.3`, adversarial reward rises to `0.419`, giving `slot_gain = +0.119`.

In all three cases, `adversarial_reward > alpha`, which means the adversary is obtaining more proposer influence than stake alone would predict.

### 2. VDF mitigation removes most of the adaptive mixing advantage

With a 32-slot VDF delay, the same adaptive mixing strategy falls back close to stake share:

- `alpha = 0.1`: reward drops from `0.143` to `0.091`
- `alpha = 0.2`: reward drops from `0.304` to `0.184`
- `alpha = 0.3`: reward drops from `0.419` to `0.279`

This is the core mitigation result. The attack only works when the adversary can evaluate how its current reveal choice affects the future proposer schedule. Once the schedule is hidden behind a VDF, that advantage disappears.

### 3. Missed slots are a clear signature of selfish mixing

The adaptive mixing strategy generates missed slots because it withholds proposals to avoid contributing unfavorable reveals:

- `11.0` missed slots on average at `alpha = 0.1`
- `29.3` missed slots on average at `alpha = 0.2`
- `37.3` missed slots on average at `alpha = 0.3`

With VDF enabled, `missed_slots = 0` across all adaptive mixing settings in the final dataset, which is consistent with the claim that the strategy loses its decision signal.

### 4. Fork rate increases with adversarial stake for private-chain behavior

For the `forking` strategy without VDF:

- `alpha = 0.1`: `fork_rate = 0.032`
- `alpha = 0.2`: `fork_rate = 0.068`
- `alpha = 0.3`: `fork_rate = 0.101`

This monotonic increase shows that larger adversarial coalitions create more visible structural instability in the block tree.

### 5. Honest baseline stays near stake share

The `honest_baseline` runs stay close to the `reward = alpha` reference line, with only small finite-sample deviations from perfect proportionality. This confirms that the excess reward in `adaptive_mixing` comes from strategic reveal selection rather than from simulator bias.

## Figure-by-Figure Narrative

### Figure 1: Adversarial Reward vs Stake Share

File: [adversarial_reward_vs_alpha.png](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/plots/adversarial_reward_vs_alpha.png)

This figure is the main result. The adaptive mixing curve without VDF sits well above the `reward = alpha` line, especially at larger `alpha`, showing a real manipulation effect. The corresponding VDF-enabled curve falls back near or below the honest baseline.

**Suggested caption:**  
*Adversarial future proposer share as a function of stake fraction. Adaptive mixing yields reward above stake when no VDF is used, but this advantage collapses when the next proposer schedule is delayed by a 32-slot VDF.*

### Figure 2: Slot Gain vs Stake Share

File: [slot_gain_vs_alpha.png](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/plots/slot_gain_vs_alpha.png)

This figure reframes the same effect as excess proposer share. Positive values mean the adversary is beating stake-proportional expectation. The adaptive mixing attack produces a clearly positive gain without VDF, while the VDF-enabled runs move back to approximately zero or slightly negative.

**Suggested caption:**  
*Excess proposer influence relative to stake share. Positive slot gain indicates that the adversary acquires more future proposer slots than its stake alone would imply. VDF removes this advantage for adaptive mixing.*

### Figure 3: Fork Rate vs Stake Share

File: [fork_rate_vs_alpha.png](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/plots/fork_rate_vs_alpha.png)

This figure highlights the structural cost of private-chain strategies. Fork rate is near zero for honest and adaptive-mixing behavior, but grows strongly with `alpha` for the forking strategy.

**Suggested caption:**  
*Fork rate as a function of adversarial stake. Private-chain behavior produces increasingly frequent forks as adversarial stake increases, while honest and reveal-skipping strategies leave the block tree largely linear.*

### Figure 4: VDF Effect on Reward

File: [vdf_effect_reward_vs_delay.png](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/plots/vdf_effect_reward_vs_delay.png)

This figure isolates the mitigation effect by using VDF delay on the x-axis. The key trend is the drop from delay `0` to delay `32` for adaptive mixing. Forking is much less sensitive because it targets chain structure more than future randomness.

**Suggested caption:**  
*Effect of VDF delay on adversarial reward. The adaptive mixing attack loses most of its benefit once the future proposer schedule is hidden behind a VDF delay, whereas forking shows only minor reward sensitivity.*

### Figure 5: Time Evolution of Honest vs Adversarial Blocks

File: [time_evolution.png](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/plots/time_evolution.png)

This figure shows how adversarial and honest blocks accumulate over time in a representative short run. The comparison is useful for understanding whether the attack changes the pace and share of observed block production.

**Suggested caption:**  
*Cumulative block production over time for a representative forking run. The no-VDF and VDF cases show similar overall growth, but the structural difference appears more clearly in the fork tree visualization.*

### Figure 6: Blockchain Tree Visualization

File: [blockchain_tree.png](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/graphs/blockchain_tree.png)

This is the most direct visualization of consensus structure. The x-axis is time in slots, the black path is the canonical chain, blue nodes are honest canonical blocks, red nodes are adversarial canonical blocks, and gray nodes are orphaned or reorged blocks. In the selected short run, the no-VDF case shows more visible branch formation and more orphaned structure than the VDF case.

**Suggested caption:**  
*Representative block-tree evolution under a forking adversary. The canonical chain is shown in black, while orphaned and reorged blocks are shown in gray. The no-VDF case exhibits more branch structure than the VDF-enabled case.*

### Figure 7: Debug Trace

File: [trace.txt](/Users/satyankar/Desktop/CS762/CS762-Project/cs762-proj/results/debug/trace.txt)

The debug trace is a correctness aid rather than a presentation figure. It records, for every slot in a representative adaptive-mixing run, the proposer type, chosen action, resulting head, fork depth, and partial RANDAO value. The skipped tail slots make the manipulation logic easy to inspect manually.

**Suggested caption:**  
*Per-slot debug trace for an adaptive mixing run, showing when adversarial proposers skip tail slots to suppress unfavorable RANDAO reveals.*

## Conclusion

The experiments support the main claim of the project: RANDAO manipulation can increase adversarial control over future proposer assignments, but a VDF significantly reduces this advantage by delaying access to the future proposer schedule. In the final dataset, adaptive mixing consistently achieved `adversarial_reward > alpha` without VDF, while the corresponding VDF-enabled runs dropped back near stake-proportional behavior. The forking experiments complement this result by showing that larger adversarial coalitions produce higher fork rates and more visible structural instability in the block tree.
