# PPO — pre-registration (final, round 2)

**Written 2026-10-25 00:44 IST**, while the v2 + spillover-0 pilot was still
running (at 614k of 1M steps) and before any test seed was used. Nothing below
changes after this point except values this document says are derived from
the pilot (the budget). Supersedes the round-2 draft; history at the end.

## Environment (rule base)
- **Rules v2 with spillover 0** — the new simulator default (`npc_growth=True`,
  smooth quits, `SPILLOVER_FRACTION = 0.0`; config 7f).
- Lobbying disabled for every agent (`lobbying_enabled=False`). Capex is not a
  PPO action.
- Training profile: v2 `train` (`config.NPC_PROFILES`); training episode seeds
  ≥ 100,000. Training config: `ppo/configs/medium.json`, no overrides.
- Legacy rules (`npc_growth=False`, spillover 0.4) are reproduced exactly by the
  golden test and are used only for the reported round-1 findings.

## Seeds (disjoint)
| Use | Profile | Seeds |
|---|---|---|
| Validation (checkpoint selection, budget, benchmark choice) | default | 50,000–50,199 |
| Held-out validation (budget, benchmark choice; never checkpoint selection) | heldout | 60,000–60,199 |
| Test (used once) | default | 10,000–10,199 |
| Held-out test (used once) | heldout | 20,000–20,199 |
| Training runs | — | run seeds 1, 2, 3, 4, 5 |

## Frozen benchmarks (validation, v2 + spillover 0)
| Profile | Benchmark | Mean final net worth |
|---|---|---|
| Default | Rule-based | **$34,303** (± 637) |
| Held-out | Rule-based | **$42,976** (± 4,505) |

Every scripted agent's default → held-out ratio (validation), reported because
held-out is **easier on average for the adaptive benchmark** (held-out NPCs
shrink to ~1.4 workers, freeing workers and customers) while harsher for
fixed-rule agents:

| Agent | Default | Held-out | Ratio |
|---|---|---|---|
| Rule-based | 34,303 | 42,976 | 1.25 |
| Hold | 14,712 | 18,724 | 1.27 |
| Premium pricer | 17,612 | 19,308 | 1.10 |
| Minimum wage | 12,227 | 12,278 | 1.00 |
| Market + 1% | 25,623 | 9,533 | 0.37 |
| Random | 2,624 | −6,872 | n/a (sign change) |
| Poacher | −10,449 | −6,151 | n/a (negative) |
| Maximum wage | −13,743 | −14,477 | n/a (negative) |

## Checkpoint selection rule
Within a run, the checkpoint with the **highest mean final net worth on the
default-profile validation seeds**, among checkpoints with validation
bankruptcy **≤ 1%**; ties → earlier checkpoint. Held-out validation is never
used for checkpoint selection or early stopping.

## Budget rule (derived from the v2 + spillover-0 pilot, then frozen)
For each validation curve (default and held-out) of the pilot (run seed 1,
1M steps, evaluated every 20k): the plateau step is the first step at which the
3-evaluation rolling mean reaches **98%** of that curve's maximum rolling mean.
**Budget = the later of the two plateau steps, rounded up to the next 50k.**

## Success criteria (test seeds; per run, then aggregated)
- **P1** — PPO beats Random, Minimum wage, Maximum wage, Poacher, Hold and
  Market + 1%: lower bound of the paired 95% CI of the net-worth difference
  > 0, for each. Rule-based and Premium pricer reported as extra rows.
- **P2** — vs the frozen default benchmark (Rule-based): **pass** PPO mean ≥ 90%
  of the benchmark's test mean, **strong** ≥ 95%, **stretch** paired CI lower
  bound > 0. Also reported: **gap closed = (PPO − Market + 1%) ÷ (Benchmark −
  Market + 1%)** with a paired-bootstrap 95% CI (headline), and the same with
  Hold as the anchor (reported; Hold loses its whole workforce by step ~10
  under v2, so it is a failure reference).
- **P3** — bankruptcy ≤ 1% of default test episodes.
- **H1** — as P1, on the held-out test seeds.
- **H2** — all three:
  - (a) PPO held-out mean ≥ 85% of the frozen held-out benchmark mean;
  - (b) (PPO ÷ benchmark on held-out) ≥ (PPO ÷ benchmark on default) − 10
    percentage points, means taken on the test seeds (replaces the drop margin,
    which breaks when the benchmark gains on held-out);
  - (c) PPO held-out bankruptcy ≤ 5%.
- **R1** — ≥ 4 of 5 runs meet P1 and P3. Also reported: how many runs meet P2,
  and the spread of run means (min / median / max / std).
  **Headline = median run** (by default-test mean); **leaderboard = best run by
  validation**, labelled as such.
- **B1** — judged on the **final selected policy at test**: "at bound" = within
  5% of the action range from an edge (decoded scale); **mean action is
  primary** (≤ 5% of steps per dimension), sampled actions reported. The wage
  range is not changed mid-experiment; if the final mean action sits at a
  bound, investigate before reporting.
- **B2** — report only: share of episodes hiring in the last 3 steps and its
  counterfactual cost. No hard-coded endgame rule.

## If P2 fails
Tune on validation only (learning rate, budget ≤ 2× the frozen budget, network
size, rollout length), at most 3 attempts. Every attempt reported with its
validation and test numbers. Test seeds are never used for tuning.

## If PPO finds another exploit
Report it as a finding with numbers. No further rules revision in this round.

## Reported findings carried into the write-up
1. **Round 1 (legacy, spillover 0.4):** PPO $106.9k vs Premium pricer $68.5k on
   validation (+$38.4k ± 1.5k); the same policy with spillover off: $43.2k
   (−$17.0k vs pricer). Held-out collapsed ($67.0k peak → ~$43k, 47% bankrupt)
   via "always request hires" absorbing workers released by bankrupt NPCs.
2. **Spillover sweep (legacy, retrained per setting, validation):** PPO beat the
   best scripted agent at every setting — spill 0.0: **+$7.0k ± 0.7k, wins
   every seed**; 0.1: +$9.5k ± 0.4k; 0.2: +$15.8k ± 0.7k — dominance grew with
   spillover. These ran on **legacy** rules.
3. **Round 2 (v2, spillover 0.4):** the same exploit persisted (PPO $69.4k vs
   Rule-based $44.3k; same policy with spillover off $33.5k). Cause is
   structural: demand ≈ 2.7× maximum output and price-level-independent.
4. **0.97× wage band (legacy):** paying 3% below market was free (inside the
   quit tolerance); profit peaked at 0.97×. Removed in v2 by smooth quits
   (peak now ~1.03×).

## Known limitations
- Spillover 0 removes demand coupling between firms: a rival's capacity
  shortage no longer sends its customers anywhere.
- Train-profile NPC bankruptcies are rare (~1–2% of episodes vs ~15–18% on
  held-out); the round-2 held-out curve did not collapse, so this was not
  re-tuned.
- Held-out is easier on average for adaptive agents (see ratios above).
- Single difficulty (Medium); test results are on one seed range per profile.

## History
- Round-2 draft (before the v2 pilot): v2 + spillover 0.4, H2 drop margin
  (benchmark drop + 10 pp), gap closed anchored on Hold. Superseded after the
  v2 pilot found the spillover exploit (finding 3).

## Frozen budget (added 2026-10-25, after the v2 + spillover-0 pilot, before any run or test)
Pilot (run seed 1, 1M steps, every 20k): default-validation 3-eval rolling mean
peaked at $38,759 and first reached 98% of it at 675,840 steps; held-out at
$46,902, reaching 98% at 512,000 steps. **Budget = 700,000 steps** (later
plateau, rounded up to 50k). Validation every 20,000 steps on 200 seeds.
Written into `ppo/configs/medium.json` (`total_timesteps`, `eval_freq`).
Criteria, benchmarks and budget are now frozen.
