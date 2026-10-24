# Deviations from the pre-registration

Times are file timestamps (IST) or the order of events in the working log.
The test seeds (10,000–10,199 and 20,000–20,199) were used exactly once, by
`python -m ppo.evaluate`, at **2026-10-25 01:06**. **Every deviation below was
decided before that moment.** Nothing was changed after seeing test results.

## Timeline
| When (2026-10-25, IST) | Event |
|---|---|
| before 00:39 | Round-2 draft pre-registration written (rules v2 + spillover 0.4), before the round-2 pilot started |
| 00:39 | Round-2 pilot (v2, spillover 0.4) ends; PPO found the spillover exploit again (finding 3) |
| 00:44 | **Final pre-registration written** (`preregistration.md`), while the v2 + spillover-0 pilot was at 614k of 1M steps |
| 00:49 | v2 + spillover-0 pilot ends |
| 00:50 | Budget (700k) derived with the pre-registered rule and appended ("Frozen budget") |
| after 00:50 | 5 training runs |
| 01:06 | One-time test evaluation |

## Deviations (draft → final pre-registration, all before any test use)
| # | What changed | From (draft) | To (final, 00:44) | Why |
|---|---|---|---|---|
| 1 | Rule base | v2 + spillover 0.4 | **v2 + spillover 0** | Pre-planned fallback: the round-2 pilot found the spillover exploit persisting under v2 (PPO $69.4k vs Rule-based $44.3k on validation; $33.5k with spillover off). |
| 2 | Gap-closed anchor | Hold | **Market + 1%** (headline); Hold still reported | Under v2, Hold loses its whole workforce by step ~10, so it is a failure reference, not a meaningful floor. |
| 3 | H2 formulation | (a) ≥ 85% of held-out benchmark; drop default → held-out ≤ benchmark's drop + 10 pp | (a) unchanged; **(b)** PPO ÷ benchmark on held-out ≥ PPO ÷ benchmark on default − 10 pp; **(c)** held-out bankruptcy ≤ 5% | The benchmark *gains* 25% on held-out with spillover off, which made the drop margin demand that PPO gain ≥ 15%, which measures nothing. (b) keeps the same intent (PPO's lead must not shrink by more than 10 pp out of distribution), and (c) adds a requirement, so H2 was not loosened. |
| 4 | Budget rule | Plateau of the default-validation curve | **Later of the default and held-out validation plateaus** | User instruction; written into the final pre-registration before the pilot finished. |

## Record-keeping notes (no effect on results)
- The round-2 **draft** pre-registration was overwritten by the final version
  rather than kept as a separate file. Its content is summarized in the
  "History" section of `preregistration.md` and in the table above. The final
  version keeps its original 00:44 timestamp; only the "Frozen budget" section
  was appended, at 00:50.
- Re-running `ppo.benchmark` under v2 overwrote the round-1 `benchmark.json`. The
  round-1 records were regenerated with `python -m ppo.benchmark --legacy` into
  `results/ppo/round1/`. They reproduce the original numbers exactly (Premium
  pricer $68,573, Rule-based $66,831; same margins and wage sweep), because the
  legacy rules are deterministic and checked by `test_golden.py`.
- After the freeze, `ppo/evaluate.py` gained a `--dry-run` mode (validation seeds
  only, for a code check on the pilot) and a fixed report title. No criterion,
  threshold, seed range, benchmark or budget changed.

## After the test evaluation
- No retraining, tuning or re-selection. The README and evaluation report state
  the results as measured, including the H2 failures of runs 1 and 3.
- Any further experiment (for example, a training profile with sub-$7 price wars)
  is **post-hoc**. It must be labelled as such and evaluated on a fresh seed
  range, never on the test ranges above.
