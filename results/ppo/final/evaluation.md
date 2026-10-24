# PPO final evaluation (test seeds)

Headline = median run (**run 1**); leaderboard = best run by validation (**run 4**).

## Criteria per run

| Run | Default mean | Held-out mean | P1 | P2 pass/strong/stretch | P3 | H1 | H2 a/b/c | B1 | Gap closed (vs Market+1%) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 38,624 | 44,554 | ✓ | ✓/✓/✓ | ✓ | ✓ | ✓/✗/✗ | ✓ | 1.59 [1.50, 1.69] |
| 2 | 37,709 | 47,722 | ✓ | ✓/✓/✓ | ✓ | ✓ | ✓/✓/✓ | ✓ | 1.48 [1.39, 1.58] |
| 3 | 37,807 | 45,596 | ✓ | ✓/✓/✓ | ✓ | ✓ | ✓/✓/✗ | ✓ | 1.49 [1.41, 1.59] |
| 4 | 40,309 | 51,762 | ✓ | ✓/✓/✓ | ✓ | ✓ | ✓/✓/✓ | ✓ | 1.80 [1.70, 1.91] |
| 5 | 38,722 | 55,199 | ✓ | ✓/✓/✓ | ✓ | ✓ | ✓/✓/✓ | ✓ | 1.60 [1.52, 1.70] |

R1: 5/5 runs meet P1 and P3 (pass); 5/5 meet P2 (pass). Run means: min 37,709, median 38,624, max 40,309, std 1,043.

## Headline run 1 vs every agent — default test seeds

| Agent | Agent mean | PPO − agent (95% CI) | PPO wins |
|---|---|---|---|
| Random | 2,663 | +35,961 [+34,769, +37,153] | 100% |
| Minimum wage | 12,223 | +26,402 [+25,882, +26,922] | 100% |
| Maximum wage | -13,995 | +52,619 [+51,940, +53,297] | 100% |
| Poacher | -10,913 | +49,537 [+49,004, +50,071] | 100% |
| Hold | 14,786 | +23,839 [+23,326, +24,351] | 100% |
| Market + 1% | 25,583 | +13,041 [+12,442, +13,640] | 100% |
| Rule-based | 33,783 | +4,841 [+4,363, +5,319] | 92% |
| Premium pricer | 17,585 | +21,039 [+20,536, +21,541] | 100% |

PPO mean 38,624 ± 517; bankrupt 0.0%.

## Headline run 1 vs every agent — heldout test seeds

| Agent | Agent mean | PPO − agent (95% CI) | PPO wins |
|---|---|---|---|
| Random | -6,690 | +51,244 [+46,002, +56,487] | 98% |
| Minimum wage | 11,493 | +33,061 [+27,226, +38,896] | 80% |
| Maximum wage | -14,197 | +58,751 [+52,594, +64,908] | 98% |
| Poacher | -6,142 | +50,696 [+45,597, +55,794] | 99% |
| Hold | 18,137 | +26,417 [+21,695, +31,139] | 76% |
| Market + 1% | 10,100 | +34,455 [+31,280, +37,629] | 99% |
| Rule-based | 43,724 | +830 [-1,146, +2,807] | 46% |
| Premium pricer | 19,049 | +25,505 [+22,226, +28,784] | 98% |

PPO mean 44,554 ± 6,033; bankrupt 12.5%.

## Scripted agents: default → held-out (test)

| Agent | Default | Held-out | Ratio | Held-out bankrupt |
|---|---|---|---|---|
| Rule-based | 33,783 | 43,724 | 1.29 | 0% |
| Hold | 14,786 | 18,137 | 1.23 | 35% |
| Market + 1% | 25,583 | 10,100 | 0.39 | 53% |
| Minimum wage | 12,223 | 11,493 | 0.94 | 11% |
| Maximum wage | -13,995 | -14,197 | n/a | 100% |
| Poacher | -10,913 | -6,142 | n/a | 87% |
| Premium pricer | 17,585 | 19,049 | 1.08 | 41% |
| Random | 2,663 | -6,690 | n/a | 76% |

## Behavior audit (run 1, default test)

- At bound, mean action (price/wage/hire): 0.0%, 0.0%, 0.0%; sampled: 0.0%, 0.0%, 0.0%
- Wage / market: p5 0.950, median 0.992, p95 1.072
- Price set: p5 $10.34, median $11.09, p95 $11.61
- Endgame: 99% of episodes hire in the last 3 steps; cost +130 ± 34 net worth
- Return vs net worth correlation 1.0000; max telescoping residual 1.33e-15
- Worst episodes (seed, net worth): [(10023, 28052), (10070, 29246), (10022, 29619)]; best: [(10104, 46691), (10116, 48107), (10010, 49438)]
- Export parity (run 1): max |torch − numpy| 3.44e-07 on 1000 observations
