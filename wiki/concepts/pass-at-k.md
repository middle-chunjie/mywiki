---
type: concept
title: Pass@k
slug: pass-at-k
date: 2026-04-20
updated: 2026-04-20
aliases: [Pass@k, 通过率@k]
tags: [evaluation, benchmark, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pass@k** (通过率@k) — a code-generation metric measuring whether at least one of `k` sampled outputs is correct according to the benchmark's oracle.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] uses pass@1 as the main conventional metric for validating RTC on HumanEval and ARCADE.
- RTC is not identical to pass@k because RTC also depends on the forward description generation step.
- The paper reports strong rank and linear correlation between `RTC_pass` and pass@1 across multiple models.
- Sensitivity analysis is discussed in parallel with pass@k-style dependence on sampling and temperature.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
