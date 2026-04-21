---
type: concept
title: Synthetic Verification
slug: synthetic-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic verifier, synthetic verification, 合成验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Verification** (合成验证) — automatically estimating the correctness or quality of a candidate solution using generated tests, learned reward signals, or other synthetic checks instead of relying only on benchmark-provided oracle tests.

## Key Points

- [[ficek-2025-scoring-2502-13820]] treats both generated test suites and code reward models as synthetic verifiers for code and reasoning tasks.
- The paper argues existing benchmarks rarely test whether a synthetic verifier can recover the true ranking among multiple candidate solutions for the same problem.
- Its benchmark-construction recipe creates ranked candidate sets with oracle pass-rate labels so synthetic verifiers can be evaluated against ground-truth scoring behavior.
- Experiments show reasoning models substantially improve synthetic verification quality through stronger test-case generation, and additional generated tests further improve accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ficek-2025-scoring-2502-13820]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ficek-2025-scoring-2502-13820]].
