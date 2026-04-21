---
type: concept
title: Round-Trip Correctness
slug: round-trip-correctness
date: 2026-04-20
updated: 2026-04-20
aliases: [RTC, Round-Trip Correctness, 往返正确性]
tags: [evaluation, llm, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Round-Trip Correctness** (往返正确性) — an evaluation principle that measures whether a model can transform an input into an intermediate representation and then reconstruct a semantically equivalent original.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] formalizes RTC as `RTC_sim(x) = E_{y ~ M(x)} E_{x_hat ~ M^{-1}(y)} [sim(x_hat, x)]`.
- The paper uses RTC to evaluate both code synthesis and code editing without human-written labels.
- RTC is estimated with sampled forward and backward generations instead of exact expectations.
- The metric is only as good as the semantic proxy `sim(·)`, such as unit-test pass or exact match.
- The paper argues RTC complements, rather than replaces, curated benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
