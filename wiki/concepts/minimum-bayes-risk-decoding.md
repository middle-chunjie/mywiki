---
type: concept
title: Minimum Bayes Risk Decoding
slug: minimum-bayes-risk-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [MBR decoding, minimum Bayes-risk decoding, 最小贝叶斯风险解码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Minimum Bayes Risk Decoding** (最小贝叶斯风险解码) — a decoding strategy that selects the output with the lowest expected task loss under a model-induced distribution or candidate set, rather than the one with highest direct probability.

## Key Points

- [[shi-2022-natural-2204-11454]] instantiates MBR over sampled programs, choosing the candidate with minimum aggregate loss against other candidates in the sample set.
- The paper uses MBR as a post-hoc selection layer on top of a frozen Codex model, so no retraining is required.
- In this setting, MBR is valuable because multiple low-probability samples can still agree semantically even when none is individually the highest-likelihood program.
- The work compares execution-based MBR against BLEU-based MBR and standard likelihood reranking, and finds the execution-aware loss more effective on execution-evaluated benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-natural-2204-11454]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-natural-2204-11454]].
