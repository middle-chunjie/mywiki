---
type: concept
title: Data Contamination
slug: data-contamination
date: 2026-04-20
updated: 2026-04-20
aliases: [benchmark contamination, data leakage, 数据污染]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Contamination** (数据污染) — unintended overlap between a model's training data and an evaluation benchmark, which can inflate measured performance without reflecting true generalization.

## Key Points

- The paper constructs SAFIM from code written after `2022-04-01` to reduce overlap with The Stack and with GPT-3.5/GPT-4 training cutoffs.
- CodeLLaMa and DeepSeekCoder still have possible date-range overlap with parts of the benchmark, so the contamination problem is mitigated rather than fully removed.
- The authors create an additional algorithmic-block test set from `2023-04` to `2024-01` to probe whether overlap materially changes results.
- Their follow-up analysis finds no major performance drop on the newer set, suggesting that contamination is not the dominant explanation for the reported rankings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2024-evaluation-2403-04814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2024-evaluation-2403-04814]].
