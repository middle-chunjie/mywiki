---
type: concept
title: Covariate Shift
slug: covariate-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [协变量偏移]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Covariate Shift** (协变量偏移) — a shift setting where the input distribution changes across domains while the conditional output distribution is still relatively compatible.

## Key Points

- In the paper's notation, covariate shift means `p_s(x) != p_t(x)` while `p_s(y | x) ~= p_t(y | x)`.
- Quasar-S is the clearest covariate-shift case identified by the proposed test.
- The results suggest that zero-shot target-side augmentation can still help substantially under covariate shift because the reader's answer behavior is not entirely misaligned.
- Retriever failures under covariate shift are often tied to surface changes such as context length and entity-type distribution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dua-2023-adapt]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dua-2023-adapt]].
