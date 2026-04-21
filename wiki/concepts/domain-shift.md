---
type: concept
title: Domain Shift
slug: domain-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [distribution shift, 领域偏移]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Domain Shift** (领域偏移) — the mismatch between training and deployment data distributions that causes a model to degrade when transferred off the original data domain.

## Key Points

- The paper identifies domain shift as a major weakness of fully tuned PLM sentence encoders and argues that prompt-only tuning reduces overfitting to training-domain lexical cues.
- PromCSE evaluates robustness on CxC-STS, where the texts are image captions rather than the news, forum, and lexical-definition text seen in standard STS benchmarks.
- Unsupervised PromCSE improves from `67.5` to `71.2` on CxC-STS relative to SimCSE, a larger gain than its `+2.24` improvement on the standard STS average.
- The paper uses this result to argue that multi-layer soft prompts preserve general language knowledge better than full-parameter contrastive tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-improved-2203-06875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-improved-2203-06875]].
