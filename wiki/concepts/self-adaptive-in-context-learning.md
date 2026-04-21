---
type: concept
title: Self-Adaptive In-Context Learning
slug: self-adaptive-in-context-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [self-adaptive ICL, adaptive in-context learning]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Adaptive In-Context Learning** — an instance-level ICL paradigm that selects and orders demonstrations separately for each input instead of reusing one global prompt organization.

## Key Points

- The paper formalizes self-adaptive ICL as a combinatorial search problem over both demonstration selection and permutation.
- It replaces corpus-level prompt choice with per-instance organization, aiming to reduce majority bias and prompt instability.
- The proposed implementation uses a two-stage select-then-rank pipeline rather than exhaustive search.
- Its ranking criterion is unsupervised and grounded in expected description length under the model's predictive distribution.
- Empirically, the method substantially improves average accuracy over random, validation-based corpus-level, and prior instance-level baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-selfadaptive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-selfadaptive]].
