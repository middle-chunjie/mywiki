---
type: concept
title: Alignment
slug: alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [对齐性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Alignment** (对齐性) — the degree to which semantically matched examples are mapped close to one another in embedding space.

## Key Points

- SimCSE measures alignment with `` `l_align = E ||f(x) - f(x^+)||^2` `` over positive pairs.
- The paper argues that pre-trained checkpoints already provide useful alignment, which is why unsupervised SimCSE can work with only dropout noise.
- Removing dropout or forcing identical dropout masks hurts alignment severely even though uniformity may improve.
- Supervised SimCSE uses NLI entailment pairs to improve alignment further beyond what unsupervised self-prediction can provide.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2022-simcse-2104-08821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2022-simcse-2104-08821]].
