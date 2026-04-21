---
type: concept
title: Mode Collapse
slug: mode-collapse
date: 2026-04-20
updated: 2026-04-20
aliases: [模式坍塌]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Mode Collapse** (模式坍塌) — a generative-model failure mode in which the model covers too few distinct modes or produces insufficiently varied samples despite apparent quality.

## Key Points

- The paper uses Vendi Score to show that number-of-modes metrics can miss finer-grained diversity failures in GAN outputs.
- On Stacked-MNIST, models with identical `nom = 1000` still differ substantially under Vendi Score, revealing residual redundancy.
- The authors argue that diversity should reflect not only which classes appear but also the internal spread within and across those classes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
