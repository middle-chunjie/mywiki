---
type: concept
title: Self-Contrast Metric
slug: self-contrast-metric
date: 2026-04-20
updated: 2026-04-20
aliases: [self-contrast]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Self-Contrast Metric** (自对比指标) — a metric that compares two outputs from the same model under different information conditions to isolate a targeted ability.

## Key Points

- In KoLA's knowledge-creating tasks, the model first writes a free continuation `T` from context `C`, then writes a grounded continuation `T_k` from `C` plus annotated foreknowledge `K`.
- The benchmark uses `partial(T, T_k)` as the core self-contrast term, arguing that similarity between the two outputs indicates that the model's free generation is already close to grounded, reasonable event knowledge.
- KoLA reports Spearman correlation `0.61` between this term and human faithfulness judgments, and shows that removing it harms the aggregate metric's alignment with human evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-kola-2306-09296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-kola-2306-09296]].
