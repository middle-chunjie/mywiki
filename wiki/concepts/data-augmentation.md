---
type: concept
title: Data Augmentation
slug: data-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [Synthetic Data Augmentation, 数据增强]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Augmentation** (数据增强) — the creation of additional training examples through controlled transformations or generation to improve robustness, coverage, and generalization.

## Key Points

- ConvAug performs augmentation at token, turn, and whole-conversation levels rather than relying on a single perturbation strategy.
- It generates both positive variants that preserve intent and hard negatives that preserve surface similarity while altering critical semantics.
- LLM generation is combined with rule-based masking and reordering so the training signal spans multiple granularities.
- The paper shows that removing any augmentation component reduces retrieval performance, validating the value of diversified synthetic contexts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
