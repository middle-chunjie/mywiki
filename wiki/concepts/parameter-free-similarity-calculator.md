---
type: concept
title: Parameter-Free Similarity Calculator
slug: parameter-free-similarity-calculator
date: 2026-04-20
updated: 2026-04-20
aliases: [parameter free similarity calculator, 无参数相似度计算器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Parameter-Free Similarity Calculator** (无参数相似度计算器) — a retrieval scoring module that aggregates and compares cross-modal representations without introducing additional trainable parameters.

## Key Points

- The paper scores each video frame against the text embedding with inner products `\alpha_j = <t, f_{i,j}>`.
- Frame scores are softmax-normalized with temperature `\tau = 5` to build a query-aware weighted average of video frames.
- This design replaces more parameter-rich similarity modules used by some competing text-video retrieval systems.
- The paper uses the same parameter-free scoring form when re-implementing several prompt-based baselines for fair comparison.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
