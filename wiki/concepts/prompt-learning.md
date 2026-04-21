---
type: concept
title: Prompt Learning
slug: prompt-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt tuning, 提示学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Learning** (提示学习) — a parameter-efficient adaptation approach that keeps a pretrained backbone frozen and optimizes only task-specific prompt tokens or embeddings.

## Key Points

- The paper treats CoOp, VPT, VL-Prompt, and MaPLe as the main prompt-based baselines for text-video retrieval.
- It argues that uni-modal prompt tuning is especially weak for multimodal retrieval because it fails to model cross-modal alignment adequately.
- Even multimodal prompt variants remain worse than the proposed cross-modal adapter on the reported datasets.
- The study uses prompt-token count as the main knob controlling the parameter budget for prompt baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
