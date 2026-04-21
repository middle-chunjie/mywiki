---
type: concept
title: Multimodal Reasoning
slug: multimodal-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal reasoning, 多模态推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Reasoning** (多模态推理) — reasoning that jointly uses information from multiple modalities such as text and images to derive intermediate steps and final answers.

## Key Points

- [[dong-2024-progressive-2412-14835]] frames multimodal reasoning as especially vulnerable to cross-modal misalignment because small early errors propagate across later steps.
- The paper models multimodal reasoning as autoregressive generation over step sequences conditioned on a multimodal query `Q^m = (x,t)` and retrieved evidence.
- AR-MCTS improves multimodal reasoning by combining step-specific retrieval, tree search, and process-level verification instead of relying on beam search over internal model knowledge alone.
- The reported gains span both mathematical and general exam-style settings, suggesting the method is not limited to one benchmark family.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2024-progressive-2412-14835]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2024-progressive-2412-14835]].
