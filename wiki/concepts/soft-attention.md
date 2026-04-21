---
type: concept
title: Soft Attention
slug: soft-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [Soft Attention, 软注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft Attention** (软注意力) — an attention mechanism that assigns continuous weights across many input positions, producing a weighted combination rather than a discrete selection.

## Key Points

- The paper argues that transformer soft attention can spread mass onto irrelevant context, repeated content, and user opinions.
- This behavior is presented as one root cause of factual errors, sentiment leakage, and distractor sensitivity in LLM generation.
- S2A is motivated as an external corrective mechanism that sharpens what the model should attend to before final decoding.
- The ablation results support the claim that merely adding instructions is weaker than explicitly changing the attended context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weston-2023-system-2311-11829]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weston-2023-system-2311-11829]].
