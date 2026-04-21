---
type: concept
title: Language Modeling Loss
slug: language-modeling-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [LM loss, 语言建模损失]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Modeling Loss** (语言建模损失) — an autoregressive token-level training objective that maximizes the likelihood of target text, often used to teach a model to generate query-conditioned outputs.

## Key Points

- [[liu-2025-gear-2501-02772]] uses `L_LM` as the second training objective alongside contrastive retrieval supervision.
- In GeAR, `L_LM` is optimized over the full vocabulary with a lightweight causal decoder conditioned on fused query-document representations.
- The overall objective is `L_GeAR = L_CL + alpha * L_LM`, with reported `alpha = 0.25`.
- The paper argues that `L_LM` is the main driver of improved local information retrieval because it teaches the model to generate intrinsic query-relevant content from the document.
- Ablations show that removing `L_LM` sharply hurts local retrieval on NQ, PAQ, and the synthetic RIR benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-gear-2501-02772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-gear-2501-02772]].
