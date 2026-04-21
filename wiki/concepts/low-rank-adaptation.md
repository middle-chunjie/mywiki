---
type: concept
title: Low-Rank Adaptation
slug: low-rank-adaptation
date: 2026-04-20
updated: 2026-04-20
aliases: [LoRA, low-rank adaptation, 低秩适配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Low-Rank Adaptation** (低秩适配) — a parameter-efficient fine-tuning method that updates a model through learned low-rank adapters instead of modifying all backbone weights directly.

## Key Points

- LLM2Vec uses LoRA for both the MNTP and SimCSE stages instead of full fine-tuning.
- The paper keeps the same adapter size across stages with `r = 16` and `alpha = 32`.
- MNTP LoRA weights are merged into the base model before starting SimCSE, then new LoRA parameters are initialized.
- The adapter-based setup is part of the paper's claim that decoder-only LLMs can be turned into strong embedders in a parameter-efficient way.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
