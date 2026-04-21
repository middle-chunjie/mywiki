---
type: concept
title: Training-Free Embedding
slug: training-free-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [training-free embedding, 免训练嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Training-Free Embedding** (免训练嵌入) — deriving usable semantic representations from a pretrained model without any additional representation-specific finetuning.

## Key Points

- This paper studies whether pretrained MoE LLMs can act as embedding models without any extra contrastive or supervised training.
- The method reuses hidden states and router outputs already produced by the model, treating routing as a free byproduct of inference.
- PromptEOL can still improve training-free performance, but the core gain comes from combining hidden-state and routing signals rather than updating model weights.
- The resulting MoEE method becomes competitive with, and in some prompt settings surpasses, supervised embedding baselines on MTEB.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-your-2410-10814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-your-2410-10814]].
