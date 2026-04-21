---
type: concept
title: Hidden-State Embedding
slug: hidden-state-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [HS embedding, 隐状态嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hidden-State Embedding** (隐状态嵌入) — a sentence representation extracted from a language model's hidden activations, typically from the final layer or the final token.

## Key Points

- The paper uses the last token's final-layer hidden state as the main hidden-state baseline for decoder-only MoE LLMs.
- Hidden-state embeddings align well with output-oriented information and can be competitive on classification tasks.
- Compared with routing-weight embeddings, HS is more sensitive to prompt wording, with lower cross-prompt consistency.
- Using all layers or all-token pooling does not beat the simple last-token final-layer baseline in the paper's STS ablations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-your-2410-10814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-your-2410-10814]].
