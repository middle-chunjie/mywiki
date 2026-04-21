---
type: concept
title: Query Formulation based on Self-Attention
slug: query-formulation-based-on-self-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [QFS, self-attention query formulation, 基于自注意力的查询构造]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query Formulation based on Self-Attention** (基于自注意力的查询构造) — a query-construction method that uses the triggering token's self-attention distribution over earlier context tokens to select the most relevant words for retrieval.

## Key Points

- [[su-2024-dragin-2403-10081]] extracts the last-layer attention vector at the trigger position, ranks previous tokens by attention score, and keeps the top `n` tokens as query terms.
- The selected tokens are reordered according to their original textual order so the final query stays readable and reflects the source context.
- QFS expands the query scope beyond the most recent sentence or local token window, allowing the query to draw evidence from anywhere in the current context.
- In the HotpotQA ablation, the paper reports DRAGIN's QFS query policy outperforming FLARE, fixed-sentence, fixed-length, and full-context alternatives for both Llama2-13B and Vicuna-13B.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2024-dragin-2403-10081]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2024-dragin-2403-10081]].
