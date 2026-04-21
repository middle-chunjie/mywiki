---
type: concept
title: Bidirectional Attention
slug: bidirectional-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [bidirectional attention, 双向注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Bidirectional Attention** (双向注意力) — an attention pattern in which each token can attend to both preceding and following tokens, allowing token representations to incorporate full-sequence context.

## Key Points

- LLM2Vec enables bidirectional attention by replacing the causal mask with an all-ones attention mask.
- Naively enabling bidirectional attention usually hurts embedding performance for S-LLaMA-1.3B and LLaMA-2-7B, so adaptation is necessary.
- Mistral-7B is an exception: bidirectional attention already improves its unsupervised sequence-level scores before any fine-tuning.
- The paper's representation analysis suggests that Mistral hidden states change much less than other backbones when bidirectionality is enabled.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
