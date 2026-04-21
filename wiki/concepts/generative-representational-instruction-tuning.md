---
type: concept
title: Generative Representational Instruction Tuning
slug: generative-representational-instruction-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [GRIT, generative representational instruction tuning, 生成式表征指令微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generative Representational Instruction Tuning** (生成式表征指令微调) — a joint fine-tuning recipe that trains one language model to handle both text generation and instruction-conditioned embedding tasks by switching behavior through format and loss design.

## Key Points

- GRIT combines representational instruction tuning and generative instruction tuning in a single backbone rather than training separate embedding and chat models.
- The method uses bidirectional attention plus mean pooling for embedding mode and causal language modeling for generation mode.
- The paper optimizes a summed objective over contrastive representation loss and generative next-token loss, with the final 7B recipe using a mixed token/sample generative loss.
- GRIT matches the performance of embedding-only and generative-only variants closely, supporting the paper's claim that the two capabilities can coexist without obvious trade-off.
- The unified formulation makes downstream systems simpler, enabling the same model to serve as retriever, reranker, and generator in RAG-style pipelines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[muennighoff-2024-generative-2402-09906]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[muennighoff-2024-generative-2402-09906]].
