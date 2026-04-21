---
type: concept
title: Representational Instruction Tuning
slug: representational-instruction-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [representational instruction tuning, 表征指令微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Representational Instruction Tuning** (表征指令微调) — a tuning paradigm in which a model learns to map an instruction-conditioned text input to a semantic vector representation suitable for retrieval or similarity-based tasks.

## Key Points

- In this paper, representational instruction tuning is the embedding half of GRIT and is trained with a contrastive objective over query-document pairs.
- The representation is built from final hidden states after bidirectional attention and mean pooling rather than from generated text.
- Instructions specify the retrieval intent or task framing and still influence the embedding even though instruction tokens are excluded from the final pooled vector.
- The paper reports that E5-style representational data, same-dataset in-batch negatives, and large embedding batches are especially helpful for this tuning regime.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[muennighoff-2024-generative-2402-09906]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[muennighoff-2024-generative-2402-09906]].
