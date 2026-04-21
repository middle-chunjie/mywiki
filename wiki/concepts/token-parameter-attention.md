---
type: concept
title: Token-parameter attention
slug: token-parameter-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [Pattention, token parameter attention, 词元-参数注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Token-parameter attention** (词元-参数注意力) — an attention-style computation in which input tokens query learnable parameter tokens, replacing fixed linear projections with dynamic token-to-parameter interactions.

## Key Points

- TokenFormer defines Pattention as `Θ(X K_P^T) V_P`, where `K_P` and `V_P` are learnable parameter-token sets rather than dense projection matrices.
- The paper uses token-parameter attention for `Q`, `K`, `V`, and output projection generation, and also reuses it as the feed-forward block.
- Its activation `Θ` uses `L2` normalization followed by `GeLU`, not exponential softmax, to reduce gradient collapse during training.
- Because parameter-token count `n` is independent of token-state width, the mechanism creates a separate parameter-scaling axis.
- Concatenating new parameter-token pairs enables incremental expansion while preserving the original function when new keys are zero-initialized.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-tokenformer-2410-23168]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-tokenformer-2410-23168]].
