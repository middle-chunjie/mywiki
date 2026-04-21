---
type: concept
title: In-Batch Memory
slug: in-batch-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [in-batch memories, training memories, 批内记忆]
tags: [language-model, memory, training, contrastive-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**In-Batch Memory** (批内记忆) — a training strategy where context-target pairs from the current training batch serve as an accessible memory set during the forward pass, enabling the model to compute memory-augmented next-token probabilities and back-propagate gradients to all memory representations simultaneously.

## Key Points

- Training memories `M_train = {(c_j, x_j)}` are constructed on the fly from the current batch, making them differentiable unlike frozen test-time datastores.
- The joint training objective (Eq. 5 in TRIME) sums the standard LM term `exp(E_w^T f_θ(c))` with a contrastive term over in-batch contexts `c_j` where `x_j = w`, using scaled dot-product similarity `sim(q,k) = q·k/√d`.
- Representation `g_θ` (input to the final FFN layer, rather than the output `f_θ`) was found empirically to produce better nearest-neighbor retrieval, consistent with kNN-LM findings.
- Batch construction strategy determines which memories are accessible: consecutive segments (long-term), BM25-similar segments (external), or default random segments (local only).
- The in-batch negative structure is similar to contrastive dense retrieval (DPR) but applied to the language modeling next-token prediction objective rather than a separate ranking loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2022-training-2205-12674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2022-training-2205-12674]].
