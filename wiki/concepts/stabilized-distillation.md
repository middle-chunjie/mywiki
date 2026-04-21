---
type: concept
title: Stabilized Distillation
slug: stabilized-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [Stabilised Distillation, 稳定蒸馏]
tags: [knowledge-distillation, training, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Stabilized Distillation** (稳定蒸馏) — a knowledge distillation variant that combines soft reward-based weights with hard ranking-based loss structure to handle volatile teacher signals, preventing training collapse when teacher scores are either too tightly clustered or too polarized.

## Key Points

- Standard KL-divergence distillation fails when teacher (LLM) reward distributions are degenerate: closely clustered rewards provide no learning signal; polarized (near-one-hot) rewards collapse distillation to a trivial step.
- Stabilized distillation re-ranks candidates by reward (descending order `p_1, …, p_N`) and uses lower-ranked candidates as the denominator in the contrastive loss, so the model always learns relative preferences regardless of absolute reward magnitudes.
- The soft weight `w_i = softmax(r_i / α)` still scales the loss per candidate, but the denominator set is `{p_{i+1}, …, p_N} ∪ in-batch-negatives`, not the full set.
- When rewards are polarized (one-hot), the loss reduces to standard contrastive learning with the top candidate as positive — preserving useful gradient signal.
- Introduced and empirically validated in LLM-Embedder (Zhang et al., 2023) for multi-task retrieval augmentation training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-retrieve-2310-07554]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-retrieve-2310-07554]].
