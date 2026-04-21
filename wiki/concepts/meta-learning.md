---
type: concept
title: Meta-Learning
slug: meta-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [learning to learn, 元学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Meta-Learning** (元学习) — a learning paradigm in which model parameters or update rules are optimized through higher-level feedback about how well lower-level learning improves downstream performance.

## Key Points

- MCLRec uses meta-learning to decouple encoder optimization from augmenter optimization rather than training both modules jointly in one step.
- In stage 1, the encoder is updated with `L_0 = L_rec + lambda L_cl1 + beta L_cl2 + gamma R`, so the recommendation objective and contrastive objectives shape the latent sequence representation.
- In stage 2, the learned encoder is fixed and the two augmenters are updated with `L_1 = L_cl2 + gamma R`, letting the augmentation models adapt to encoder behavior.
- The paper argues that this two-stage design reduces the objective mismatch between encoder and augmenter training and yields more discriminative augmented views than joint learning.
- Empirically, the meta-optimized variant outperforms the joint-learning variant MCLRec-J across all reported datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qin-2023-metaoptimized]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qin-2023-metaoptimized]].
