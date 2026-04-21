---
type: concept
title: Mutual Information Neural Estimation
slug: mutual-information-neural-estimation
date: 2026-04-20
updated: 2026-04-20
aliases: [MINE, 互信息神经估计]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Mutual Information Neural Estimation** (互信息神经估计) — a neural method for estimating and maximizing mutual information between two random variables using a learnable statistics network.

## Key Points

- RobustGER uses MINE to align a language-space noise embedding with noisy audio embeddings while pushing it away from clean-audio embeddings.
- The estimator optimizes `I_Theta(X; Z) = sup_theta E[P_XZ][psi_theta] - log E[P_X P_Z][exp(psi_theta)]`.
- The paper implements the statistics network as an MLP with Sigmoid output that consumes projected Whisper audio features and language-space noise features.
- Compared with teacher-student KL distillation and contrastive learning, MINE yields the best WER improvements in the paper's ablations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2401-10446]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2401-10446]].
