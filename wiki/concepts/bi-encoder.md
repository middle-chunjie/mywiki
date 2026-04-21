---
type: concept
title: Bi-Encoder
slug: bi-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [Bi Encoder, 双编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bi-Encoder** (双编码器) — an architecture that encodes two inputs separately into vector representations and scores their relationship with a similarity function such as cosine similarity.

## Key Points

- For C-STS, the paper encodes each sentence together with the condition using a Siamese network and scores the pair with `-cos(f_theta(s_1; c), f_theta(s_2; c))`.
- The setup is compatible with contrastively pre-trained sentence encoders such as SimCSE and DiffCSE.
- Fine-tuned bi-encoders are the strongest encoder baselines in the paper, outperforming both cross-encoder and tri-encoder variants on test Spearman correlation.
- SimCSE-Large bi-encoder reaches the best overall encoder result, Spearman `47.5 ± 0.1`, but still leaves a large gap to human-level conditional reasoning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deshpande-2023-csts-2305-15093]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deshpande-2023-csts-2305-15093]].
