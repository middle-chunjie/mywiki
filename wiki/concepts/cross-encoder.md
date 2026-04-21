---
type: concept
title: Cross-Encoder
slug: cross-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [Cross Encoder, 交叉编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-Encoder** (交叉编码器) — a scoring architecture that jointly encodes multiple inputs in a single transformer pass and predicts a relation or scalar score from the combined representation.

## Key Points

- In this paper, the cross-encoder baseline consumes the full triplet `(s_1, s_2, c)` and directly regresses a conditional similarity score.
- The formulation integrates sentence interaction and condition information early, rather than comparing separately encoded embeddings.
- Cross-encoders outperform weak zero-shot baselines on C-STS but remain below the best bi-encoder results after fine-tuning.
- The best reported test result in this paper for a cross-encoder is SimCSE-Large with Spearman `43.2 ± 1.2`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deshpande-2023-csts-2305-15093]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deshpande-2023-csts-2305-15093]].
