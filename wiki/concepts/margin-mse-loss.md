---
type: concept
title: Margin-MSE Loss
slug: margin-mse-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [MarginMSE, 边际均方误差损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Margin-MSE Loss** (边际均方误差损失) — a teacher-student ranking objective that regresses the student's score margin between a positive and a negative passage toward the teacher's corresponding score margin.

## Key Points

- The loss used in the paper is `MSE(M_s(q, p^+) - M_s(q, p^-), M_t(q, p^+) - M_t(q, p^-))`.
- It is applied both to static pairwise supervision from `BERT_CAT` and to in-batch supervision derived from ColBERT interactions.
- Compared with KLDiv, ListNet, and LambdaRank variants for in-batch training, Margin-MSE yields the strongest recall on both TREC-DL query sets.
- The paper argues that matching teacher score magnitudes, not only teacher orderings, is especially helpful for dense retriever recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hofst-tter-2021-efficiently-2104-06967]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hofst-tter-2021-efficiently-2104-06967]].
