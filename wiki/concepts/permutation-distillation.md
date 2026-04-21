---
type: concept
title: Permutation Distillation
slug: permutation-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [permutation-based distillation, 排列蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Permutation Distillation** (排列蒸馏) — a distillation scheme that trains a smaller ranking model to imitate the ordered permutation output by a teacher model rather than only binary or pointwise relevance labels.

## Key Points

- The teacher is `gpt-3.5-turbo`, which produces a full ordering over `20` BM25-retrieved MS MARCO passages for each of `10K` sampled queries.
- The student learns from the teacher's relative ordering, enabling pairwise and listwise objectives such as RankNet and LambdaLoss.
- In this paper, permutation-distilled DeBERTa models consistently outperform supervised counterparts trained on original MS MARCO labels.
- A `435M` DeBERTa-Large distilled from ChatGPT reaches `53.03` average nDCG@10 on BEIR, exceeding `monoT5 (3B)` at `51.36`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-chatgpt-2304-09542]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-chatgpt-2304-09542]].
