---
type: concept
title: Plackett-Luce Ranking
slug: plackett-luce-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [Plackett-Luce model, partial Plackett-Luce ranking, Plackett-Luce 排序模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Plackett-Luce Ranking** (Plackett-Luce 排序模型) — a probabilistic ranking model over permutations that factorizes the probability of each rank position through successive softmax choices.

## Key Points

- Syntriever uses a partial Plackett-Luce construction to model the relation `c_i^+ ≻ c_i^- ≻` in-batch negatives for retriever alignment.
- The paper derives the alignment loss by marginalizing over all unspecified orderings of the in-batch negatives, so only the top-two preference constraints are fixed.
- This gives the alignment stage a contrastive form that keeps both preferred and less-preferred top-`K` passages away from irrelevant batch items.
- The resulting objective regularizes alignment toward the geometry learned in the distillation stage, reducing forgetting relative to simpler pairwise preference models.
- On FiQA and NFCorpus, partial Plackett-Luce outperforms Bradley-Terry by large margins in nDCG@10.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kim-2025-syntriever-2502-03824]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kim-2025-syntriever-2502-03824]].
