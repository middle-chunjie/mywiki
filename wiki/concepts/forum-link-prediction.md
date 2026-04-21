---
type: concept
title: Forum Link Prediction
slug: forum-link-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [论坛链接预测, post relatedness]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Forum Link Prediction** (论坛链接预测) — the task of determining whether two developer forum posts are semantically related enough to be linked.

## Key Points

- BLANCA uses linked vs. unlinked forum pairs as positive and negative examples for representation learning.
- The dataset contains `23,516` training pairs and `5,854` test pairs.
- Fine-tuning with contrastive loss sharply improves separation between linked and unlinked post embeddings.
- The paper also inspects direct, indirect, duplicate, and isolated StackOverflow relations to probe embedding structure.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
