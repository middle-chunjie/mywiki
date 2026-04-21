---
type: concept
title: Clustering-Based Prompting
slug: clustering-based-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [cluster-based prompting, 基于聚类的提示构造]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Clustering-Based Prompting** (基于聚类的提示构造) — a prompting strategy that clusters example embeddings and samples demonstrations from different clusters so generation covers distinct latent perspectives rather than repeating one prompt distribution.

## Key Points

- [[yu-2023-generate-2209-10063]] generates one initial document per training question, embeds each `[q_i, d_i]` pair with GPT-3 into a `12,288`-dimensional vector, and applies K-means clustering.
- The number of clusters `K` equals the number of documents to be generated, so each cluster can seed one distinct generation path.
- For each cluster, the prompt uses `n = 5` sampled question-document demonstrations as in-context examples before the new query.
- The method is designed to improve coverage relative to repeated prompting or [[top-p-sampling]], which often produces redundant generated documents.
- In the paper, clustering prompts improve both Recall@K and downstream EM over sampling and over prompts sampled globally from the whole dataset.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-generate-2209-10063]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-generate-2209-10063]].
