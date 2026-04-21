---
type: concept
title: Contrastive Perplexity
slug: contrastive-perplexity
date: 2026-04-20
updated: 2026-04-20
aliases: [对比困惑度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Perplexity** (对比困惑度) — a continuous retrieval evaluation metric that measures how likely a model is to rank a labeled positive passage above sampled negatives under a contrastive scoring distribution.

## Key Points

- The paper defines contrastive perplexity as `` `-log( exp(s(q, p+)) / (exp(s(q, p+)) + sum_j exp(s(q, p_j-))) )` `` over one positive and sampled negatives.
- It uses `W = 256` sampled unlabeled passages at evaluation time so the metric is smoother than cutoff-based metrics such as NDCG@K or MAP.
- The metric is strongly correlated with standard IR metrics, with recall@1000 showing an especially close linear relationship in the paper's experiments.
- Because its structure closely matches the training objective, the metric supports fitting scaling laws over model size and annotation count.
- The paper uses contrastive perplexity as the target quantity for model-size scaling, data-size scaling, annotation-quality analysis, and budget-aware prediction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2024-scaling-2403-18684]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2024-scaling-2403-18684]].
