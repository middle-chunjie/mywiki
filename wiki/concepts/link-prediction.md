---
type: concept
title: Link Prediction
slug: link-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [link prediction, 链接预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Link Prediction** (链接预测) - a graph learning task that estimates whether an edge should exist between two nodes.

## Key Points

- RAGraph treats dynamic recommendation as a link-prediction problem within its unified graph-task definition.
- The paper predicts a link when the similarity between task-output vectors exceeds a margin relative to known linked nodes.
- Dynamic experiments are reported on TAOBAO, KOUBEI, and AMAZON using `Recall@20` and `nDCG@20`.
- Fine-tuned and noise-tuned RAGraph variants improve over several recommendation-oriented graph baselines on these datasets.
- Retrieved toy graphs provide additional historical, environmental, structural, and semantic evidence during prediction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragraph-2410-23855]].
