---
type: concept
title: Pre-Trained Model
slug: pre-trained-model
date: 2026-04-20
updated: 2026-04-20
aliases: [PTM, pretrained model, 预训练模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pre-Trained Model** (预训练模型) — a model first optimized on large self-supervised data and then adapted to downstream tasks with task-specific supervision.

## Key Points

- The paper presents CodePTMs as the software-engineering analogue of NLP PTMs, motivated by the same labeled-data bottleneck.
- It argues that source-code PTMs should not simply retrain NLP models unchanged because code includes comments, syntax trees, and semantic program structure.
- The survey reviews `20` CodePTMs and shows that pre-training generally improves downstream SE performance, especially when task-specific data is scarce.
- The authors emphasize that transfer quality depends on how well the pre-training setup matches the downstream task type and modality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[niu-2022-deep-2205-11739]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[niu-2022-deep-2205-11739]].
