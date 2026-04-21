---
type: concept
title: Cross-Task Generalization
slug: cross-task-generalization
date: 2026-04-20
updated: 2026-04-20
aliases: [task transfer generalization, 跨任务泛化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-task generalization** (跨任务泛化) — the ability of a model trained on one retrieval supervision regime to maintain strong performance on different downstream tasks or datasets.

## Key Points

- The paper argues that changing retrieval granularity can improve generalization without modifying retriever parameters.
- Proposition indexing gives the clearest gains on SQuAD and EntityQuestions, which are less aligned with some retrievers' training data.
- Benefits are strongest for unsupervised retrievers, suggesting finer units compensate for weaker task-specific supervision.
- The entity-frequency analysis shows larger gains on questions about rarer entities, linking granularity to robustness on long-tail information.
- The paper frames proposition indexing as an orthogonal strategy to data augmentation or retriever retraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-dense-2312-06648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-dense-2312-06648]].
