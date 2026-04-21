---
type: concept
title: Zero-Shot Retrieval
slug: zero-shot-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [zero-shot dense retrieval, 零样本检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Retrieval** (零样本检索) — retrieval evaluation in which a model is applied to a target retrieval task without task-specific supervised finetuning on that task.

## Key Points

- SANTA is explicitly designed to improve zero-shot retrieval by making pretrained encoders more sensitive to structured semantics before downstream finetuning.
- On the main code-search benchmark, SANTA reaches `MRR = 46.1` zero-shot, far above standard pretrained baselines and ahead of CodeRetriever.
- On product search, the same pretraining recipe lifts zero-shot `NDCG@100` from the low `70s` for T5 to `76.38/77.14`.
- The appendix shows the gains transfer to CodeSearch and Adv, suggesting that the learned structure-aware representations generalize beyond a single benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structureaware-2305-19912]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structureaware-2305-19912]].
