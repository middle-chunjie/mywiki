---
type: concept
title: Dynamic Listwise Distillation
slug: dynamic-listwise-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [动态列表蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Listwise Distillation** (动态列表蒸馏) — a joint-training strategy where a retriever and a re-ranker exchange soft listwise relevance distributions and update simultaneously instead of distilling from a frozen teacher.

## Key Points

- In [[ren-2023-rocketqav-2110-07367]], both the dual-encoder retriever and cross-encoder re-ranker score the same candidate list for each query.
- The method normalizes scores with a softmax over the candidate list, yielding `\\tilde{s}_de(q,p)` and `\\tilde{s}_ce(q,p)` as listwise relevance distributions.
- Mutual training minimizes `L_KL` between the two distributions while still supervising the re-ranker with a listwise cross-entropy objective.
- The approach differs from earlier RocketQA-style distillation by updating the re-ranker during training instead of keeping it fixed.
- Ablations show static distillation reduces MSMARCO retriever `MRR@10` from `37.4` to `36.0`, supporting the value of the dynamic variant.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-rocketqav-2110-07367]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-rocketqav-2110-07367]].
