---
type: concept
title: Topic Model
slug: topic-model
date: 2026-04-20
updated: 2026-04-20
aliases: [topic models, 主题模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Topic Model** (主题模型) — a probabilistic model that represents documents using latent topic variables which explain patterns of word occurrence.

## Key Points

- [[wang-2024-large-2301-11916]] explicitly borrows the intuition of neural topic models to motivate a latent concept variable for LLM prompting.
- The paper treats the latent concept as a modernized analogue of a topic variable, but allows it to be continuous and task-specific rather than discrete only.
- Topic-model intuition is used to argue that a prompt may be compressible into an approximate sufficient statistic for downstream prediction.
- The proposed concept tokens operationalize this topic-like latent state inside a standard autoregressive LM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-large-2301-11916]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-large-2301-11916]].
