---
type: concept
title: Unified Embedding Model
slug: unified-embedding-model
date: 2026-04-20
updated: 2026-04-20
aliases: [Unified Retrieval Embedding, 统一嵌入模型]
tags: [retrieval, embedding, multi-task]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Unified Embedding Model** (统一嵌入模型) — a single dense text embedding model trained jointly across multiple heterogeneous retrieval tasks, such that it supports diverse query-document semantic relationships without task-specific model variants.

## Key Points

- Contrasts with two prior paradigms: task-specific retrievers (high performance on one task, poor transferability) and general-purpose retrievers (broad coverage, not optimized for LLM augmentation).
- Training challenge: different retrieval tasks encode qualitatively different semantic relationships (e.g., factual QA vs. stylistic example retrieval vs. tool API matching), which can interfere during multi-task optimization.
- Task disambiguation at inference time is achieved via instruction-based fine-tuning: a task-specific prefix instruction is prepended to the query before encoding, giving the model distinct activation patterns per task.
- Homogeneous in-batch negative sampling is critical: keeping negatives within the same task prevents cross-task negatives from being trivially easy, preserving discriminative power.
- LLM-Embedder (Zhang et al., 2023) is the first unified model covering all four LLM augmentation modes: knowledge, memory (long-context), example (in-context learning), and tool retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-retrieve-2310-07554]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-retrieve-2310-07554]].
