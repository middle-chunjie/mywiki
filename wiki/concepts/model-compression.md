---
type: concept
title: Model Compression
slug: model-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [compression, 模型压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model Compression** (模型压缩) — a family of methods that reduces model size, memory footprint, or inference cost through distillation, pruning, low-rank approximation, or related structural simplifications.

## Key Points

- The survey treats knowledge distillation and network pruning as the two major compression branches for efficient LLM serving.
- It distinguishes white-box distillation from black-box distillation motivated by API-based teacher models such as ChatGPT-style services.
- The paper notes that pruning is only valuable when sparsity patterns map cleanly to real system speedups rather than merely reducing parameter count on paper.
- Recent unstructured pruning methods are summarized as often reaching about `50-60%` sparsity, while semi-structured `2:4` and `4:8` patterns better align with sparse tensor-core execution.
- Compression is framed as a memory-and-throughput optimization that must be balanced against retraining cost and accuracy degradation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[miao-2023-efficient-2312-15234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[miao-2023-efficient-2312-15234]].
