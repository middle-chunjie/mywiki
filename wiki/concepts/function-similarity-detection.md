---
type: concept
title: Function Similarity Detection
slug: function-similarity-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [function similarity detection, 函数相似性检测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Function Similarity Detection** (函数相似性检测) — the task of judging whether two functions are semantically similar from their learned code representations.

## Key Points

- The paper uses mean-pooled function embeddings and cosine similarity between two learned representations to solve this task.
- It motivates the task through security applications such as vulnerability search and malware-family analysis.
- On unseen permutations, SymC maintains `0.96` AUC across all transformation percentages while PalmTree degrades substantially.
- The paper also reports strong gains on unseen compilers, optimizations, obfuscations, and longer sequences for this task.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
