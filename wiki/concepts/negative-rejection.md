---
type: concept
title: Negative Rejection
slug: negative-rejection
date: 2026-04-20
updated: 2026-04-20
aliases: [rejecting unsupported questions]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Negative Rejection** — the ability of a retrieval-augmented model to refuse answering when none of the retrieved documents contain the required evidence.

## Key Points

- RGB constructs negative-rejection examples by supplying only noisy negative documents while instructing the model to emit an explicit insufficient-information response.
- The paper shows that current LLMs are poor at this behavior: exact rejection tops out at `31.00%` in English and `8.67%` in Chinese.
- ChatGPT-assisted judging raises the best observed rejection rates only to `45.00%` in English and `43.33%` in Chinese, indicating that instruction following is also unstable.
- The authors argue that negative rejection is especially important in real-world RAG because search engines frequently retrieve topically related but answer-free documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-benchmarking-2309-01431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-benchmarking-2309-01431]].
