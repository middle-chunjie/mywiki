---
type: concept
title: Embedding Inversion
slug: embedding-inversion
date: 2026-04-20
updated: 2026-04-20
aliases: [embedding inversion, 嵌入反演]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Embedding Inversion** (嵌入反演) — the task of reconstructing an original input from a dense representation produced by an encoder, typically by searching for text whose embedding matches a target vector.

## Key Points

- The paper formulates textual inversion as `x̂ = argmax_x cos(φ(x), e)` under black-box access to the encoder `φ`.
- Vec2Text shows that modern text embeddings can be inverted far more accurately than prior bag-of-words recovery methods.
- On GTR-base Wikipedia passages of `32` tokens, the paper reaches `92.0%` exact reconstruction with iterative search.
- Recovery quality tracks embedding-space similarity closely, suggesting that geometric closeness contains unusually detailed lexical information.
- The paper treats successful inversion as a concrete privacy threat for vector database deployments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2023-text-2310-06816]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2023-text-2310-06816]].
