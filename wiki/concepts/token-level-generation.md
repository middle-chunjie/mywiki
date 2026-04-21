---
type: concept
title: Token-Level Generation
slug: token-level-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [token-level decoding, 词元级生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Token-Level Generation** (词元级生成) — a generation strategy that makes control or selection decisions at each next-token step instead of only at sequence or document level.

## Key Points

- Tok-RAG runs pure LLM and RAG decoding in parallel and only intervenes when the two systems propose different next tokens.
- The selection rule compares `cos(w_RAG, w_IR)` against `cos(w_RAG, w_LLM)` to decide which token to keep.
- The chosen token is appended to the prefix and reused by both generators at the next step, creating collaborative decoding.
- The paper argues that token-level control is finer-grained than passage filtering or sequence-level post-hoc correction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2025-theory-2406-00944]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2025-theory-2406-00944]].
