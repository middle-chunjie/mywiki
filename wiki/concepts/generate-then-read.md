---
type: concept
title: Generate-Then-Read
slug: generate-then-read
date: 2026-04-20
updated: 2026-04-20
aliases: [generate then read, 生成后阅读]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generate-Then-Read** (生成后阅读) — a two-stage knowledge-intensive inference pipeline in which a language model first generates question-conditioned background documents and then answers by reading those generated contexts.

## Key Points

- [[yu-2023-generate-2209-10063]] defines GENREAD as an alternative to retrieve-then-read that uses parametric knowledge instead of fetching documents from an external corpus.
- The paper writes the answer distribution as `` `p(a|q) = \sum_i p(a|d_i, q) p(d_i|q)` ``, approximated in practice with a small set of generated documents.
- In zero-shot mode, the same InstructGPT model is used both to generate background text and to read it for final answer prediction.
- In supervised mode, generated documents are consumed by [[fusion-in-decoder]] readers, showing that the paradigm is not tied to one specific reader architecture.
- The paper reports that generated context often yields higher answer inclusion and better readability than top retrieved passages on several QA benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-generate-2209-10063]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-generate-2209-10063]].
