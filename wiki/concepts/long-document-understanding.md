---
type: concept
title: Long Document Understanding
slug: long-document-understanding
date: 2026-04-20
updated: 2026-04-20
aliases: [Long Document Understanding, 长文档理解]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long Document Understanding** (长文档理解) — the problem of modeling documents that exceed a model's standard input window while preserving both document-level semantics and token-level context for downstream tasks.

## Key Points

- [[li-2024-chulo-2410-11119]] frames long document understanding as covering both document classification and token classification.
- ChuLo preserves full-document coverage by splitting a document into `m = ceil(l_D / n)` chunks instead of truncating content beyond a fixed token limit.
- The paper argues that emphasizing semantically important phrases inside each chunk helps retain global meaning after compression.
- Reported gains are strongest when inputs exceed `2048` or `4096` tokens, especially on LUN and CoNLL-2012.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-chulo-2410-11119]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-chulo-2410-11119]].
