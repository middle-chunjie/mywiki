---
type: concept
title: Recursive Summarization
slug: recursive-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [recursive summary generation, 递归摘要]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Recursive Summarization** (递归摘要) — a hierarchical summarization process that repeatedly compresses groups of lower-level nodes into higher-level summary nodes.

## Key Points

- [[unknown-nd-sirerag]] uses recursive summaries on both the similarity tree and the relatedness tree rather than only on raw text chunks.
- On the similarity side, recursive summaries operate over clusters of semantically similar chunks in a `4`-level tree.
- On the relatedness side, recursive summaries are built over proposition aggregates instead of over individual propositions.
- Removing the relatedness-side recursive summaries lowers average F1 from `65.83` to `64.04`, indicating that the summaries add useful synthesized evidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-sirerag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-sirerag]].
