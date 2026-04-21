---
type: concept
title: Attribution Tracing
slug: attribution-tracing
date: 2026-04-20
updated: 2026-04-20
aliases: [data attribution tracing, provenance tracing, 归因追踪]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Attribution Tracing** (归因追踪) — the post-hoc inspection of model outputs against training data to estimate whether generated content overlaps with memorized source material.

## Key Points

- The paper treats attribution as a release requirement for open code LLMs, not only as an interpretability convenience.
- The proposed workflow combines a Bloom-filter membership check over `50`-character strings with a searchable BM25 index over the retained training files.
- The BM25 tool uses Elasticsearch `7.17`, lowercase and ASCII folding, `3`-gram tokenization, and metadata fields such as username and license for filtering.
- Both tools are integrated into the public demo so users can inspect potentially copied generations and recover matched repositories or source files.
- The authors explicitly note that the tools only cover the retained training subset and cannot reliably separate boilerplate code from protected content.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-starcoder-2305-06161]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-starcoder-2305-06161]].
