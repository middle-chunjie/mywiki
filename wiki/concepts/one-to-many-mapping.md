---
type: concept
title: One-to-Many Mapping
slug: one-to-many-mapping
date: 2026-04-20
updated: 2026-04-20
aliases: [1-to-N mapping, 一对多映射]
tags: [retrieval, indexing]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**One-to-Many Mapping** (一对多映射) — a design in which one source item is intentionally associated with multiple target identifiers or representations so different inputs can retrieve or reference the same underlying item.

## Key Points

- Few-Shot GR assigns multiple docids to each document instead of a single identifier.
- The motivation is that one document can satisfy multiple diverse information needs, making a single lexicalized docid too restrictive.
- Empirically, increasing the number of docids per document improves retrieval quality, with gains saturating around `10` generated docids.
- The paper treats one-to-many mapping as one of the two main drivers of its performance gains, together with choosing a strong LLM backbone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[askari-2024-generative-2408-02152]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[askari-2024-generative-2408-02152]].
