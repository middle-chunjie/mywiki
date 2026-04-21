---
type: concept
title: Binary-Source Code Matching
slug: binary-source-code-matching
date: 2026-04-20
updated: 2026-04-20
aliases: [binary-to-source matching, 二进制-源代码匹配]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Binary-Source Code Matching** (二进制-源代码匹配) — the task of determining whether a binary artifact and a source-code artifact implement the same functionality.

## Key Points

- The paper extends binary-source matching from same-language settings to cross-language settings.
- XLIR compares binary and source embeddings after both are translated into LLVM-IR.
- Matching is decided with cosine similarity and a default threshold of `0.8`.
- The method is evaluated on both curated cross-language data and the single-language POJ-104 benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gui-2022-crosslanguage-2201-07420]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gui-2022-crosslanguage-2201-07420]].
