---
type: concept
title: Memory Token
slug: memory-token
date: 2026-04-20
updated: 2026-04-20
aliases: [memory tokens]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memory Token** — a globally accessible marker used to preserve information from code statements with broad scope, such as imports and definitions, so later completions can retrieve them efficiently.

## Key Points

- In LongCoder, memory tokens are attached to statements identified from the code's syntax tree rather than placed at uniform intervals.
- They are meant to preserve imports, classes, functions, and structures whose effects can recur far from their original position.
- The model allocates up to `k = 64` global tokens for this mechanism in the reported setup.
- Removing memory tokens lowers Exact Match by roughly `1%` on the LCC ablation study, indicating that globally scoped identifiers materially affect completion accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2023-longcoder-2306-14893]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2023-longcoder-2306-14893]].
