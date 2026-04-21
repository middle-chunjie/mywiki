---
type: concept
title: Enumerative Search
slug: enumerative-search
date: 2026-04-20
updated: 2026-04-20
aliases: [enumerative search, 枚举搜索]
tags: [search, program-synthesis]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Enumerative Search** (枚举搜索) — a synthesis strategy that systematically explores candidate programs generated from a grammar or DSL and tests them against the specification.

## Key Points

- [[balog-2017-deepcoder-1611-01989]] presents DFS enumeration as a strong baseline for DSL program synthesis from input-output examples.
- The paper introduces a sort-and-add variant that repeatedly enlarges an active set of allowed functions according to neural predictions and restarts search.
- Appendix D reports an optimized DFS implementation with prefix caching that explores roughly `3 x 10^6` programs per second.
- Neural guidance yields major improvements for enumerative methods, including `62.2x` speedup at 20% solved on `T = 3` tasks and `907x` on the main `T = 5` setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[balog-2017-deepcoder-1611-01989]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[balog-2017-deepcoder-1611-01989]].
