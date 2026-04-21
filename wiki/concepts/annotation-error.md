---
type: concept
title: Annotation Error
slug: annotation-error
date: 2026-04-20
updated: 2026-04-20
aliases: [label error, 标注错误]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Annotation Error** (标注错误) — a flaw in gold labels or evaluation setup that causes a benchmark item to be misleading, invalid, or effectively unsolvable.

## Key Points

- A central claim of the paper is that leaderboard analysis should identify annotation errors rather than silently absorb them into aggregate accuracy.
- Negative discriminability and low feasibility are both used as signals that an item may contain annotation problems.
- Manual inspection of `60` SQuAD items shows that IRT-selected suspicious examples have a much higher rate of flawed or wrong annotations.
- The paper gives concrete examples where answer spans are incomplete, too long, or incompatible with the benchmark's scoring protocol.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
