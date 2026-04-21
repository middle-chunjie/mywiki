---
type: concept
title: Character-Level Span Selection
slug: character-level-span-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [character-level FIM span selection, 字符级跨度选择]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Character-Level Span Selection** (字符级跨度选择) — choosing FIM span boundaries in raw character space rather than only on token or line boundaries.

## Key Points

- The paper selects split points uniformly at random over characters so the model regularly sees boundaries that cut through subword units.
- This reduces train-test mismatch for practical code editing, where a user may delete text from the middle of a token.
- In the reported ablation, character-level span selection gives the best random-span infilling pass rate (`0.321`) by a large margin.
- The authors recommend always mixing in character-level random spans as a default training practice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
