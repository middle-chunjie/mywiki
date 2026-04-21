---
type: concept
title: Lexical Unit
slug: lexical-unit
date: 2026-04-20
updated: 2026-04-20
aliases: [lexical units, 词汇单元]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Lexical Unit** (词汇单元) — the basic chunk of text used for scoring or filtering, such as a token, phrase, or sentence.

## Key Points

- [[li-2023-compressing]] defines the pruning granularity through lexical units rather than forcing all decisions at the token level.
- The paper studies token-, phrase-, and sentence-level units and finds phrase-level filtering to be the most effective overall.
- Scores for longer units are obtained by summing the self-information of constituent tokens.
- The choice of unit changes the trade-off between coherence and precision: token-level pruning can become fragmented, while sentence-level pruning is more unstable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
