---
type: concept
title: Concept Extraction
slug: concept-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [概念抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Concept Extraction** (概念抽取) — the task of identifying words or phrases in text that denote salient domain concepts and labeling their token spans.

## Key Points

- [[fang-2021-guided]] formulates concept extraction as a token-level sequence labeling problem with labels `S-CON`, `B-CON`, `I-CON`, `E-CON`, and `O`.
- The paper argues that structured signals such as titles, topic distributions, and clue words are useful supervision beyond span annotations alone.
- GACEN combines topic-aware and position-aware attention to improve both recall and precision in concept extraction.
- The task is evaluated on three datasets with very different genres: CSEN course captions, KP-20K scientific articles, and MTB mathematics textbooks.
- The paper emphasizes low-resource benefit: additional clue-word supervision helps the model perform well when labeled data is limited.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
