---
type: concept
title: Back-Translation
slug: back-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [BT, 回译]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Back-Translation** (回译) — generating synthetic source-target sentence pairs by translating target-side monolingual data in the reverse direction and using the result as additional training data.

## Key Points

- The survey identifies back-translation as the central data-augmentation technique for low-resource NMT.
- It notes that target-side monolingual data is usually preferred because it improves target fluency more than starting from the source side.
- The paper highlights practical refinements such as iterative BT, monolingual-data selection, synthetic-data filtering, and explicit tagging or weighting of synthetic examples.
- It also warns that BT quality depends heavily on the seed MT system, synthetic-to-real data ratio, and domain match between monolingual and parallel corpora.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ranathunga-2021-neural-2106-15115]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ranathunga-2021-neural-2106-15115]].
