---
type: concept
title: Co-Occurrence
slug: co-occurrence
date: 2026-04-20
updated: 2026-04-20
aliases: [co-occurrence statistics, 共现]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Co-Occurrence** (共现) — the frequency with which two textual items appear together in a corpus, used here to rank keywords associated with copyrighted character names.

## Key Points

- The paper counts whether a candidate keyword and a target character name co-occur in documents from LAION-2B, C4, OpenWebText, and The Pile, then sorts by that count.
- LAION-based co-occurrence is the strongest keyword-ranking signal in the reported image and video experiments.
- Using only `5` co-occurrence-ranked LAION keywords nearly matches the effectiveness of much longer `60`-word descriptions.
- Co-occurrence evidence also guides negative-prompt construction, where LAION keywords reduce `DETECT` more than LM-ranked alternatives.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-fantastic-2406-14526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-fantastic-2406-14526]].
