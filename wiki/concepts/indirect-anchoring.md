---
type: concept
title: Indirect Anchoring
slug: indirect-anchoring
date: 2026-04-20
updated: 2026-04-20
aliases: [indirect anchor, 间接锚定]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Indirect Anchoring** (间接锚定) — prompting a generative model with descriptive words or attributes that evoke a copyrighted target without explicitly naming it.

## Key Points

- The paper defines indirect anchoring as the key failure mode that bypasses naive safeguards based on character-name detection alone.
- GPT-4-generated `60`-word descriptions produce copyrighted characters for about `~48%` of the studied targets despite omitting the names.
- Short keyword sets can also act as indirect anchors, especially when ranked by LAION co-occurrence or embedding similarity.
- The same indirect anchors transfer beyond open image generators to DALL-E 3 and a video generation system.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-fantastic-2406-14526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-fantastic-2406-14526]].
