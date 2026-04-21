---
type: concept
title: Intent Awareness
slug: intent-awareness
date: 2026-04-20
updated: 2026-04-20
aliases: [intent-aware, intent awareness, 意图感知]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Intent Awareness** (意图感知) — the ability of a system to detect whether a user's instruction explicitly requests externally grounded or retriever-backed answers.

## Key Points

- UAR adds intent awareness as a retrieval criterion that is largely absent from earlier active retrieval methods.
- The paper notes that some users want verifiable answers or explicit external information even when the base model could answer from memory.
- Intent-aware training data are built by generating `3,000` retrieval intents with Self-Instruct and concatenating them with sampled user inputs.
- In AR-Bench, intent awareness is one of UAR's strongest dimensions, reaching over `91` accuracy on both 7B and 13B models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
