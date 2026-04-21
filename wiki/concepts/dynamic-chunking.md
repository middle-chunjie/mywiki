---
type: concept
title: Dynamic Chunking
slug: dynamic-chunking
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive chunking, variable-length chunking, 动态切块]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Chunking** (动态切块) — a chunking strategy that adjusts chunk boundaries and lengths according to content transitions instead of using a fixed window or separator rule.

## Key Points

- LumberChunker operationalizes dynamic chunking by letting an LLM decide where semantic divergence begins inside a paragraph group.
- The method allows chunks to be shorter or longer depending on how quickly the narrative content changes.
- Threshold `θ` controls only the size of the local decision context, not the final chunk length itself.
- The reported average chunk length is `334` tokens, substantially below the tuned prompt threshold of `550`, indicating that the model often selects boundaries well before the end of the local window.
- The paper argues that dynamic chunking better captures semantically independent retrieval units than paragraph-level or recursive baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[duarte-2024-lumberchunker-2406-17526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[duarte-2024-lumberchunker-2406-17526]].
