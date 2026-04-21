---
type: concept
title: Source Awareness
slug: source-awareness
date: 2026-04-20
updated: 2026-04-20
aliases: [source-aware reasoning, provenance awareness, 来源感知]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Source Awareness** (来源感知) — explicit tracking of where each evidence item comes from so the model can reason differently about internally generated and externally retrieved information.

## Key Points

- Astute RAG annotates each passage with its source and preserves those labels throughout consolidation and final answering.
- The paper argues source labels help the model weigh cross-source confirmation, frequency, and reliability when evidence conflicts.
- Source-awareness is implemented simply through source indicators `S_0`, making it compatible with black-box APIs.
- An ablation without source labels reduces Gemini overall accuracy from `59.2` to `58.6`, showing that provenance matters even in a prompt-only setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-astute-2410-07176]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-astute-2410-07176]].
