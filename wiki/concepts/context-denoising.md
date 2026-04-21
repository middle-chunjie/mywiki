---
type: concept
title: Context Denoising
slug: context-denoising
date: 2026-04-20
updated: 2026-04-20
aliases: [上下文去噪]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Denoising** (上下文去噪) — the process of removing or down-weighting irrelevant contextual information so a model conditions on only the history that is useful for the current decision.

## Key Points

- HAConvDR uses pseudo relevance judgment to decide which historical turns should remain in the conversational retrieval context.
- The denoised reformulation keeps both the historical query and its gold passage when the turn is judged relevant to the current query.
- This design is motivated by shortcut history dependency, where a retriever over-attends to irrelevant past turns and ranks old gold passages too highly.
- The paper shows relevant historical turns are usually only a small fraction of all prior turns, peaking at roughly `20%`, which empirically justifies explicit denoising.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mo-2024-historyaware-2401-16659]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mo-2024-historyaware-2401-16659]].
