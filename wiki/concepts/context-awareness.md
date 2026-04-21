---
type: concept
title: Context Awareness
slug: context-awareness
date: 2026-04-20
updated: 2026-04-20
aliases: [context awareness, 上下文感知]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Awareness** (上下文感知) — a model's ability to condition its predictions on the relevant local evidence provided in the input context rather than on spurious associations or parametric defaults.

## Key Points

- The paper argues that truncation weakens context awareness because relevant grounding spans may fall outside the training sequence.
- Improvements on NQ-Swap and MemoTrap are used as evidence that preserving documents helps models follow contradictory or unusual context more faithfully.
- In summarization, packing also improves adherence to explicit length instructions, suggesting better use of prompt context.
- The concept is operationalized through downstream tasks where the answer should be derived from the provided passage rather than memorized knowledge.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-fewer-2404-10830]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-fewer-2404-10830]].
