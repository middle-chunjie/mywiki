---
type: concept
title: Retrieval Ensemble
slug: retrieval-ensemble
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval ensembling, 检索集成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Ensemble** (检索集成) — an inference strategy that runs a language model separately on multiple retrieved contexts and combines their output distributions with retrieval-dependent weights.

## Key Points

- RePlug prepends each retrieved document independently instead of concatenating all documents into one long prompt.
- The final next-token distribution is a weighted average over `k` LM passes, with weights normalized from retriever similarity scores.
- This design lets the method incorporate more retrieved documents than a single prompt would fit within a fixed context window.
- The analysis shows that relevant-document ensembling helps, whereas random-document ensembling hurts performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-replug-2301-12652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-replug-2301-12652]].
