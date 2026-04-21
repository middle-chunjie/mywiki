---
type: concept
title: Consensus Filtering
slug: consensus-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [consensus filter, 共识过滤]
tags: [data-synthesis, llm, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Consensus Filtering** (共识过滤) — a data-quality technique that retains training samples only when two or more independently derived annotation signals agree on the label, discarding instances where they disagree.

## Key Points

- [[zhang-2025-lessons-2501-07301]] applies consensus filtering to PRM training data by retaining samples only when both Monte Carlo estimation and LLM-as-a-judge agree on the error-step location in the reasoning trace.
- Approximately `40%` of the original 860k samples survive filtering; the retained set is denser in high-quality, unambiguous step-level annotations.
- The filtered dataset reaches comparable PROCESSBENCH performance to the full LLM-as-a-judge dataset at `40%` of the annotation cost.
- The mechanism compensates for the fundamental noise in MC estimation (which can reward correct answers derived from incorrect steps) by cross-checking with a direct critic model.
- Combined with hard labels, consensus filtering yields the best-performing PRMs in both BoN and step-level evaluation, demonstrating that data quality matters more than scale for process supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2025-lessons-2501-07301]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2025-lessons-2501-07301]].
