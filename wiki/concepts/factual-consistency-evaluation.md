---
type: concept
title: Factual Consistency Evaluation
slug: factual-consistency-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [FCE, factuality evaluation, 事实一致性评估]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Factual Consistency Evaluation** (事实一致性评估) — the task of determining whether generated text is fully supported by a provided source context or reference evidence.

## Key Points

- Face4RAG reframes FCE for RAG as a benchmark problem that should be evaluated independently of any single generator's error distribution.
- The paper distinguishes factual inconsistency into hallucination, knowledge error, and logical fallacy rather than treating unsupportedness as a single undifferentiated failure mode.
- L-Face4RAG performs FCE at the segment level and aggregates to the answer level only when every segment passes both fact and logic checks.
- The reported gains show that logic-aware evaluation materially improves FCE accuracy beyond prior decomposition-based baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-facerag-2407-01080]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-facerag-2407-01080]].
