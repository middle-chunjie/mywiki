---
type: concept
title: IterativeRAG
slug: iterative-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [IterativeRAG]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**IterativeRAG** — a retrieval-augmented generation paradigm that alternates retrieval, answer generation, and self-evaluation across multiple rounds to refine evidence and sub-questions.

## Key Points

- The implementation starts with a direct answer attempt, then retrieves additional context only if a self-evaluator judges the answer insufficient.
- The loop keeps accumulated evidence under an `8000`-token budget and stops when the answer is sufficient, no new sub-question is produced, a loop is detected, or `T = 3` iterations are reached.
- In this benchmark IterativeRAG can be highly cost-efficient, especially on Medical where it uses roughly `7k` average context tokens in the main table and `13.27M` total tokens in the appendix accounting.
- The paper also finds that IterativeRAG often underperforms expectations when initial retrieval is poor, because later iterations amplify rather than fix early errors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
