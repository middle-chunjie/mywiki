---
type: concept
title: Answer Aggregation
slug: answer-aggregation
date: 2026-04-20
updated: 2026-04-20
aliases: [answer voting, 答案聚合]
tags: [aggregation, decoding]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Answer Aggregation** (答案聚合) — the process of combining multiple candidate outputs into one final prediction using a deterministic scoring or voting rule.

## Key Points

- Self-consistency separates answer aggregation from path generation: first sample diverse outputs, then aggregate only over final answers.
- The paper compares unweighted sum, normalized weighted sum, unnormalized weighted sum, and weighted average for aggregation.
- Unweighted majority voting is nearly as strong as normalized weighted sum because normalized path probabilities are often similar across sampled outputs.
- Weighted average performs much worse, showing that the aggregation rule is not interchangeable.
- For evaluation, answers are task-dependently parsed from the final answer span rather than comparing entire reasoning traces.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfconsistency-2203-11171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfconsistency-2203-11171]].
