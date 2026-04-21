---
type: concept
title: Answer Ignored
slug: answer-ignored
date: 2026-04-20
updated: 2026-04-20
aliases: [groundtruth ignored, ignored answer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Answer Ignored** — a failure mode where an agent encounters the correct answer during search but does not use it in the final response.

## Key Points

- [[yen-2025-lost-2510-18939]] defines answer ignored as a synthesis-stage failure rather than a retrieval failure.
- The paper detects it by checking whether any tool response contains the ground-truth answer, using an LLM judge for fuzzy matching.
- Slim still exhibits a high answer-ignored rate of `30.7` in the matched-budget analysis.
- Search-o1 also shows a large answer-ignored rate (`26.0`), suggesting that longer trajectories can increase evidence encounter without improving final selection.
- The paper identifies answer selection from long trajectories as an open improvement direction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2025-lost-2510-18939]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2025-lost-2510-18939]].
