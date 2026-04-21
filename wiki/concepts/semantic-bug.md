---
type: concept
title: Semantic Bug
slug: semantic-bug
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic error, 语义错误]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Bug** (语义错误) — a defect that preserves syntactic well-formedness but causes the program's behavior to violate the intended specification.

## Key Points

- [[dinh-2023-large-2306-03438]] restricts its study to semantic bugs because syntax errors are easier to identify and less informative for code-model robustness.
- The synthetic benchmark mutates operators while preserving parseability, isolating semantic failure rather than compiler failure.
- Execution-based evaluation is necessary because semantic bugs may leave the generated code fluent and well-typed while still failing tests.
- The paper shows code LLMs are highly vulnerable to semantic disturbances in the prefix even when those disturbances are only a single-token change.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dinh-2023-large-2306-03438]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dinh-2023-large-2306-03438]].
