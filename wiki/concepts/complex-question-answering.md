---
type: concept
title: Complex Question Answering
slug: complex-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [Complex QA, 复杂问题问答]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Complex Question Answering** (复杂问题问答) — question answering on queries that require multi-hop composition, constraint satisfaction, or other non-trivial reasoning steps beyond a single fact lookup.

## Key Points

- The paper positions ComplexWebQuestions and WebQuestionsSP as benchmarks where success depends on recovering multi-hop evidence rather than a single relation.
- It argues that complex QA fails when the system retrieves locally relevant but globally inconsistent evidence paths.
- The proposed EPR method treats structural dependencies among topic entities, relations, and answers as first-class retrieval targets.
- Results show that explicit evidence-structure modeling helps much more on the harder CWQ benchmark than on the simpler WebQSP benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-enhancing-2402-02175]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-enhancing-2402-02175]].
