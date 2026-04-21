---
type: concept
title: Knowledge Consolidation
slug: knowledge-consolidation
date: 2026-04-20
updated: 2026-04-20
aliases: [evidence consolidation, 知识整合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Consolidation** (知识整合) — the process of regrouping multiple evidence pieces into consistent, conflicting, and irrelevant subsets so that downstream answering can operate on refined information rather than raw passages.

## Key Points

- Astute RAG merges retrieved and generated passages into a shared evidence pool before consolidation.
- The consolidation prompt asks the LLM to identify agreement across passages, isolate conflicts, and filter irrelevant content.
- Consolidation can be iterated `t` times, although the paper uses `t = 1` by default for efficiency.
- The regrouped evidence retains source attribution, which lets the model compare reliability across internal and external evidence clusters.
- The paper reports intermediate-step accuracy of `98.2%` for knowledge consolidation under LLM-as-a-judge evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-astute-2410-07176]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-astute-2410-07176]].
