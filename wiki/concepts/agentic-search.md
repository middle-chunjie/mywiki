---
type: concept
title: Agentic Search
slug: agentic-search
date: 2026-04-20
updated: 2026-04-20
aliases: [search agent, 代理式搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agentic Search** (代理式搜索) — a search setting in which a language model explicitly plans, issues tool calls, reads retrieved evidence, and iteratively updates its reasoning state before answering.

## Key Points

- InfoSeeker is trained as an agentic search model rather than as a single-shot retrieval-augmented generator.
- Each reasoning turn begins with explicit planning, then issues one or more search actions before integrating evidence into the working context.
- The workflow is designed to improve both retrieval coverage and reasoning control on difficult multi-step information-seeking tasks.
- The paper evaluates agentic-search behavior on both classic QA benchmarks and BrowseComp-Plus, where tool use and iterative evidence gathering matter.
- Training data from InfoSeek is intended to transfer search-centric behavior better than NQ and HotpotQA alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2025-open-2509-00375]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2025-open-2509-00375]].
