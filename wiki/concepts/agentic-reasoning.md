---
type: concept
title: Agentic Reasoning
slug: agentic-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [agentic reasoning, 代理式推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agentic Reasoning** (代理式推理) — a reasoning setup in which a primary language model dynamically delegates subproblems to external agents or tools during inference and integrates their outputs back into the reasoning chain.

## Key Points

- The paper instantiates agentic reasoning with three specialized tools: web search, coding, and Mind-Map memory.
- Tool invocation is controlled inside the reasoning trace through explicit tool-call tokens rather than a fixed pipeline.
- The framework emphasizes that tool quality and complementarity matter more than simply increasing the number of available tools.
- In the reported experiments, agentic reasoning substantially improves expert-level QA and deep-research generation over direct reasoning baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-agentic-2502-04644]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-agentic-2502-04644]].
