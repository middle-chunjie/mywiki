---
type: concept
title: Graph-of-Thought
slug: graph-of-thought
date: 2026-04-20
updated: 2026-04-20
aliases: [GoT, graph of thought]
tags: [llm, reasoning, search]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Graph-of-Thought** (思维图) — a reasoning framework that allows intermediate thought branches to merge, refine, and form graph-structured inference traces rather than a strict chain or tree.

## Key Points

- The paper positions GoT as more topologically flexible than CoT or ToT because it can merge thoughts during intermediate search.
- In the reported baselines, GoT still relies on the LLM to generate and evaluate thought candidates, so flexibility comes with high inference cost.
- For multi-solution settings, GoT is instructed to keep multiple thoughts instead of only the top-1 merge, but its MultiAcc remains far below XoT.
- XoT adopts the flexibility goal of GoT while replacing expensive LLM-centered search with MCTS guided by a lightweight policy/value model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-everything-2311-04254]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-everything-2311-04254]].
