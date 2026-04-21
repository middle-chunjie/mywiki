---
type: concept
title: Answer Completion
slug: answer-completion
date: 2026-04-20
updated: 2026-04-20
aliases: [completion, 答案补全]
tags: [reasoning, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Answer Completion** (答案补全) — supplying missing intermediate knowledge for a reasoning step after the model explicitly marks that sub-question as unsolved.

## Key Points

- SearChain uses `[Unsolved Query]` as an explicit signal that the LLM cannot currently answer a node in the Chain-of-Query.
- For such nodes, retrieval still extracts answer `g*` from the retrieved document `d_i*`, but unlike verification it feeds the answer back regardless of whether confidence exceeds `theta`.
- The returned answer and evidence let the LLM continue reasoning by generating a new Chain-of-Query rooted at the previously unsolved node.
- Completion decouples what the model already knows from what retrieval should supply, which is the paper's main mechanism for reducing harmful unnecessary intervention.
- Ablation shows completion materially contributes to final accuracy, for example reducing T-REx from `65.07` to `56.03` when removed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-searchinthechain-2304-14732]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-searchinthechain-2304-14732]].
