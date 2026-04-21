---
type: concept
title: Node Evaluation
slug: node-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [tree node scoring, 节点评估]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Node Evaluation** (节点评估) — a scoring procedure that estimates the quality of tree nodes and uses those scores to retain promising branches while pruning weaker ones.

## Key Points

- SolutionRAG scores solution nodes by the average logits of a reliability suffix conditioned on their child comments.
- It scores comment nodes by the average logits of a helpfulness suffix conditioned on the old solution, the comment, and the refined child solutions.
- After scoring, only the top-`W` nodes are kept at each layer; the paper uses `W = 1`.
- The retained nodes produce substantially better solutions than pruned nodes in the paper's analysis figure.
- Node evaluation is presented as the mechanism that balances inference quality against tree-growth cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-deepsolution-2502-20730]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-deepsolution-2502-20730]].
