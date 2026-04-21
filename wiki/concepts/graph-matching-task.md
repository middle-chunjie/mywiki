---
type: concept
title: Graph Matching Task
slug: graph-matching-task
date: 2026-04-20
updated: 2026-04-20
aliases: [graph matching, 图匹配任务]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Matching Task** (图匹配任务) — a self-supervised instruction task in which a model must match graph tokens to the correct node texts, thereby learning which structural token corresponds to which semantic node identity.

## Key Points

- In GraphGPT, each central node is expanded into an `h`-hop sampled subgraph and represented as a sequence of graph tokens.
- The instruction presents a shuffled list of node texts and asks the LLM to reorder them according to the graph-token sequence.
- This creates supervision without downstream labels, making it suitable for the first stage of graph instruction tuning.
- The paper uses the task to teach the projector and LLM interface how graph tokens relate to textual semantics before task-specific fine-tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-graphgpt-2310-13023]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-graphgpt-2310-13023]].
