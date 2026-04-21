---
type: concept
title: Graph Prompt Learning
slug: graph-prompt-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [graph prompt learning, graph prompting, 图提示学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Prompt Learning** (图提示学习) - a transfer-learning approach for graph models that adapts a pre-trained GNN to downstream tasks by injecting task-relevant structures, nodes, or parameters as prompts.

## Key Points

- RAGraph frames retrieved toy graphs as prompt-like external context for a pre-trained GNN backbone.
- The paper contrasts its design with prior graph prompting baselines such as GraphPrompt and GraphPro, arguing retrieval adds task-relevant evidence rather than only prompt parameters.
- The prompting mechanism is designed to be plug-and-play and usable across node, graph, and link tasks.
- Noise-based graph prompt tuning augments prompt training with irrelevant toy-graph noise to improve robustness.
- The method aims to retain backbone reuse while reducing the need for task-specific re-training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragraph-2410-23855]].
