---
type: concept
title: Message Passing
slug: message-passing
date: 2026-04-20
updated: 2026-04-20
aliases: [message passing, 消息传递]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Message Passing** (消息传递) - a graph computation pattern in which node representations are updated by aggregating information from neighboring nodes and edges.

## Key Points

- RAGraph uses standard GNN message passing as the mechanism for injecting retrieved knowledge rather than adding a separate reader model.
- The paper propagates both hidden embeddings `h` and task-specific output vectors `o`, not just latent node features.
- Toy-graph intra propagation aggregates neighbor information to each toy graph's master node before cross-graph transfer.
- Query-toy inter propagation then connects retrieved master nodes to the query center node so the query graph absorbs external evidence.
- The propagation can be parameter-free for efficiency or learnable for prompt tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragraph-2410-23855]].
