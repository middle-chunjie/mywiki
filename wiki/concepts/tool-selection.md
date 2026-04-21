---
type: concept
title: Tool Selection
slug: tool-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [API selection, tool choice, 工具选择]
tags: [agents, retrieval, tool-use]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Selection** (工具选择) — the process of choosing the most appropriate external tool or API for a user query or intermediate reasoning step.

## Key Points

- ToolWeaver reframes tool selection as constrained next-token generation over hierarchical code sequences rather than ranking isolated tool IDs.
- The paper evaluates tool selection with `NDCG@1`, `NDCG@3`, and `NDCG@5` across single-tool and multi-tool ToolBench tasks.
- Collaborative regularization is most helpful on the hardest I3 setting, indicating that tool selection quality depends on modeling inter-tool relationships.
- Error analysis shows wrong-tool prediction remains the dominant failure type in complex trajectories.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
