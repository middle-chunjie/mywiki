---
type: concept
title: Tool Learning
slug: tool-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [tool use learning, 工具学习]
tags: [llm, agents, synthetic-data]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Learning** (工具学习) — the process of training language models to understand tool schemas, decide when tools are relevant, and compose valid calls that help complete user tasks.

## Key Points

- ToolACE frames tool learning as a data-generation problem requiring diversity, complexity control, and accuracy guarantees rather than only access to public APIs.
- The paper's Tool Self-evolution Synthesis module expands tool-learning coverage to `26,507` APIs across `390` domains, including nested parameter structures.
- Self-Guided Dialog Generation uses user, assistant, and tool agents to synthesize clarifications, parallel calls, dependent calls, and tool-abstention behavior.
- Dual-layer verification improves tool-learning data quality by combining rule-based executability checks with model-based hallucination and consistency checks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-toolace-2409-00920]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-toolace-2409-00920]].
