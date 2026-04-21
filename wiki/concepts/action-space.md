---
type: concept
title: Action Space
slug: action-space
date: 2026-04-20
updated: 2026-04-20
aliases: [agent action space, 动作空间]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Action Space** (动作空间) — the set of actions an agent is allowed to produce and execute when interacting with users, tools, or an environment.

## Key Points

- [[wang-2024-executable-2402-01030]] treats action-space design as a central bottleneck for LLM agents rather than a mere output-format choice.
- The paper argues that text and JSON formats constrain the action space to predefined tools and awkwardly support composition.
- CodeAct expands the action space by allowing agents to call existing Python libraries and compose multi-step procedures within one action.
- The paper's broader claim is that a richer action space especially matters on complex multi-tool tasks such as M3ToolEval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-executable-2402-01030]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-executable-2402-01030]].
