---
type: concept
title: Tool Budget
slug: tool-budget
date: 2026-04-20
updated: 2026-04-20
aliases: [tool-call budget, 工具预算]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Budget** (工具预算) — the maximum number of tool calls an agent is allowed to make during a trajectory.

## Key Points

- [[yen-2025-lost-2510-18939]] formalizes the tool budget as `T`, with each tool call corresponding to one interaction turn.
- Existing frameworks fail either by exhausting the budget directly or by filling context so quickly that the budget becomes unusable.
- Search-o1 is reported to fail mainly by exceeding tool budgets, while ReAct often fails earlier from context overflow.
- Slim is designed to stretch effective budget usage by making each search and browse step cheaper and more selective.
- The paper evaluates behavior across a wide range of budgets, up to `T = 150`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2025-lost-2510-18939]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2025-lost-2510-18939]].
