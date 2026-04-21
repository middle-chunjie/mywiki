---
type: concept
title: Tool-Augmented Agent
slug: tool-augmented-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [工具增强智能体, tool-using agent]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tool-Augmented Agent** (工具增强智能体) — an AI agent that extends language-model reasoning with external tools such as search APIs, notebooks, or execution environments.

## Key Points

- AstaBench evaluates agents with standardized tool access rather than text-only prompting.
- The Asta Environment exposes both scientific-corpus retrieval tools and a stateful computational notebook.
- Tool access is decoupled from agent implementation so benchmark results better isolate agent capability from privileged infrastructure.
- The same benchmark suite supports both native tool-calling agents and code-writing agents such as ReAct-style and CodeAct-style systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
