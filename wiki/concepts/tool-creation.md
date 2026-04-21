---
type: concept
title: Tool Creation
slug: tool-creation
date: 2026-04-20
updated: 2026-04-20
aliases: [automatic tool creation, dynamic tool creation, 工具创建]
tags: [agents, tool-creation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Creation** (工具创建) — the process by which an agent synthesizes or adapts executable tools so its action space can expand during task execution.

## Key Points

- The paper distinguishes tool creation from tool learning: the former builds or adapts tools, while the latter only improves tool selection or use.
- Earlier methods are described as limited because they create simple tools from scratch and usually cannot interact with the operating system.
- TOOLMAKER reframes tool creation as repository-grounded integration, where the agent installs dependencies and exposes an existing research codebase as an LLM-callable tool.
- The authors treat dynamic tool creation as a prerequisite for autonomous scientific workflows in domains with many specialized software packages.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[w-lflein-2025-llm-2502-11705]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[w-lflein-2025-llm-2502-11705]].
