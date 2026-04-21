---
type: concept
title: Agent Generation
slug: agent-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [agent generation, 智能体生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agent Generation** (智能体生成) — the automatic creation of new agents or agent configurations from an initial seed agent to expand behavioral diversity and task coverage.

## Key Points

- EvoAgent frames agent generation as its central problem, arguing that manually specifying roles and prompts limits the scalability of multi-agent systems.
- New agents are produced by evolving role, skill, and prompt settings instead of hand-authoring fixed agent templates.
- The paper emphasizes that generated agents should be both high quality and meaningfully different from their parents, so diversity is an explicit objective.
- The resulting agents are used collaboratively rather than independently: each produces candidate outputs that are later integrated into an improved final answer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2024-evoagent-2406-14228]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2024-evoagent-2406-14228]].
