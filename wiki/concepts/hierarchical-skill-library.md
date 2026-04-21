---
type: concept
title: Hierarchical Skill Library
slug: hierarchical-skill-library
date: 2026-04-20
updated: 2026-04-20
aliases: [SkillBank, 分层技能库]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Skill Library** (分层技能库) — a structured memory representation that separates reusable agent skills into global guidance and task-specific heuristics for efficient retrieval and application.

## Key Points

- SkillRL organizes SkillBank into general skills `\mathcal{S}_g` and task-specific skills `\mathcal{S}_k` indexed by task category.
- General skills are always provided to the policy as foundational guidance, while task-specific skills are selectively retrieved.
- Each stored skill includes a name, a principle, and `when_to_apply` conditions so the agent receives actionable rather than narrative memory.
- The hierarchy supports both transfer across tasks and specialization within categories such as ALFWorld subtasks or search-style QA tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
