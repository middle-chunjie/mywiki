---
type: concept
title: Multiturn Collaboration
slug: multiturn-collaboration
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-turn collaboration, human-llm collaboration, 多轮协作]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multiturn Collaboration** (多轮协作) — an interaction setting where a user and model jointly refine goals and intermediate outputs across multiple dialogue turns to reach a higher-quality final outcome.

## Key Points

- The paper argues that many real tasks start from incomplete user intent, so collaboration quality must be judged over the whole conversation rather than any single response.
- CollabLLM treats document editing, coding assistance, and math help as multiturn collaboration problems where the model should clarify, suggest, and adapt.
- The objective explicitly includes user-centered factors such as efficiency and interactivity instead of task completion alone.
- Results show that collaborative behavior can improve both final-task metrics and the quality of the interaction process.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-collabllm-2502-00640]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-collabllm-2502-00640]].
