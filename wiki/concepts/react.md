---
type: concept
title: ReAct
slug: react
date: 2026-04-20
updated: 2026-04-20
aliases: [reasoning and acting, thought-action-observation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**ReAct** — an agent framework that alternates explicit reasoning steps with tool actions and environment observations.

## Key Points

- WebSailor adopts ReAct as its base interaction protocol for web search and page visiting.
- Each trajectory consists of repeated thought, action, and observation tuples until the model emits a final answer.
- The paper's cold-start and RL stages both preserve the ReAct structure, including special tokens around thoughts, tool calls, tool responses, and answers.
- ReAct is central to the paper's claim that long-horizon web reasoning needs explicit action planning rather than one-shot answering.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-websailor-2507-02592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-websailor-2507-02592]].
