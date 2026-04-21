---
type: concept
title: Critic-In-The-Loop
slug: critic-in-the-loop
date: 2026-04-20
updated: 2026-04-20
aliases: [critic in the loop, 评论者在环]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Critic-In-The-Loop** (评论者在环) — a control mechanism in which an additional critic agent or human evaluates candidate actions or responses and selects better branches during multi-agent task solving.

## Key Points

- CAMEL introduces a critic as an optional extension for improving controllability beyond the base assistant-user dialogue.
- The critic can either provide feedback or select among proposals from the role-playing agents.
- The paper describes the mechanism as tree-search-like decision making inspired by Monte Carlo tree search.
- This extension is motivated by cases where human preferences or stronger oversight are needed to steer autonomous cooperation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-camel-2303-17760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-camel-2303-17760]].
