---
type: concept
title: Conversational Search
slug: conversational-search
date: 2026-04-20
updated: 2026-04-20
aliases: [Conversational IR, 对话搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Conversational Search** (对话搜索) — an information retrieval setting where a system must interpret and answer a user's evolving information need through multi-turn natural language interaction.

## Key Points

- The paper treats conversational search as passage retrieval conditioned on the full context `C_n = {q_1, r_1, ..., q_n}` rather than on the current query alone.
- It argues that real users can realize the same intent through many alternative conversational trajectories, making robustness to conversational variation a central requirement.
- ConvAug improves conversational search by generating controlled positive and negative conversational variants instead of relying only on observed dialogues.
- Later and more complex turns benefit most from augmentation, indicating that long interaction histories are especially challenging.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
