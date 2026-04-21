---
type: concept
title: User Simulator
slug: user-simulator
date: 2026-04-20
updated: 2026-04-20
aliases: [user simulation, 用户模拟器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**User Simulator** (用户模拟器) — a model that generates plausible user responses conditioned on conversation history and an implicit goal, enabling scalable training or evaluation without real users in the loop.

## Key Points

- In CollabLLM, the user simulator is the engine used to sample future conversation turns for MR estimation.
- The simulator is prompted to preserve the user's language style and to behave like a real user with incomplete knowledge or evolving needs.
- Experimental sections use GPT-4o-mini as the simulator across document, code, and math tasks.
- The appendix notes that weaker open models often fail to stay in the user role, making simulator quality a central bottleneck.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-collabllm-2502-00640]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-collabllm-2502-00640]].
