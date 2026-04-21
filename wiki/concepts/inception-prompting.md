---
type: concept
title: Inception Prompting
slug: inception-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [nested prompting, 嵌套提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Inception Prompting** (嵌套提示) — a prompt design strategy in which initial system prompts define roles, protocol, and task constraints so that agents can continue prompting each other autonomously afterward.

## Key Points

- CAMEL uses three inception prompts: a task-specifier prompt plus separate system prompts for the assistant and the user.
- The prompts encode role identity, output format, safety constraints, and task persistence before any dialogue begins.
- After initialization, the agents continue the task through autonomous multi-turn interaction without additional human prompting.
- The paper shows that small prompt changes materially affect termination behavior, role stability, and flake-message frequency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-camel-2303-17760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-camel-2303-17760]].
