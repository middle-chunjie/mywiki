---
type: concept
title: Code as Actions
slug: code-as-actions
date: 2026-04-20
updated: 2026-04-20
aliases: [code action, executable code action, 代码即动作]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code as Actions** (代码即动作) — an agent action design in which the model emits executable program fragments, rather than textual commands or JSON schemas, to operate on an environment.

## Key Points

- [[wang-2024-executable-2402-01030]] proposes Python code as the default action format for LLM agents interacting with tools and environments.
- The action can directly express control flow such as loops and conditionals, instead of forcing every behavior into a fixed tool schema.
- Execution results and error traces become structured feedback that the agent can use in later turns.
- The paper argues this format is a better fit for LLM pretraining because models already consume large amounts of code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-executable-2402-01030]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-executable-2402-01030]].
