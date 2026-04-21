---
type: concept
title: Coding Agent
slug: coding-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [code agent, coding tool agent, 代码代理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Coding Agent** (代码代理) — an external programming module that writes, executes, and summarizes code needed to support a reasoning task, instead of forcing the main reasoner to manage execution directly.

## Key Points

- The coding agent receives the user query together with Mind-Map-derived reasoning context.
- It is instructed to generate code, run it, and return a natural-language answer that can be stitched back into the reasoning trace.
- The paper uses Claude 3.5 Sonnet for code generation and Python 3.11 for execution.
- This modularization is intended to preserve long reasoning coherence by offloading execution-heavy subproblems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-agentic-2502-04644]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-agentic-2502-04644]].
