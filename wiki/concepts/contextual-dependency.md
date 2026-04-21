---
type: concept
title: Contextual Dependency
slug: contextual-dependency
date: 2026-04-20
updated: 2026-04-20
aliases: [context dependency, 上下文依赖]
tags: [code-generation, program-analysis, context]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contextual dependency** (上下文依赖) — the set of code elements outside the current function that the function must access to compile or run correctly.

## Key Points

- CoderEval treats contextual dependency as the main difference between realistic repository code generation and standalone benchmark tasks.
- The paper classifies dependencies by source, including standard libraries, public libraries, class-level context, file-level context, and project-level context.
- It further decomposes contextual dependency into `TypeReference`, `APIInvocation`, and `VarReference` during analysis.
- The benchmark stores both `oracle_context` and `all_context`, separating actually used context from all accessible context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-codereval-2302-00288]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-codereval-2302-00288]].
