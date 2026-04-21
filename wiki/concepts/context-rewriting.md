---
type: concept
title: Context Rewriting
slug: context-rewriting
date: 2026-04-20
updated: 2026-04-20
aliases: [context regeneration, 上下文重写]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Rewriting** (上下文重写) — the practice of transforming an original prompt or context into a cleaned version that preserves relevant evidence while removing ambiguity, bias, or distractors.

## Key Points

- S2A operationalizes context rewriting as a first-pass generation step that maps the original input `x` to a revised context `x'`.
- The rewritten context explicitly preserves the question while discarding opinionated or irrelevant text that could distort the final answer.
- The paper contrasts this with response refinement methods, arguing that S2A improves the input context rather than post-editing the output.
- On math word problems, context rewriting removes distractor sentences before the solver performs chain-of-thought reasoning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weston-2023-system-2311-11829]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weston-2023-system-2311-11829]].
