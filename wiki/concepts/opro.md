---
type: concept
title: OPRO
slug: opro
date: 2026-04-20
updated: 2026-04-20
aliases: [Optimization by Prompting, Optimization by PROmpting]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**OPRO** — an optimization framework in which a language model iteratively proposes new candidate solutions from a meta-prompt containing prior solutions and their scores.

## Key Points

- The optimizer never receives gradients; it infers promising edits directly from the optimization trajectory shown in context.
- The method uses natural-language task descriptions, making the same loop applicable to continuous, discrete, and prompt-space optimization problems.
- OPRO samples up to `8` candidates per step to improve stability and exploration.
- In prompt optimization, the meta-prompt keeps the best `20` instructions and `3` task exemplars.
- The paper shows OPRO can improve prompts on GSM8K and BBH while also solving small linear regression and TSP instances.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-large-2309-03409]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-large-2309-03409]].
