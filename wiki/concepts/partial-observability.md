---
type: concept
title: Partial Observability
slug: partial-observability
date: 2026-04-20
updated: 2026-04-20
aliases: [部分可观测性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Partial Observability** (部分可观测性) — a setting where the agent's current input does not fully specify the latent environment state needed for optimal action selection.

## Key Points

- The paper argues that a finite history window is generally not a sufficient statistic of the latent state in language-based environments.
- Partial observability makes credit assignment increasingly difficult as task horizon grows.
- D2Skill treats retrieved task and step skills as auxiliary information that compensates for missing state information in the prompt.
- The motivation for step-level skills is especially tied to correcting fine-grained errors caused by incomplete local context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
