---
type: concept
title: Meta Plan
slug: meta-plan
date: 2026-04-20
updated: 2026-04-20
aliases: [meta plans]
tags: [agents, planning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Meta Plan** — a high-level natural-language strategy for solving a task that abstracts away environment-specific details while still guiding downstream action planning.

## Key Points

- In [[xiong-2025-mpo-2503-02682]], meta plans are generated from task instructions and expert trajectories, then manually filtered for correctness, abstraction, and formatting quality.
- The paper treats meta plans as portable explicit guidance that can be inserted directly into an agent prompt without retraining the downstream agent itself.
- MPO evaluates a meta plan by the downstream agent's empirical task success rate under repeated rollouts rather than by human preference labels.
- The paper shows that inserting the meta plan into the task instruction is more effective than injecting it into thoughts or observations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2025-mpo-2503-02682]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2025-mpo-2503-02682]].
