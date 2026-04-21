---
type: concept
title: Theory-of-Mind Tracking
slug: theory-of-mind-tracking
date: 2026-04-20
updated: 2026-04-20
aliases: [ToM tracking, 心智理论追踪]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Theory-of-Mind Tracking** (心智理论追踪) — a task formulation that requires tracking an agent's belief state about the world under partial observability as events unfold over time.

## Key Points

- In LONGPROC, ToM Tracking asks the model to track a target person's belief about an object's location across a sequence of story events.
- The required trace records four fields per step: agent location, object location, visibility, and the agent's current belief.
- Unlike simpler retrieval tasks, belief updates require deduction about whether the target person can observe an object movement.
- The task appears in all three difficulty levels and reaches average output lengths up to `7.9K` tokens in the `8K` split.
- GPT-4o error analysis shows `19/20` sampled `8K` failures come from incorrect long-range inferences about location changes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2025-longproc-2501-05414]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2025-longproc-2501-05414]].
