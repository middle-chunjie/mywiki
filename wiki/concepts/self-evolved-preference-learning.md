---
type: concept
title: Self-Evolved Preference Learning
slug: self-evolved-preference-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [自演化偏好学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Evolved Preference Learning** (自演化偏好学习) — a preference-optimization scheme in which the model iteratively generates its own positive-negative trajectory pairs and then applies preference learning to refine future behavior.

## Key Points

- Tool-Light's self-evolved preference learning is implemented with DPO rather than a separate reward model.
- The first round emphasizes reducing redundant tool use through short, low-entropy correct trajectories versus longer incorrect ones.
- Later rounds adapt positive and negative pair selection differently for easy and hard samples, making the criterion depend on the model's current competence.
- The approach is central to the paper's gains in efficiency, necessity, and average benchmark performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2026-effective-2509-23285]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2026-effective-2509-23285]].
