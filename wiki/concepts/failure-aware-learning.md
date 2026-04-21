---
type: concept
title: Failure-Aware Learning
slug: failure-aware-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [失败感知学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Failure-Aware Learning** (失败感知学习) — learning that explicitly preserves, analyzes, and abstracts failed attempts so they can inform future behavior instead of being discarded as noise.

## Key Points

- SkillRL keeps failed trajectories instead of dropping them after rollout collection.
- The teacher summarizes each failed trajectory into a short lesson covering failure point, mistaken reasoning, corrective action, and preventive principle.
- Validation-time failure analysis drives dynamic skill creation for categories whose success rate remains low.
- This design turns failure cases into structured supervision for both the external skill library and the RL policy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
