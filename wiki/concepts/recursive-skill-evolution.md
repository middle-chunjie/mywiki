---
type: concept
title: Recursive Skill Evolution
slug: recursive-skill-evolution
date: 2026-04-20
updated: 2026-04-20
aliases: [递归技能演化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Recursive Skill Evolution** (递归技能演化) — an RL-time update mechanism that expands or refines an agent's skill library by analyzing validation failures and feeding the resulting skills back into future policy optimization.

## Key Points

- SkillRL does not treat its skill memory as static; after validation epochs it inspects failed categories and updates SkillBank.
- Categories with success rate `Acc(C) < \delta` trigger diversity-aware sampling of failed trajectories for analysis.
- The teacher model identifies uncovered failure patterns, proposes new skills, and suggests refinements to ineffective existing skills.
- This creates a co-evolution loop in which stronger policies expose new edge cases that drive further skill growth.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
