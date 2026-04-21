---
type: concept
title: Coding Style Hiding Attack
slug: coding-style-hiding-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [Style Hiding Attack, 代码风格隐藏攻击]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Coding Style Hiding Attack** (代码风格隐藏攻击) — an untargeted authorship attack that alters a program's style so the victim model no longer attributes the code to its true author.

## Key Points

- The attacker evaluates alternative target authors and chooses the one expected to yield the highest misattribution probability.
- The transformation process reuses the imitation-attack machinery but optimizes for any incorrect attribution rather than a specific author.
- Untargeted hiding is empirically easier than targeted imitation, with average success rates much higher on baseline models.
- The paper shows that purposeful hiding outperforms random replacement by `12.7%` average attack success rate.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-ropgen-2202-06043]].
