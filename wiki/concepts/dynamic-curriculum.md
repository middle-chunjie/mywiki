---
type: concept
title: Dynamic Curriculum
slug: dynamic-curriculum
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive curriculum]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Curriculum** — a staged training schedule that adaptively changes which auxiliary skills remain available based on the current policy's measured dependence on them.

## Key Points

- Skill0 uses a linearly decaying skill budget `M^(s)` so the number of allowed skill files decreases across stages until it reaches zero.
- Each skill file is evaluated by helpfulness `Delta_k`, computed as the performance difference with and without that file on a matched validation sub-task.
- The curriculum keeps only helpful skills by filtering non-positive `Delta_k`, ranking the rest, and selecting the top files under the current budget.
- Compared with static full-skill or random-selection schedules, the dynamic curriculum produces much stronger skill-free inference performance.
- The paper argues that gradual withdrawal avoids abrupt distribution shift in rendered context and stabilizes policy optimization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2026-skill-2604-02268]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2026-skill-2604-02268]].
