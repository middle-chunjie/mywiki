---
type: concept
title: Capacity Gap
slug: capacity-gap
date: 2026-04-20
updated: 2026-04-20
aliases: [teacher-student capacity gap, 能力鸿沟]
tags: [distillation, optimization, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Capacity Gap** (能力鸿沟) — the regime in distillation where making the teacher more capable eventually worsens the student because the student cannot faithfully model the teacher's signal.

## Key Points

- The paper observes a U-shaped dependence of student loss on teacher strength: stronger teachers help at first and then hurt.
- This effect appears in fixed-`M` teacher/student sweeps where student cross-entropy eventually rises as teacher size increases.
- The authors argue the gap is about relative learning capacity, not merely a raw size comparison between teacher and student.
- Appendix analyses use kernel regression and synthetic MLPs as controlled demonstrations of the same phenomenon.
- Calibration diagnostics are used to probe what aspects of strong-teacher outputs are difficult for students to imitate.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[busbridge-2025-distillation-2502-08606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[busbridge-2025-distillation-2502-08606]].
