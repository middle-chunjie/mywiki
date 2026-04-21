---
type: concept
title: Teacher-Student Framework
slug: teacher-student-framework
date: 2026-04-20
updated: 2026-04-20
aliases: [教师学生框架]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Teacher-Student Framework** (教师学生框架) — a two-model training design in which a teacher learns general structure or supervision signals and a student specializes on the downstream objective.

## Key Points

- ASGN splits representation learning and property optimization into separate teacher and student GNNs to reduce loss conflict.
- The teacher learns from supervised loss plus unsupervised node-level and graph-level objectives over labeled and unlabeled molecules.
- The student initializes from teacher weights and fine-tunes only on labeled property prediction.
- The student also generates pseudo labels for unlabeled molecules, so information flows back into the next teacher iteration.
- Ablation results show the full teacher-student design outperforms both teacher-only and student-only variants.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
