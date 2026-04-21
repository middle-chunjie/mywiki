---
type: concept
title: Active Learning
slug: active-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [主动学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Active Learning** (主动学习) — a training strategy that iteratively selects unlabeled examples for annotation so a model can improve with fewer labeled instances.

## Key Points

- ASGN uses active learning after representation learning to choose new molecules whose labels should be acquired next.
- The selection criterion is diversity-oriented rather than uncertainty-oriented because the paper argues batch uncertainty sampling tends to return near-duplicate points.
- Molecules are compared in the teacher embedding space, so selection depends on learned molecular representations instead of handcrafted graph distances.
- The framework adds newly labeled molecules back into the supervised set and repeats teacher-student training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
