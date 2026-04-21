---
type: concept
title: Pseudo-Labeling
slug: pseudo-labeling
date: 2026-04-20
updated: 2026-04-20
aliases: [pseudo labeling, 伪标签]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pseudo-Labeling** (伪标签) — a semi-supervised technique that assigns model-generated labels to unlabeled examples so they can be reused as additional supervision.

## Key Points

- In ASGN, the student predicts properties for unlabeled molecules after fine-tuning on the labeled subset.
- Those predictions become pseudo labels that are injected into the next teacher-training round.
- Pseudo labels let the framework reuse unlabeled molecular graphs in the supervised property term rather than only in reconstruction or clustering losses.
- The paper treats this feedback loop as a way for the teacher to learn knowledge distilled from the student.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
