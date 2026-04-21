---
type: concept
title: Premise Selection
slug: premise-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [premise retrieval, premise ranking, 前提选择]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Premise Selection** (前提选择) — the task of identifying which previously defined lemmas, theorems, or definitions from a formal library are most useful for proving the current goal.

## Key Points

- [[yang-nd-leandojo]] treats premise selection as a central bottleneck in Lean theorem proving because the full math library is far too large to pass directly into the tactic generator.
- LeanDojo records where premises are defined and used, turning premise selection into a supervised retrieval problem over traced Lean proofs.
- ReProver restricts retrieval to premises accessible to the current theorem, cutting the average candidate pool from about `128k` to `33,160`.
- The retriever improves over BM25 and over ablations that either search all premises or remove in-file hard negatives.
- The paper uses a harder `novel_premises` split to test whether premise selection generalizes when the needed premises were never used in training proofs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-nd-leandojo]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-nd-leandojo]].
