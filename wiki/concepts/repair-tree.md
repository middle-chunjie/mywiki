---
type: concept
title: Repair Tree
slug: repair-tree
date: 2026-04-20
updated: 2026-04-20
aliases: [repair search tree, 修复树]
tags: [search, debugging, evaluation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Repair Tree** (修复树) — the branching structure formed by an initial specification, sampled candidate programs, generated feedback, and downstream repair attempts.

## Key Points

- [[unknown-nd-selfrepair-2306-09896]] defines the repair tree `T` as the basic search object for evaluating self-repair.
- The tree starts at a specification, branches into initial programs, then branches again into feedback strings and repair candidates.
- Its effective sampling budget is measured by the number of programs in the tree, such as `|programs(T)| = n_p + n_p n_fr` in the joint feedback-repair setting.
- The paper bootstraps many subtrees from one large frozen repair tree per task to estimate pass rates across many hyperparameter settings efficiently.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-selfrepair-2306-09896]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-selfrepair-2306-09896]].
