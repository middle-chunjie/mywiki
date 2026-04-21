---
type: concept
title: Multi-Round Editing
slug: multi-round-editing
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative editing, 多轮编辑]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Round Editing** (多轮编辑) — an interactive editing setting in which a model and a user alternately modify the same artifact over several rounds, with later predictions conditioned on earlier accepted edits.

## Key Points

- [[unknown-nd-coeditorleveraging-2305-18584]] formulates prediction as `P(Delta u | Delta_k ... Delta_1, U)`, so each new suggestion conditions on all earlier contextual edits.
- The target region is allowed to overlap with earlier modifications, which makes repeated editing of the same function or block a first-class use case.
- The paper's evaluation simulates a user who accepts exactly correct predicted lines and otherwise performs the next ground-truth change manually before the next round.
- Multi-round interaction materially improves utility in the reported benchmark, raising changed-line automation from `28.5%` in single-round mode to `46.7%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-coeditorleveraging-2305-18584]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-coeditorleveraging-2305-18584]].
