---
type: concept
title: Multi-turn Revision
slug: multi-turn-revision
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative revision, 多轮修订]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-turn Revision** (多轮修订) — an iterative correction scheme in which a model revises an earlier response after receiving explicit feedback from a prior attempt.

## Key Points

- CodeI/O++ extends CodeI/O with one extra revision turn after execution-based verification of the initial prediction.
- Roughly `50%` of first-turn predictions are already correct, and about `10%` of the incorrect responses can be fixed after one revision overall.
- A second extra revision turn yields very small additional benefit and can even regress performance, so the main system stops after one revision.
- The paper shows that this single-turn correction improves average benchmark performance over the no-revision version for several models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-codeio-2502-07316]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-codeio-2502-07316]].
