---
type: concept
title: Pairwise Ranking Loss
slug: pairwise-ranking-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [pairwise preference loss, ranking loss]
tags: [training, ranking]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pairwise Ranking Loss** (成对排序损失) — a training objective that encourages a model to score a preferred candidate above a less preferred alternative.

## Key Points

- CR-Planner trains each critic on chosen-versus-rejected sibling observations collected from MCTS trajectories.
- The loss is applied separately for rationale, query, document, and sub-goal critics.
- This lets the critics learn relative action quality without generating full solutions themselves.
- The training setup follows the paper's ranking-based view that action evaluation is a preference estimation problem.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-can-2410-01428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-can-2410-01428]].
