---
type: concept
title: Post-execution Evaluation
slug: post-execution-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [post-evaluation, grounded reward, 执行后评估]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Post-execution Evaluation** (执行后评估) — scoring the realized output of a tool call after execution so the planner can update action values using grounded evidence rather than hypothetical reasoning alone.

## Key Points

- ToolTree computes `r_post(s_t, a) = J(C_t, a, o_{t+1})` after each executed action using an LLM judge.
- This grounded score is propagated upward through the search tree to update `Q(s, a)` and future exploitation decisions.
- Post-evaluation also controls post-pruning by marking branches with low realized utility as non-expandable.
- The paper's ablation finds post-evaluation to be one of the most critical components, with removal causing the largest accuracy loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2026-tooltree-2603-12740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2026-tooltree-2603-12740]].
