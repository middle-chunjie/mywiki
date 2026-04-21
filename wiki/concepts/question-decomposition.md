---
type: concept
title: Question Decomposition
slug: question-decomposition
date: 2026-04-20
updated: 2026-04-20
aliases: [question decomposition, problem decomposition, task decomposition, 问题分解]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Question Decomposition** (问题分解) — the process of rewriting a complex question into simpler sub-questions whose answers can be solved separately and then aggregated.

## Key Points

- RA-ISF triggers question decomposition only after the model judges that it cannot answer from internal knowledge and that retrieved passages are all irrelevant.
- The decomposition model `\mathcal{M}_{decom}` outputs sub-questions `Q_sub = {q_1, ..., q_n}`, and RA-ISF recursively solves them before synthesizing a final answer.
- The framework caps recursive decomposition with `D_th = 3`; if a branch exceeds the threshold, the returned answer is `"unknown"`.
- The paper argues decomposition is especially useful for knowledge-intensive and multi-hop questions where the model understands parts of the problem but not the full composition.
- Removing the question decomposition module drops GPT-3.5 HotpotQA exact match from `46.5` to `34.9`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-raisf-2403-06840]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-raisf-2403-06840]].
