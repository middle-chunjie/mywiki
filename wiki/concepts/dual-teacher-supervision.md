---
type: concept
title: Dual-Teacher Supervision
slug: dual-teacher-supervision
date: 2026-04-20
updated: 2026-04-20
aliases: [dual supervision, 双教师监督]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dual-Teacher Supervision** (双教师监督) — a distillation setup in which a student retriever is trained simultaneously from two complementary teachers that provide different but compatible supervision signals.

## Key Points

- The paper combines a pairwise `BERT_CAT` teacher with a ColBERT in-batch teacher to train a `BERT_DOT` student retriever.
- `BERT_CAT` supplies stronger passage-pair margins, while ColBERT supplies efficient in-batch comparisons that scale linearly with batch size.
- The total objective is `L_DS = L_Pair + \alpha L_InB` with `\alpha = 0.75`.
- Empirically, dual supervision improves both ranking quality and recall over using only pairwise or only in-batch teacher signals.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hofst-tter-2021-efficiently-2104-06967]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hofst-tter-2021-efficiently-2104-06967]].
