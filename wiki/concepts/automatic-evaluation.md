---
type: concept
title: Automatic Evaluation
slug: automatic-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [automatic metrics, 自动评测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Automatic Evaluation** (自动评测) — the use of programmatic metrics rather than human annotators to score model outputs at scale and with high reproducibility.

## Key Points

- ALCE combines three automatic dimensions: fluency, correctness, and citation quality, instead of relying on a single score.
- Correctness is dataset-specific: EM recall for ASQA, recall-5 for QAMPARI, and NLI-based claim recall for ELI5.
- Citation quality is decomposed into recall and precision so systems cannot game the benchmark by citing many irrelevant passages.
- The paper validates the automatic metrics with human evaluation and reports strong agreement, including citation-recall kappa `0.698`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2023-enabling-2305-14627]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2023-enabling-2305-14627]].
