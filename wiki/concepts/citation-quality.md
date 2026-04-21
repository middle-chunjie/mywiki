---
type: concept
title: Citation Quality
slug: citation-quality
date: 2026-04-20
updated: 2026-04-20
aliases: [citation support quality, 引文质量]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Citation Quality** (引文质量) — the degree to which generated claims are fully supported by cited evidence and avoid unnecessary or irrelevant citations.

## Key Points

- ALCE defines citation quality with two metrics: statement-level citation recall and citation-level precision.
- A statement gets recall `1` only when it has at least one citation and the concatenated cited passages entail the statement.
- A citation is marked irrelevant when it does not support the statement on its own and removing it does not reduce support from the remaining citations.
- The metric intentionally allows redundant citations and does not require a minimal citation set, aligning better with how humans often cite.
- Reranking by automatic citation recall consistently improves citation quality on ASQA and ELI5 in the experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2023-enabling-2305-14627]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2023-enabling-2305-14627]].
