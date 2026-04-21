---
type: concept
title: Meta Evaluation
slug: meta-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [元评估]
tags: [evaluation, benchmark]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Meta Evaluation** (元评估) — the evaluation of an automatic metric by measuring how well its outputs align with human judgments over model responses.

## Key Points

- RAGChecker validates its metrics on a meta-evaluation dataset of `280` response pairs sampled from `8` baseline RAG systems over `10` domains and `28` system pairs.
- Each instance is annotated by two annotators on correctness, completeness, and overall assessment using a five-level pairwise preference scale.
- The annotation process reaches `90.95%` agreement under a one-point tolerance, which supplies a practical ceiling for automatic metrics.
- RAGChecker outperforms BLEU, ROUGE, BERTScore, TruLens, ARES, RAGAS, and CRUD-RAG on correlation with human preferences.
- The paper frames meta-evaluation as necessary because raw metric scores across frameworks are not directly comparable, but relative agreement with human ranking is.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ru-2024-ragchecker-2408-08067]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ru-2024-ragchecker-2408-08067]].
