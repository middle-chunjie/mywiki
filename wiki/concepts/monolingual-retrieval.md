---
type: concept
title: Monolingual Retrieval
slug: monolingual-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [单语检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Monolingual Retrieval** (单语检索) — a retrieval setting in which the query and the indexed passage are written in the same language.

## Key Points

- [[thakur-2024-leveraging-2311-05800]] studies monolingual retrieval on MIRACL, covering `18` languages with language-specific Wikipedia corpora.
- For monolingual synthetic data generation, the paper uses `3` prompt exemplars per language instead of `5` to control prompt length and cost.
- The synthetic SWIM-X model reaches `46.4` average `nDCG@10` on MIRACL, outperforming zero-shot baselines but trailing supervised mContriever-X with hard negatives.
- The authors switch to language-unmixed fine-tuning for MIRACL because prior work and their setup suggest it is better aligned with monolingual retrieval training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[thakur-2024-leveraging-2311-05800]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[thakur-2024-leveraging-2311-05800]].
