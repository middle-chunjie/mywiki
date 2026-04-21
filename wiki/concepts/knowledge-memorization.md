---
type: concept
title: Knowledge Memorization
slug: knowledge-memorization
date: 2026-04-20
updated: 2026-04-20
aliases: [KM]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Memorization** (知识记忆) — the ability of a model to faithfully recall factual knowledge it has previously absorbed from training data.

## Key Points

- KoLA operationalizes this level with three fact-probing tasks: high-frequency knowledge, low-frequency knowledge, and an evolving memorization test.
- The known-data tasks are reconstructed from Wikidata5M-linked Wikipedia facts, while the evolving task keeps `100` newly annotated triplets that cannot be inferred from prior corpora.
- In KoLA's analysis, memorization is a strong foundation for downstream understanding and reasoning abilities, and its ranking has Spearman correlation `0.79` with model size among unaligned models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-kola-2306-09296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-kola-2306-09296]].
