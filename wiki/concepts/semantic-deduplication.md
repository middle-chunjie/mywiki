---
type: concept
title: Semantic Deduplication
slug: semantic-deduplication
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic dedup, 语义去重]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semantic Deduplication** (语义去重) — the removal of near-duplicate examples based on semantic similarity rather than exact surface matching, typically to reduce redundancy and training pathologies.

## Key Points

- This paper applies semantic deduplication before context construction by using retrieval scores to identify near-duplicate documents in the pretraining corpus.
- The deduplication step is motivated by the risk that highly similar neighboring documents let the LM copy from prior context instead of learning useful cross-document reasoning.
- In the ablation, removing deduplication worsens perplexity from `7.3` to `8.3`, making it one of the paper's clearest design wins.
- The method is tightly coupled to related-document pretraining because retrieved nearest neighbors otherwise increase the chance of packing duplicates together.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2024-incontext-2310-10638]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2024-incontext-2310-10638]].
