---
type: concept
title: Word Mover's Distance
slug: word-mover-distance
date: 2026-04-20
updated: 2026-04-20
aliases: [WMD, Word Movers Distance]
tags: [text-similarity, word-embedding, nlp, adversarial]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Word Mover's Distance** (词移距离) — A text similarity metric that measures the minimum total distance that the embedded words of one document must travel to reach the embedded words of another document, formalized as a minimum-cost flow problem over word embeddings.

## Key Points

- Proposed by Kusner et al. (2015) as an instance of the Earth Mover's Distance applied to word embedding spaces; minimizes `Σ_{i,j} T_{ij} ||e_i - e_j||_2` subject to flow constraints on the normalized bag-of-words distributions.
- Requires pre-trained word embeddings (e.g., Word2Vec); the metric is symmetric and scales with vocabulary and document length.
- In textual adversarial attack literature, WMD is used as a perturbation constraint to ensure that word substitution adversaries stay semantically close to the original text, avoiding grossly semantics-altering replacements.
- Computationally expensive (solving a linear program per pair); approximate variants exist for efficiency.
- Provides a semantically grounded distance that goes beyond surface-level edit distance by incorporating embedding geometry.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2019-adversarial-1901-06796]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2019-adversarial-1901-06796]].
