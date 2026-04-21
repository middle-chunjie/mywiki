---
type: concept
title: Hubness
slug: hubness
date: 2026-04-20
updated: 2026-04-20
aliases: [hub phenomenon]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hubness** — a high-dimensional retrieval phenomenon in which a small set of embeddings appears disproportionately often in nearest-neighbor lists, biasing retrieval toward generic or misleading items.

## Key Points

- The paper uses hubness as one of its semantic-space diagnostics for corpus characterization in adaptive routing.
- Hubness is quantified as the skewness of the `k`-occurrence distribution over nearest-neighbor lists.
- High hubness is interpreted as retrieval interference that reduces the reliability of pure vector retrieval.
- The Legal corpus is highlighted as a case where high hubness and low dispersion help explain why dense retrieval alone performs only moderately well.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
