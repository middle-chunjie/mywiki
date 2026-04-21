---
type: concept
title: Centroid Interaction
slug: centroid-interaction
date: 2026-04-20
updated: 2026-04-20
aliases: [centroid-only interaction]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Centroid Interaction** — an approximate late-interaction scoring method that replaces each token embedding with its nearest centroid and performs MaxSim-style ranking over centroid scores instead of reconstructed vectors.

## Key Points

- PLAID computes query-centroid similarities once as `S_{c,q} = C \cdot Q^T` and reuses those scores across all candidate passages.
- Each candidate passage is treated as a bag of centroid IDs, allowing cheap approximate passage scores before residual decompression.
- The paper reports that centroid-only retrieval recovers `99%+` of the top-`k` passages returned by vanilla ColBERTv2 within roughly `10k` candidates.
- Centroid interaction is the main algorithmic reason PLAID can avoid exhaustive full-vector scoring for most passages.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[santhanam-2022-plaid-2205-09707]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[santhanam-2022-plaid-2205-09707]].
