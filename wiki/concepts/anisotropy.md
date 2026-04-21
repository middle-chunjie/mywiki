---
type: concept
title: Anisotropy
slug: anisotropy
date: 2026-04-20
updated: 2026-04-20
aliases: [各向异性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Anisotropy** (各向异性) — a geometric pathology of embedding spaces in which representations concentrate in a narrow cone and fail to use directions in the vector space evenly.

## Key Points

- The paper connects poor sentence embedding expressiveness in pre-trained encoders to anisotropic representation geometry.
- SimCSE argues that contrastive learning alleviates anisotropy by pushing negatives apart and flattening the singular-value spectrum of the embedding matrix.
- This view is linked to the uniformity metric: a more isotropic embedding space is also a more uniformly distributed one on the hypersphere.
- SimCSE positions contrastive training as an alternative to post-processing methods such as whitening or principal-component removal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2022-simcse-2104-08821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2022-simcse-2104-08821]].
