---
type: concept
title: Text-to-3D Retrieval
slug: text-to-3d-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [文本到三维检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Text-to-3D Retrieval** (文本到三维检索) — retrieving 3D objects by ranking candidate shapes according to similarity with a natural-language query.

## Key Points

- MixCon3D uses cosine similarity between text embeddings and 3D shape embeddings from the ensembled dataset as the retrieval score.
- The paper evaluates retrieval to test whether the learned 3D features stay aligned with the CLIP text embedding space.
- Qualitative examples indicate better fine-grained retrieval than OpenShape, including more accurate matches for detailed object descriptions.
- The retrieval results support the claim that joint image-3D alignment improves semantic indexing of 3D shapes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
