---
type: concept
title: Siamese Network
slug: siamese-network
date: 2026-04-20
updated: 2026-04-20
aliases: [siamese encoder, 孪生网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Siamese Network** (孪生网络) — an architecture with parameter-sharing subnetworks that encode multiple inputs into a comparable representation space for similarity or relation prediction.

## Key Points

- The paper uses a siamese CodeBERT encoder so queries and code are encoded by identical shared-weight transformers.
- Query and code representations come from pooled `[CLS]` outputs and are combined with difference and product features before classification.
- This design supports both binary code question answering and ranked code retrieval under a common matching function.
- CoCLR extends the siamese setup with additional losses over in-batch negatives and rewritten positive queries.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2021-cosqa-2105-13239]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2021-cosqa-2105-13239]].
