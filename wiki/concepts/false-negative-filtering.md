---
type: concept
title: False-Negative Filtering
slug: false-negative-filtering
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 假负例过滤
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**False-Negative Filtering** (假负例过滤) — a contrastive training strategy that removes likely positive or near-positive examples from the negative set to prevent the model from learning contradictory supervision.

## Key Points

- The paper defines a surrogate equivalence class `S(q, d)` using a pretrained embedding model and excludes members of `S(q, d)` from the contrastive denominator.
- This step is especially important because the proposed clustering-based batching deliberately puts semantically similar examples together, which otherwise raises false-negative rates.
- The authors report that filtering is critical for performance and that smaller, harder clusters help most clearly when filtering is enabled.
- Their motivation is empirical: nearest-neighbor inspection in large retrieval corpora shows many top neighbors are plausible positives for the same query.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2024-contextual-2410-02525]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2024-contextual-2410-02525]].
