---
type: concept
title: Parameter token
slug: parameter-token
date: 2026-04-20
updated: 2026-04-20
aliases: [parameter token, 参数词元]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parameter token** (参数词元) — a learnable tokenized representation of model parameters that can serve as keys and values in attention-based computation.

## Key Points

- TokenFormer represents model parameters as learnable key-value token pairs instead of fixed weight matrices.
- Attention/output projections use parameter-token counts aligned with hidden size, while the FFN uses `4d` parameter-token pairs.
- Scaling from `124M` to `1.4B` parameters is performed by appending new parameter tokens rather than changing layer count or widening the token-state dimension.
- Zero-initialized new key parameter tokens leave the old output unchanged under the paper's Pattention formulation, which makes checkpoint reuse practical.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-tokenformer-2410-23168]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-tokenformer-2410-23168]].
