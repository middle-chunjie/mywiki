---
type: concept
title: Residual Stream
slug: residual-stream
date: 2026-04-20
updated: 2026-04-20
aliases: [残差流]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Residual Stream** (残差流) — the hidden pathway that carries token representations across layers through skip connections, independent of the heavy layer transformation `F`.

## Key Points

- The paper expands the residual stream from `C` to `n x C`, creating `n` parallel residual streams that can exchange information.
- Three mappings govern the stream: `H_l^pre` reads from the stream into the block, `H_l^post` writes block outputs back, and `H_l^res` mixes information inside the stream itself.
- This widened stream increases topological complexity without changing the FLOPs of the core block computation, making it a new scaling axis beyond model size and data size.
- The residual mapping `H_l^res` is identified as the most important HC component in the paper's ablation, contributing the largest absolute loss improvement.
- The wider stream also creates the main systems burden, increasing memory traffic, activation storage, and pipeline communication unless kernels and recomputation are optimized.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2026-mhc-2512-24880]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2026-mhc-2512-24880]].
