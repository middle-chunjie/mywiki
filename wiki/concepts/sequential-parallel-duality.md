---
type: concept
title: Sequential-Parallel Duality
slug: sequential-parallel-duality
date: 2026-04-20
updated: 2026-04-20
aliases: [Sequential Parallel Duality, Layer Arrangement Duality, 串并联对偶性]
tags: [neural-architecture, transformer, theory]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sequential-Parallel Duality** (串并联对偶性) — the theoretical property of [[hyper-connections]] that the same framework can represent both standard sequential layer arrangements (where each layer processes the previous layer's output) and parallel arrangements (where layers process the same input independently), and that learnable HC matrices can implement any soft mixture between these extremes.

## Key Points

- Standard sequential residual networks are reproduced when the HC matrix for each layer takes the form of Eq. 17 in the paper (depth-connection degenerates to a residual connection).
- [[parallel-transformer-block]] arrangements emerge when adjacent HC matrices take the complementary form of Eq. 18, causing two consecutive layers to effectively share input.
- [[hyper-connections]] thus define a continuous space over possible layer-connection topologies, with sequential and parallel as degenerate boundary cases.
- Dynamic Hyper-Connections (DHC) extend this further by allowing the arrangement to vary per input token, producing a soft and input-conditioned mixture of sequential and parallel computation.
- Empirical visualization of OLMo-1B-DHC×4 confirms that the model learns a hybrid: a global Λ-shaped decay (Post-Norm-like) combined with selective early-layer access (Pre-Norm-like) and local PTB patterns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]] as a theoretical lens on what hyper-connections can represent.
