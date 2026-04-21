---
type: concept
title: Relation-gated Convolution
slug: relation-gated-convolution
date: 2026-04-20
updated: 2026-04-20
aliases: [关系门控卷积, RGC]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Relation-gated Convolution** (关系门控卷积) — a knowledge-graph convolution that uses relation embeddings as gates to regulate how much neighbor information enters each entity representation.

## Key Points

- RHGN computes `e_i^(k+1) = tanh(Σ W_e^k (e_j^k ⊗ sigmoid(r_ij^k)))`, so relations modulate messages rather than being naively fused with entities.
- This design is motivated by the claim that directly incorporating relations into entity embeddings introduces noise and encourages over-smoothing.
- Relation embeddings are updated separately by `r_ij^(k+1) = W_r^k r_ij^k`, preserving a distinct semantic space for relations.
- In experiments, RGC outperforms GCN, GAT, R-GCN, and CompGCN on OpenEA benchmarks under matched settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-rhgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-rhgn]].
