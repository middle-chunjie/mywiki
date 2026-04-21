---
type: concept
title: Lossless Compression
slug: lossless-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [无损压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Lossless Compression** (无损压缩) — compression that preserves the original data exactly or, in some systems writing float32 outputs, to a fidelity intended to be operationally indistinguishable from the source precision.

## Key Points

- The paper is motivated by the gap between high-ratio lossy embedding compression and low-ratio exact floating-point baselines.
- Its spherical pipeline is compared primarily against lossless floating-point compressors such as transpose+shuffle+zstd and ZipNN.
- The method is not bit-exact because of transcendental transforms, but the reported reconstruction error stays below float32 machine epsilon.
- The work targets storage, transfer, and archival scenarios where high-fidelity recovery matters more than extreme compression ratios.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
