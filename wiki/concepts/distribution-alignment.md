---
type: concept
title: Distribution Alignment
slug: distribution-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [分布对齐, feature distribution alignment]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Distribution Alignment** (分布对齐) — the adjustment of one representation so that its statistics better match the distribution expected by another computation path or model component.

## Key Points

- ReFIR introduces distribution alignment because direct insertion of fused target-reference features would shift the original denoising distribution of the target chain.
- The paper implements this alignment with `AdaIN(O_fuse, O_intra)`, preserving the target chain's original mean and variance statistics.
- Ablation results show removing distribution alignment worsens performance, including a `4.36` FID drop on CUFED5 for the representative SUPIR setting.
- The module is important because the retrieved reference and the low-quality query differ in both image quality and content distribution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2024-refir-2410-05601]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2024-refir-2410-05601]].
