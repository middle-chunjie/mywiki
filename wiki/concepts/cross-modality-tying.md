---
type: concept
title: Cross-Modality Tying
slug: cross-modality-tying
date: 2026-04-20
updated: 2026-04-20
aliases: [CMT, 跨模态绑定]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Modality Tying** (跨模态绑定) — a parameter-sharing strategy that calibrates modality-specific parameters from a shared factor space so different modalities remain better aligned during adaptation.

## Key Points

- MV-Adapter shares a factor `` `f_C` `` between video and text branches and projects it with modality-specific matrices to generate calibration vectors.
- These calibration vectors rescale the downsample weights in both modalities rather than forcing explicit pairwise fusion.
- The design preserves the efficiency benefits of independent video and text encoding while still improving alignment.
- On MSR-VTT, adding CMT improves `R@Sum` from `200.9/204.4` to `202.1/205.9` for text-to-video/video-to-text retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-mvadapter-2301-07868]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-mvadapter-2301-07868]].
