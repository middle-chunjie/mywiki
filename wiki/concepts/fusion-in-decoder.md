---
type: concept
title: Fusion-in-Decoder
slug: fusion-in-decoder
date: 2026-04-20
updated: 2026-04-20
aliases: [FID, fusion in decoder]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Fusion-in-Decoder** — a retrieval-augmented generation architecture that encodes each retrieved document separately and fuses their evidence in the decoder during token generation.

## Key Points

- [[li-2024-building]] adopts FID in CONAN-G so multiple retrieved code documents can be used without simply concatenating them into one long encoder input.
- The paper writes generation as `` `P(t_j | q, D, t_<j) = FID(q, D, t_<j)` `` and encodes each retrieved document with `` `CodeT5-Encoder(d^i ⊕ q)` `` before decoder-side fusion.
- In the ablation study, removing FID drops CgCSN-Python from `32.9` to `27.5` BLEU and CgCSN-Java from `37.7` to `32.0`, indicating that decoder-side multi-document fusion matters beyond retrieval alone.
- The authors argue FID helps overcome encoder input-length limits and reduces the impact of noisy retrieved passages by letting the decoder weight evidence across documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-building]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-building]].
