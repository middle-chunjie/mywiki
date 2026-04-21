---
type: concept
title: Bounding Box Quantization
slug: bounding-box-quantization
date: 2026-04-20
updated: 2026-04-20
aliases: [bounding box quantization, 边界框量化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Bounding Box Quantization** (边界框量化) — the conversion of continuous bounding-box coordinates into a small discrete vocabulary so layouts can be generated or processed as text tokens.

## Key Points

- The paper normalizes `xyxy` coordinates into `[0,1]` and quantizes them into `100` bins before LM generation.
- This tokenization choice makes layout prediction compact enough for language-model decoding.
- The authors explicitly compare box tokens against segmentation maps and note that `4` coordinate tokens are far cheaper than `4096` mask tokens for a `64 x 64` map.
- An appendix ablation shows that `100` bins slightly outperform `1000` bins on both skill-based (`51.0%` vs `49.8%`) and open-ended (`68.3%` vs `68.2%`) evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cho-2023-visual-2305-15328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cho-2023-visual-2305-15328]].
