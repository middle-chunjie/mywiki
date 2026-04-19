---
type: concept
title: Positional Encoding
slug: positional-encoding
date: 2026-04-17
updated: 2026-04-17
aliases: [Positional Encoding, 位置编码]
tags: [attention, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-17
---

## Definition

Positional Encoding (位置编码) — a vector added to token embeddings that injects absolute or relative position information into otherwise order-invariant attention-based models.

## Key Points

- The [[transformer]] has no recurrence or convolution, so raw self-attention is permutation-equivariant; positional encoding breaks that symmetry.
- [[vaswani-2017-attention-1706-03762]] uses fixed sinusoidal encodings: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(...)`.
- Wavelengths form a geometric progression from `2π` to `10000·2π`, allowing linear relationships between `PE_pos` and `PE_{pos+k}` for any fixed offset `k` — intended to help the model learn relative-position attention.
- Compared to learned positional embeddings, sinusoidal encoding yields nearly identical BLEU (Table 3, row E); the authors chose sinusoids for the hope of length extrapolation.
- Added (not concatenated) to embeddings at the bottom of both encoder and decoder stacks; `d_model`-dimensional so the two can be summed directly.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] with sinusoidal formulation; note that learned and sinusoidal encodings perform nearly identically in the original experiments.
