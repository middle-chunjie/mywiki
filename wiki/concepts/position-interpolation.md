---
type: concept
title: Position Interpolation
slug: position-interpolation
date: 2026-04-20
updated: 2026-04-20
aliases: [PI, positional interpolation, linear position interpolation]
tags: [long-context, position-encoding, context-length-extension, rope]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Position Interpolation** (位置插值, PI) — A fine-tuning-free (or fine-tuning-assisted) method for extending the context window of RoPE-based LLMs by linearly scaling down position indices to fit within the model's trained range, rather than extrapolating to unseen positions.

## Key Points

- For a model trained on context length `L`, Position Interpolation scales all position indices by `L / L'` when processing a longer sequence of length `L'`, so all positions remain within the range `[0, L]` seen during training.
- Avoids "out-of-distribution" positional values that cause catastrophic performance degradation in RoPE models, at the cost of reduced position resolution (position distinguishability decreases as interpolation compresses a wider range into fewer values).
- Can be applied at inference time without any fine-tuning, but benefits significantly from a short period of continued fine-tuning on sequences of length `L'` to recover perplexity lost from reduced position resolution.
- Proposed by Chen et al. (2023) as a simple extension of RoPE (Rotary Position Embedding); benchmarked against NTK-Aware Scaling and YaRN as a competing approach.
- Performance degrades at very long contexts (e.g., >32K) compared to methods that address the resolution-loss problem more directly (YaRN, NTK) or avoid positional encoding issues altogether (Activation Beacon's plug-in approach).

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-long-2401-03462]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-long-2401-03462]].
