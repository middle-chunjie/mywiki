---
type: concept
title: Heavy Hitter
slug: heavy-hitter
date: 2026-04-20
updated: 2026-04-20
aliases: [Heavy Hitters, H2, heavy hitter tokens, 重要词元]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Heavy Hitter** (重要词元) — a small subset of tokens whose accumulated attention scores across all decoding steps are disproportionately large, making them critical for maintaining LLM generation quality; identified empirically as following a power-law distribution.

## Key Points

- Heavy hitters (H2) are defined by their position in the power-law tail of cumulative attention score distributions across all attention heads and layers of pre-trained LLMs.
- H2 tokens correlate strongly with high-frequency co-occurrence of words in the training corpus, suggesting their importance is tied to statistical salience rather than positional proximity.
- Removing heavy hitters from the KV cache causes severe accuracy collapse (e.g., COPA accuracy drops from 85% to ~48% on OPT-30B at 20% cache budget) while removing only recent tokens degrades much less.
- Local statistics (accumulated scores of preceding tokens only) approximate global statistics (including future tokens) well enough for practical use, enabling a deployment-friendly greedy eviction.
- The H2 property generalizes across OPT (6.7B–175B), LLaMA-7B/13B, and GPT-NeoX-20B, suggesting it is an intrinsic property of dense transformer pretraining rather than a model-specific artifact.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-ho-2306-14048]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-ho-2306-14048]].
