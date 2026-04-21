---
type: concept
title: Language-Space Noise Embedding
slug: language-space-noise-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [语言空间噪声嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language-Space Noise Embedding** (语言空间噪声嵌入) — a representation extracted from textual diversity in ASR `N`-best hypotheses to encode the noise condition of the source speech without directly conditioning on audio features.

## Key Points

- The embedding is motivated by the observation that harder noise conditions make beam-search decoding more uncertain and therefore produce more diverse hypotheses.
- The paper combines an utterance-level component built from pairwise SBERT sentence-vector differences with a token-level component built from edit-style aligned token differences.
- The final representation is `E_LN = [E_LN^utt; E_LN^tok]`, and the token-level part contributes more to performance than the utterance-level part.
- RobustGER injects this embedding into LLaMA-Adapter by subtracting a projected version from the adapter prompt to indicate denoising.
- Visualizations in the paper show that this embedding partially separates noise types even before audio distillation and improves further after distillation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2401-10446]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2401-10446]].
