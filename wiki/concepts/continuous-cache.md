---
type: concept
title: Continuous Cache
slug: continuous-cache
date: 2026-04-20
updated: 2026-04-20
aliases: [neural cache, continuous neural cache, 连续缓存]
tags: [language-model, memory, cache, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Continuous Cache** (连续缓存) — a non-parametric memory mechanism for language models (Grave et al., 2017) that stores recent hidden representations as a key-value cache and interpolates the standard LM distribution with a probability derived from similarity to cached representations at inference time only.

## Key Points

- Proposed by Grave et al. (2017b) for LSTM-based LMs; the local memory is the set of recent hidden state–token pairs `{(h_j, x_j)}` for `j < t`.
- Effective for capturing local repetitions but argued to be less useful for Transformer models since self-attention already attends to local context.
- Applied only at test time in the original formulation; TRIME demonstrates that explicitly training with local memory in the same spirit yields larger improvements (18.70 → 17.76 vs. 18.70 → 18.26 on WIKITEXT-103).
- Can be extended to long-term memory by enlarging the cache window, though performance gains are smaller than TRIME's batching approach at equivalent context lengths.
- Interpolation form: `P(w|c) = (1-λ) P_LM(w|c) + λ P_cache(w|c)`, where `P_cache` is softmax over similarity scores to cached representations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2022-training-2205-12674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2022-training-2205-12674]].
