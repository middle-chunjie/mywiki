---
type: concept
title: Audio Noise Distillation
slug: audio-noise-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [音频噪声蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Audio Noise Distillation** (音频噪声蒸馏) — a training strategy that transfers noise information from audio representations into a language-space embedding so that denoising can be performed in text-conditioned LLM adaptation.

## Key Points

- The paper treats noisy Whisper embeddings as positive samples and clean-audio embeddings as negative samples when estimating mutual information with the language-space noise embedding.
- Distillation is needed because the raw language-space embedding does not always separate all noise types clearly.
- A learnable projection `T_omega` modulates the language-space embedding so that it carries more information about real acoustic noise.
- The adapter tuner and LLM are jointly optimized with a GER objective plus a mutual-information term weighted by `lambda = 0.5`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2401-10446]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2401-10446]].
