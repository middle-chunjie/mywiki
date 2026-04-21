---
type: concept
title: Denoising Autoencoding
slug: denoising-autoencoding
date: 2026-04-20
updated: 2026-04-20
aliases: [denoising autoencoder, denoising pre-training]
tags: [pretraining, self-supervision]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Denoising Autoencoding** (去噪自编码) — a self-supervised learning objective that corrupts an input and trains a model to reconstruct the original sequence, forcing it to capture syntax and semantics.

## Key Points

- In [[ahmad-2021-unified]], PLBART is pretrained by reconstructing code or text `x` from a corrupted version `f(x)`.
- The corruption pipeline combines token masking, token deletion, and span infilling with `35%` token corruption and span lengths from `` `Poisson(λ = 3.5)` ``.
- The paper uses denoising instead of masked-only objectives so that both encoder and decoder receive meaningful pretraining.
- The authors attribute PLBART's strong low-resource transfer partly to the syntactic and semantic priors learned through denoising reconstruction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]
- [[tehranijamsaz-2024-coderosetta-2410-20527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
