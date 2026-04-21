---
type: concept
title: Gaussian Noise
slug: gaussian-noise
date: 2026-04-20
updated: 2026-04-20
aliases: [normal noise, 高斯噪声]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gaussian Noise** (高斯噪声) — additive random perturbation sampled from a normal distribution, often used to test robustness or privacy-utility trade-offs in continuous representations.

## Key Points

- The paper studies a simple defense `φ_noisy(x) = φ(x) + λ · ε` with `ε ~ N(0, 1)` applied directly to text embeddings.
- At `λ = 0.01`, mean BEIR retrieval drops only from `0.302` to `0.296`, while reconstruction BLEU collapses from `80.372` to `10.334`.
- At `λ = 0.1`, both retrieval and reconstruction largely fail, so the defense exposes a steep utility cliff.
- The authors treat this as a sanity-check defense rather than a complete solution because adaptive attackers could train against noisy embeddings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2023-text-2310-06816]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2023-text-2310-06816]].
