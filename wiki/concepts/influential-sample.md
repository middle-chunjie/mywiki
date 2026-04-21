---
type: concept
title: Influential Sample
slug: influential-sample
date: 2026-04-20
updated: 2026-04-20
aliases: [影响样本, highly-connected sample, influential negative]
tags: [contrastive-learning, negative-sampling, representation-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Influential Sample** (影响样本) — in contrastive learning, a sample that has high average cosine similarity (connectivity) to many other samples in a queue, indicating it likely belongs to a dense semantic region; such samples are candidates to be pruned from the negative set to prevent semantic collision.

## Key Points

- Connectivity `C(x_i)` is computed over a momentum queue `Q` of size `M`: `C(x_i) = (1/M) Σ_j cos(x_i, x_j)`.
- Samples with `C(x_i) > γ` (threshold `γ = 0.9` in CrossCLR) form the influential set `I_x`.
- Two mechanisms exploit connectivity: (1) **negative set pruning** — influential samples are excluded from the negative set to avoid treating semantically similar items as negatives; (2) **proximity weighting** — loss weight `w(x_i) = exp(C(x_i) / κ)` up-weights influential samples and down-weights outliers.
- Influential samples tend to reside at the center of semantic clusters or act as bridges between clusters; they are the most likely source of false negatives in large minibatches.
- The concept generalizes across modality pairs: CrossCLR computes separate connectivity scores and influential sets for each modality `A` and `B`.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zolfaghari-2021-crossclr-2109-14910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zolfaghari-2021-crossclr-2109-14910]].
