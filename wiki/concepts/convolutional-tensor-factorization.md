---
type: concept
title: Convolutional Tensor Factorization
slug: convolutional-tensor-factorization
date: 2026-04-20
updated: 2026-04-20
aliases: [CTF]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Convolutional Tensor Factorization** — a hybrid model that regularizes tensor factorization with patent representations produced by a convolutional content encoder.

## Key Points

- CTF couples a three-way lawsuit tensor with NCNN so collaborative structure and patent content are learned jointly.
- The patent latent vector is tied to the content encoder output through `P_k = O_k + epsilon_k`, which makes the two components interact directly.
- The final score sums three pairwise latent interactions: plaintiff-defendant, plaintiff-patent, and defendant-patent.
- The paper uses CTF to rank candidate patents for a given company pair instead of predicting litigation in isolation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
