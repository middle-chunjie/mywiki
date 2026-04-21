---
type: concept
title: Cosine Learning Rate Decay
slug: cosine-learning-rate-decay
date: 2026-04-20
updated: 2026-04-20
aliases: [余弦学习率衰减]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cosine Learning Rate Decay** (余弦学习率衰减) — a scheduling strategy that decreases the learning rate along a cosine curve, often paired with warmup for stable early optimization.

## Key Points

- The paper replaces the baseline step decay schedule with CLIP-style cosine decay plus warmup.
- Training uses `` `10` `` warmup epochs and a base learning rate of `` `1e-3` `` under linear scaling with batch size.
- This scheduler change improves the strong baseline from `` `48.0%` `` to `` `48.5%` `` Top1 on Objaverse-LVIS and from `` `53.5%` `` to `` `54.1%` `` on ScanObjectNN.
- The authors argue the scheduler is part of a broader recipe transfer from strong image-language pre-training to language-image-3D contrastive learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
