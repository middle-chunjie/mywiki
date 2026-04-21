---
type: concept
title: Predictive Coding
slug: predictive-coding
date: 2026-04-20
updated: 2026-04-20
aliases: [预测编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Predictive Coding** (预测编码) — the idea of learning or compressing representations by predicting missing or future information from context rather than directly reconstructing every observed detail.

## Key Points

- The paper positions CPC as a modern predictive-coding method operating in latent space instead of raw signal space.
- Its motivation is that distant-future prediction emphasizes slow, globally shared factors such as phonetic structure, objects, or narrative semantics.
- CPC inherits predictive-coding intuition from signal processing and neuroscience, but uses neural encoders plus a contrastive objective for tractable training.
- The approach explicitly tries to discard local noise and preserve information that is useful for future prediction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
