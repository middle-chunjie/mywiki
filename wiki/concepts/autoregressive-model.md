---
type: concept
title: Autoregressive Model
slug: autoregressive-model
date: 2026-04-20
updated: 2026-04-20
aliases: [自回归模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Autoregressive Model** (自回归模型) — a model that summarizes past observations or latents in sequential order so the next or future state can be predicted from preceding context.

## Key Points

- In CPC, the autoregressive component `g_ar` converts latent history `z_{\le t}` into the context representation `c_t`.
- The paper uses GRUs for speech and language, a PixelCNN-style network for images, and an LSTM-based RL agent backbone for control.
- This context model is what lets CPC target longer-horizon structure rather than purely local correlations.
- The downstream representation can be either local latent `z_t` or contextual state `c_t`, depending on task needs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]
- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
