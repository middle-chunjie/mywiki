---
type: concept
title: Model Scaling
slug: model-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [scaling across parameters, 模型扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model Scaling** (模型扩展) — the systematic study of how model behavior changes as parameter count increases under otherwise comparable training conditions.

## Key Points

- Pythia spans `8` parameter scales from `70M` to `12B`, enabling cross-scale analysis under controlled settings.
- The suite keeps data ordering and major architectural decisions consistent so scale is the primary changing variable.
- The paper reports that term-frequency correlations emerge mainly for models of `2.8B` parameters or larger.
- It also finds that larger models show stronger effects in the gender-bias intervention study.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
