---
type: concept
title: Exponential Moving Average
slug: exponential-moving-average
date: 2026-04-20
updated: 2026-04-20
aliases: [EMA, 指数移动平均]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exponential Moving Average** (指数移动平均) — a parameter-smoothing technique that tracks a decayed average of model weights to stabilize optimization and evaluation.

## Key Points

- MixCon3D adds EMA because the authors observe strong fluctuations on ScanObjectNN during pre-training on synthetic data.
- The default decay rate is `` `0.9995` ``, chosen after a sweep over values from `` `0.99` `` to `` `0.99999` ``.
- EMA improves the strong baseline from `` `54.1%` `` to `` `55.6%` `` Top1 on ScanObjectNN and from `` `48.5%` `` to `` `49.8%` `` on Objaverse-LVIS.
- Extremely large decay `` `0.99999` `` is unstable and collapses performance, showing that EMA tuning matters materially in this setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
