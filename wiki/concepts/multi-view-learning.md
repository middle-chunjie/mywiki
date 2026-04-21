---
type: concept
title: Multi-View Learning
slug: multi-view-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [多视图学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-View Learning** (多视图学习) — a learning setup that combines observations from multiple views of the same object to build a more complete representation than any single view can provide.

## Key Points

- MixCon3D renders `` `12` `` canonical object views and studies how many views should be fused during training.
- The method replaces the single sampled image with a fused image representation `` `z^I = g^MV({z_(i,j)^I})` `` built from multiple rendered views.
- View-pooling is the default fusion function because it improves Objaverse-LVIS and ScanObjectNN more reliably than max pooling or an added FC layer.
- Increasing views helps in-distribution Objaverse-LVIS performance up to `` `53.2%` `` Top1 at `` `12` `` views, but too many views reduce out-of-distribution ScanObjectNN accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
