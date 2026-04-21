---
type: concept
title: Linear Probing
slug: linear-probing
date: 2026-04-20
updated: 2026-04-20
aliases: [linear probe, 线性探测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Linear Probing** (线性探测) — an adaptation strategy that freezes a pretrained representation model and trains only a shallow linear classifier or output head on top of it.

## Key Points

- The paper treats linear probing as a low-memory baseline against which MeZO is compared on both masked-LM and autoregressive-LM tasks.
- On several tasks, MeZO substantially outperforms linear probing, indicating that forward-only parameter updates can recover adaptation capacity that frozen-feature baselines miss.
- For generation and multiple-choice settings where a plain linear classifier is ill-suited, the paper switches to head tuning rather than a strict feature-only probe.
- The experiments also test LP-MeZO and find it is unstable as a recipe: it can help in some cases but can also hurt performance severely.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[malladi-2024-finetuning-2305-17333]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[malladi-2024-finetuning-2305-17333]].
