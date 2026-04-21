---
type: concept
title: Fill-in-the-Middle
slug: fill-in-the-middle
date: 2026-04-20
updated: 2026-04-20
aliases: [FIM, fill in the middle, 中间填充]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Fill-in-the-Middle** (中间填充) — an autoregressive generation setup in which a model predicts a missing span while conditioning on both a left prefix and a right suffix.

## Key Points

- The paper teaches FIM to decoder-only models by reordering each document into prefix, suffix, and middle segments separated by sentinel tokens.
- A central empirical claim is that FIM can be added during pretraining without hurting standard left-to-right capability when the FIM rate stays below `100%`.
- The authors recommend joint PSM/SPM training, context-level augmentation, and character-level span selection as a practical default recipe.
- The work frames FIM as directly useful for coding-assistant behaviors such as import insertion, docstring completion, and partial function editing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
