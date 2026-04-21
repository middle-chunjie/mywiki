---
type: concept
title: Autoregressive Language Model
slug: autoregressive-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [AR language model, causal language model, 自回归语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Autoregressive Language Model** (自回归语言模型) — a language model that predicts each next token conditioned only on previously generated tokens in a chosen ordering.

## Key Points

- The paper treats standard decoder-only LMs as the dominant large-scale baseline and asks whether they can acquire infilling without architectural changes.
- FIM training preserves the local next-token objective because the prefix, suffix, and middle are still modeled autoregressively after reordering.
- The strongest empirical claim is that AR capability, measured by loss and downstream benchmarks, remains essentially unchanged for FIM rates up to `0.9`.
- The paper contrasts cheap AR-to-FIM joint pretraining with comparatively expensive post hoc FIM finetuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
