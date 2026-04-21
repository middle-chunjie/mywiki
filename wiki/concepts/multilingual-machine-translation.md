---
type: concept
title: Multilingual Machine Translation
slug: multilingual-machine-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-NMT, multilingual NMT, 多语种机器翻译]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Machine Translation** (多语种机器翻译) — machine translation with a single model that supports multiple language pairs through shared parameters, shared representations, or both.

## Key Points

- The survey treats multilingual NMT as one of the most promising low-resource strategies because it transfers signal across languages via shared representations.
- It distinguishes one-to-many, many-to-one, and many-to-many translation settings, plus universal versus per-language encoder-decoder parameterizations.
- The paper notes that multilingual models often outperform bilingual baselines for low-resource pairs, especially when the number of languages is moderate and related.
- It also highlights the main failure modes: data imbalance, noisy corpora, insufficient model capacity, and weak overlap in input representations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ranathunga-2021-neural-2106-15115]]
- [[dabre-2021-survey]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ranathunga-2021-neural-2106-15115]].
