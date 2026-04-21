---
type: concept
title: Zero-Shot Translation
slug: zero-shot-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [zero-resource translation, 零样本翻译]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Translation** (零样本翻译) — translating between a language pair without direct parallel training data by relying on multilingual training with shared intermediate languages or representations.

## Key Points

- The survey describes zero-shot translation as more direct than pivoting because no pivot language is used at decoding time.
- Early multilingual models achieved zero-shot translation but typically underperformed pivot-based systems.
- The paper attributes part of the gap to mismatched context representations for unseen language pairs.
- It also highlights strong data dependence: zero-shot performance degrades sharply in truly low-resource settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dabre-2021-survey]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dabre-2021-survey]].
