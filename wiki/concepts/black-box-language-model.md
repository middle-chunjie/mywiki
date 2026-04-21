---
type: concept
title: Black-Box Language Model
slug: black-box-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [black box language model, 黑盒语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Language Model** (黑盒语言模型) — a language model that exposes only limited inference interfaces such as token probabilities or text outputs while hiding parameters, activations, and training-time modification hooks.

## Key Points

- RePlug formalizes retrieval augmentation for the setting where hidden states and parameter updates are unavailable.
- The paper treats API-served models such as GPT-3 and Codex as primary black-box targets rather than edge cases.
- The black-box constraint rules out methods that require cross-attention changes, LM fine-tuning, or hidden-state indexing.
- The work shows that useful retrieval augmentation can still be implemented by manipulating only the external prompt and aggregating output probabilities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-replug-2301-12652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-replug-2301-12652]].
