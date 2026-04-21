---
type: concept
title: Multilingual Pre-training
slug: multilingual-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [multilingual pretraining, joint code-text pretraining]
tags: [pretraining, transfer-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Pre-training** (多语言预训练) — pre-training a model on multiple languages or modalities so it can learn shared representations that transfer across tasks and domains.

## Key Points

- [[ahmad-2021-unified]] trains PLBART on Java, Python, and English developer text in one shared model.
- Language identifiers such as `` `<java>` ``, `` `<python>` ``, and `` `<En>` `` are injected so the model can condition decoding on the target language.
- To avoid overwhelming the model with programming-language data, the paper smooths sampling probabilities with `α = 0.3`.
- The reported translation and unseen-language classification results are used as evidence that joint multilingual pre-training improves transfer beyond the pretraining languages.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
