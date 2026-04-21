---
type: concept
title: Domain-Adaptive Pretraining
slug: domain-adaptive-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [continued pretraining, domain-specific pretraining, DAPT]
tags: [pretraining, transfer-learning, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Domain-Adaptive Pretraining** (领域自适应预训练) — the practice of continuing the pretraining of a pretrained language model on a domain-specific corpus using the same (or similar) self-supervised objective, to improve performance on downstream tasks in that domain.

## Key Points

- Popularized by Gururangan et al. (2020) "Don't Stop Pretraining," which showed consistent gains from continued MLM on domain-specific text.
- [[zhou-2023-codebertscore-2302-05527]] applies domain-adaptive pretraining to CodeBERT: they continue MLM pretraining on language-specific code corpora (Python, Java, C++, C, JavaScript from CodeParrot) for `1,000,000` steps each with batch size `32`, learning rate `5e-5` decayed to `3e-5`.
- Even after `1M` steps, no model completes a full epoch of the CodeParrot corpus, confirming the scale of the training data.
- The resulting language-specific models consistently outperform the base CodeBERT-base on most HumanEval correlation metrics, though base CodeBERT is competitive.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-codebertscore-2302-05527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-codebertscore-2302-05527]].
