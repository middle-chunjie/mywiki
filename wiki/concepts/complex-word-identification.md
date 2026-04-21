---
type: concept
title: Complex Word Identification
slug: complex-word-identification
date: 2026-04-20
updated: 2026-04-20
aliases: [CWI, 复杂词识别, lexical complexity prediction]
tags: [nlp, text-simplification, lexical-complexity]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Complex Word Identification** (复杂词识别) — the NLP task of predicting whether a word or phrase in a given sentence context requires simplification, either as a binary classification or a continuous complexity score.

## Key Points

- CWI is highly context-dependent: the same word can be simple in one sentence and complex in another, requiring sentence-level feature extraction rather than lexicon lookup alone.
- Datasets are scarce and domain-specific; CompLex LCP 2021 covers biblical, biomedical, and parliamentary domains in English; CWI 2018 covers English (News, WikiNews, Wikipedia), German, Spanish, and French.
- Modern approaches use pretrained Transformer encoders (RoBERTa, XLM-RoBERTa) to extract contextual features, combined with character-level representations of the target token.
- Domain adaptation via gradient reversal can smooth cross-domain and cross-lingual distributional gaps, improving Pearson correlation over vanilla fine-tuning.
- CWI is evaluated via Pearson Correlation Coefficient (Pearson) and Mean Absolute Error (MAE) when framed as a regression task predicting a continuous complexity score.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
