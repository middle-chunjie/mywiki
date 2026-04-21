---
type: concept
title: Common Token Bias
slug: common-token-bias
date: 2026-04-20
updated: 2026-04-20
aliases: [pretraining frequency bias, token frequency bias, 常见词偏差]
tags: [bias, few-shot-learning, language-model]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Common Token Bias** (常见词偏差) — the intrinsic preference of language models for generating tokens that are statistically frequent in their pre-training corpus, independent of the downstream task distribution, leading to systematic over-prediction of common answer tokens.

## Key Points

- For the LAMA fact retrieval task, GPT-3 often predicts common entities like "America" even when the correct answer is a rare entity — the model's prior over next tokens is dominated by pre-training frequency.
- For text classification, label names that appear more frequently in web text are systematically over-predicted: on DBPedia 14-way classification, "book" is predicted 11× more often than "artist"; correlation between Google Ngrams frequency and prediction rate is r = 0.67.
- Distinct from [[majority-label-bias]] (which depends on label distribution in the prompt) — common token bias arises from the model's pre-training distribution and persists even in 0-shot settings.
- Explains why the choice of label names (e.g., "Positive"/"Negative" vs. "good"/"bad") matters: rarer label strings will be intrinsically under-predicted.
- Mitigated by [[contextual-calibration]], whose content-free estimate absorbs the pre-training frequency prior.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhao-2021-calibrate-2102-09690]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhao-2021-calibrate-2102-09690]].
